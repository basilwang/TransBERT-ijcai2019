import os
import pickle
import numpy
from whim_common.data.coref import Entity

from cam.whim.entity_narrative.chains.document import predicate_relation, Event, predicate_relation_with_transitivity
from cam.whim.entity_narrative.models.base.coherence import CoherenceScorer
from cam.whim.entity_narrative.models.base.predict import NarrativeChainPredictor
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel, magnitude
from cam.whim.entity_narrative.models.word2vec.shell import Word2VecShell
from cam.whim.entity_narrative.models.word2vec.train import Word2VecTrainer
from whim_common.utils.base import str_to_bool
from whim_common.utils.vectors import vector_cosine_similarity
from gensim.models.word2vec import Word2Vec


class Word2VecNarrativeChainModel(VectorSpaceNarrativeChainModel, NarrativeChainPredictor, CoherenceScorer):
    """
    Train embeddings for event words on event chain data using word2vec.
    This is the model called Word2vec-pred+arg in the paper (or Word2vec-pred, with pred_only=T).

    """
    MODEL_TYPE_NAME = "word2vec"
    TRAINER_CLASS = Word2VecTrainer
    SHELL_TYPE = Word2VecShell
    MODEL_OPTIONS = dict(VectorSpaceNarrativeChainModel.MODEL_OPTIONS, **{
        "cosmul": {
            "help": "Use the Omer & Levy cosmul operation to compare the context to a candidate, instead of just "
                    "summing the projections",
            "type": str_to_bool,
            "default": False,
        },
        "arg_vectors": {
            "help": "Path to the binary vector file of Mikolov vectors to use as representations of argument words",
        },
        "arg_sum": {
            "help": "Sum up the arg vectors and concatenate the result with the verb vector, instead of concatenating "
                    "everything",
            "type": str_to_bool,
        },
        "pred_only": {
            "help": "Only project predicates, not args, even if the model was trained to use both",
            "type": str_to_bool,
            "default": False,
        },
    })
    DEFAULT_METRIC = "euclid"

    def __init__(self, word2vec, arg_word2vec=None, with_args=False, predicative_adjectives=False, transitivity=False,
                 **kwargs):
        super(Word2VecNarrativeChainModel, self).__init__(**kwargs)
        self.transitivity = transitivity
        self.predicative_adjectives = predicative_adjectives
        self.with_args = with_args
        self.word2vec = word2vec
        self.word2vec.init_sims()

        self.arg_word2vec = arg_word2vec

    @property
    def vector_size(self):
        if self.arg_word2vec is not None:
            if self.model_options["arg_sum"]:
                return self.word2vec.layer1_size + self.arg_word2vec.layer1_size
            else:
                return self.word2vec.layer1_size + 3*self.arg_word2vec.layer1_size
        else:
            return self.word2vec.layer1_size

    @property
    def word_generator(self):
        if self.with_args and not self.model_options["pred_only"]:
            gen = Word2VecNarrativeChainModel.extract_chain_word_lists_with_args
        else:
            gen = Word2VecNarrativeChainModel.extract_chain_word_lists
        return lambda chains: gen(chains, predicative_adjectives=self.predicative_adjectives,
                                  transitivity=self.transitivity)

    def project_chains(self, chains, progress=False):
        vectors = numpy.zeros((len(chains), self.vector_size), dtype=numpy.float64)
        for chain_num, chain_word_vecs in enumerate(self.get_chain_word_vectors(chains)):
            # Just sum up the vectors to get the chain repr
            vectors[chain_num] += numpy.sum(chain_word_vecs, axis=0)
        return vectors

    def get_chain_word_vectors(self, chains):
        if self.arg_word2vec is not None:
            chain_args = list(Word2VecNarrativeChainModel.extract_chain_arg_lists(chains))
        else:
            chain_args = None

        for chain_num, chain_words in enumerate(self.word_generator(chains)):
            chain_vectors = []
            # Get vector representations of each of the words (events) in the chain
            # Skip words that we've not learned a representation for
            for word_num, word in enumerate(chain_words):
                #basilwang 2020-10-05 should use word2vec.wv.vocab
                #if word in self.word2vec.vocab:
                if word in self.word2vec.wv.vocab:
                    vector = self.word2vec.wv.syn0norm[self.word2vec.wv.vocab[word].index]

                    # If we've got argument vectors, concatenate them
                    if chain_args is not None:
                        event_args = chain_args[chain_num][word_num]
                        arg_vectors = []
                        for arg in event_args:
                            if arg is None or arg not in self.arg_word2vec.vocab:
                                # Use a zero-vector
                                arg_vec = numpy.zeros(self.arg_word2vec.layer1_size, dtype=numpy.float64)
                            else:
                                arg_vec = self.arg_word2vec.wv.syn0norm[self.arg_word2vec.vocab[arg].index]
                            arg_vectors.append(arg_vec)

                        if self.model_options["arg_sum"]:
                            # Sum the arg vectors instead of concatenating
                            args_vector = sum(arg_vectors)
                        else:
                            args_vector = numpy.concatenate(tuple(arg_vectors))
                        vector = numpy.concatenate((vector, args_vector))
                    chain_vectors.append(vector)
            yield chain_vectors

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        """
        Vector space models have a common implementation of the scoring function that makes use of the
        projection defined by the specific model.

        """
        if not self.model_options["cosmul"]:
            # Do some custom scoring if we're using the cosmul comparison
            # Otherwise just do the usual projections
            return super(Word2VecNarrativeChainModel, self)._score_choices(entities, contexts, choice_lists,
                                                                           progress=progress)

        # Don't show progress, it's too fast to be worth it!
        scores = numpy.zeros((len(contexts), max(len(choices) for choices in choice_lists)), dtype=numpy.float64)

        # Perform projection of context chains into vector space, keeping the projections of words separate
        for i, (entity, choice_list, word_vectors) in enumerate(zip(entities,
                                                                    choice_lists,
                                                                    self.get_chain_word_vectors(
                                                                            zip(entities, contexts)))):
            # Project each of the choices into the vector space, as if it's a whole chain
            choice_projections = self.project_events([(entity, choice) for choice in choice_list])
            # L2 normalize the projections of the candidates
            choice_projections = numpy.ma.divide(
                choice_projections,
                numpy.sqrt((choice_projections ** 2).sum(axis=-1))[..., numpy.newaxis]
            )
            choice_projections = choice_projections.data
            # Perform a comparison between each of the candidate projections and the corresponding context projection
            # Equation 4 of Levy & Goldberg "Linguistic Regularities...", distances shifted to [0,1] per footnote 7
            # Like Gensim's implementation, except we have no negative terms
            pos_dists = [((1. + numpy.dot(choice_projections, term)) / 2.) for term in word_vectors]
            scores[i] = numpy.prod(pos_dists, axis=0)

        return scores

    def predict_next_event(self, entity, context_events):
        # Project the context into the vector space
        context_vector = numpy.sum(
            list(self.get_chain_word_vectors([(entity, context_events)]))[0], axis=0
        )
        if len(context_vector.shape) == 0:
            # No known features in the input: can't make predictions
            raise StopIteration
        context_vector /= magnitude(context_vector)

        # Look for nearest neighbours among the predicates only (for now)
        sim = numpy.dot(self.word2vec.syn0norm, context_vector)
        for index in reversed(sim.argsort()):
            word = self.word2vec.index2word[index]
            # Skip arg words
            if not word.startswith("arg:"):
                event_kwargs = {}

                if self.transitivity:
                    # Split off the transitivity marker
                    word, __, trans = word.partition("[")
                    event_kwargs["transitivity"] = trans[:-1]

                if word.startswith("adj:"):
                    # Special case to handle the correct formatting of predicative adjectives
                    verb, __, adjective = word[4:].partition(":")
                    event_kwargs["subject"] = entity
                    event_kwargs["object"] = adjective
                    event_kwargs["type"] = "predicative"
                else:
                    verb, __, dep = word.partition(":")

                    # Put the entity in the right position
                    if dep == "subj":
                        event_kwargs["subject"] = entity
                    elif dep == "obj":
                        event_kwargs["object"] = entity
                    elif dep.startswith("prep_"):
                        event_kwargs["iobject"] = (dep[5:], entity)
                    else:
                        raise ValueError("could not split up predicate from vocabulary: %s" % word)

                    # Handle predicative adjectives that weren't correctly encoded
                    if verb == "be" or verb == "become":
                        event_kwargs["type"] = "predicative"

                yield Event(verb, verb, 0, **event_kwargs), float(sim[index])

    @staticmethod
    def extract_chain_word_lists(chains, predicative_adjectives=False, transitivity=False):
        """
        Chains should be given as a list of (entity, event list) pairs.
        Produces a list of string representations of the events of each chain, which will be treated as
        the words of a sentence.

        """
        for entity, chain in chains:
            yield [Word2VecNarrativeChainModel.get_event_repr(entity, ev,
                                                              predicative_adjectives=predicative_adjectives,
                                                              transitivity=transitivity)
                   for ev in chain]

    @staticmethod
    def extract_chain_word_lists_with_args(chains, predicative_adjectives=False, transitivity=False):
        """
        Chains should be given as a list of (entity, event list) pairs.
        Produces a list of string representations of the events of each chain, which will be treated as
        the words of a sentence.

        """
        for entity, chain in chains:
            yield sum(
                [[Word2VecNarrativeChainModel.get_event_repr(entity, ev,
                                                             predicative_adjectives=predicative_adjectives,
                                                             transitivity=transitivity)] +
                 ["arg:%s" % arg for arg in event_args if arg is not None] for ev, event_args in
                 zip(chain, Word2VecNarrativeChainModel.extract_chain_arg_list(entity, chain))], [])

    @staticmethod
    def extract_chain_arg_lists(chains):
        for entity, chain in chains:
            yield Word2VecNarrativeChainModel.extract_chain_arg_list(entity, chain)

    @staticmethod
    def extract_chain_arg_list(entity, events):
        chain_features = []
        for event in events:
            features = []
            # A word for each argument, if it's filled
            for arg_num, arg in enumerate([event.subject, event.object,
                                           event.iobject[1] if event.iobject is not None else None]):
                if arg is None:
                    # Arg is not filled: use None to indicate this
                    features.append(None)
                elif type(arg) is Entity:
                    # Try getting a headword for the entity
                    arg_word = arg.get_head_word()
                    if arg_word is None:
                        # Treat as if there was no entity: we have no other way to get a representation
                        features.append(None)
                    else:
                        features.append(arg_word)
                else:
                    # Otherwise it must be a string -- the headword of an NP
                    features.append(str(arg).lower())
            chain_features.append(features)
        return chain_features

    @staticmethod
    def get_event_repr(entity, event, predicative_adjectives=False, transitivity=False):
        if transitivity:
            return predicate_relation_with_transitivity(entity, event, handle_predicative=predicative_adjectives)
        else:
            return predicate_relation(entity, event, handle_predicative=predicative_adjectives)

    def chain_coherence(self, entity, events):
        """ Implements abstract method of CoherenceScorer """
        # Project each event into the vector space
        event_vectors = self.get_chain_word_vectors([(entity, events)]).next()
        # Compute average pairwise similarity of events
        sims = []
        for i, event0 in enumerate(event_vectors[:-1]):
            for event1 in event_vectors[i+1:]:
                # Ignore identical vectors: repeated events do not constitute coherence!
                if numpy.all(event0 != event1):
                    sims.append(vector_cosine_similarity(event0, event1))
        if len(sims):
            return sum(sims, 0.) / len(sims)
        else:
            return 0.

    @property
    def _description(self):
        return """\
Mikolov Word2Vec-based model
Vector size:        %d
Predicates:         %d
Arg vectors:        %s
Args included:      %s
Predicative adjs:   %s
Transitivity markers: %s""" % (
            self.word2vec.layer1_size,
            len(self.word2vec.wv.vocab),
            self.arg_word2vec.layer1_size if self.arg_word2vec is not None else "none",
            self.with_args,
            self.predicative_adjectives,
            self.transitivity,
        )

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        word2vec = Word2Vec.load(os.path.join(directory, "word2vec"))

        # Load word2vec vectors to use for the argument representation if one has been given
        model_options = kwargs.get('model_options', {}) or {}
        if "arg_vectors" in model_options:
            # Load the binary file
            arg_word2vec = Word2Vec.load_word2vec_format(model_options["arg_vectors"], binary=True)
        else:
            arg_word2vec = None

        params = {}
        if os.path.exists(os.path.join(directory, "params")):
            with open(os.path.join(directory, "params"), "r") as f:
                params = pickle.load(f)
        kwargs.update(params)

        return Word2VecNarrativeChainModel(word2vec, arg_word2vec=arg_word2vec, **kwargs)

    def _save_to_directory(self, directory, model_name, human_name=None):
        self.word2vec.save(os.path.join(directory, "word2vec"))
        with open(os.path.join(directory, "params"), "w") as f:
            pickle.dump({
                "with_args": self.with_args,
                "predicative_adjectives": self.predicative_adjectives,
                "transitivity": self.transitivity,
            }, f)
