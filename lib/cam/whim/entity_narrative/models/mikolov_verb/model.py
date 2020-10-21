import numpy
from whim_common.data.coref import Entity

from cam.whim.entity_narrative.models.base.model import ModelLoadError
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel
from whim_common.utils.base import str_to_bool
from gensim.models.word2vec import Word2Vec


class MikolovVerbNarrativeChainModel(VectorSpaceNarrativeChainModel):
    """
    Unlike Word2VecVerbNarrativeChainModel, which uses word2vec to infer a model of predicate similarity,
    this just uses verb vectors trained with word2vec (for example, those ready-made that can be downloaded
    from the word2vec page).

    Note that the model name is ignored with this model, since it doesn't need a trained model, just a bunch
    of vectors. The vector path option, however, is required.

    """
    MODEL_TYPE_NAME = "mikolov-verb"
    MODEL_OPTIONS = dict(VectorSpaceNarrativeChainModel.MODEL_OPTIONS, **{
        "vectors": {
            "help": "Path to the binary vector file",
        },
        "cosmul": {
            "help": "Use the Omer & Levy cosmul operation to compare the context to a candidate, instead of just "
                    "summing the projections",
            "type": str_to_bool,
            "default": False,
        },
        "args": {
            "help": "Project the arg words as well as the verbs to get the vector representations",
            "type": str_to_bool,
            "default": False,
        },
        "concat": {
            "help": "Concatenate the (summed) arg vectors with the verb vectors instead of summing. Only has "
                    "an effect together with --args",
            "type": str_to_bool,
            "default": False,
        },
    })
    DEFAULT_METRIC = "cosine"

    def __init__(self, word2vec, **kwargs):
        super(MikolovVerbNarrativeChainModel, self).__init__(**kwargs)
        self.word2vec = word2vec
        self.word2vec.init_sims()
        self._zero_vec = numpy.zeros(self.word2vec.layer1_size, dtype=numpy.float64)
        
    @property
    def vector_size(self):
        if self.model_options["args"] and self.model_options["concat"]:
            return self.word2vec.layer1_size * 2
        else:
            return self.word2vec.layer1_size

    def project_chains(self, chains, progress=False):
        vectors = numpy.zeros((len(chains), self.vector_size), dtype=numpy.float64)
        for chain_num, chain_word_vecs in enumerate(self.get_chain_word_vectors(chains)):
            # Just sum up the vectors to get the chain repr
            vectors[chain_num] = numpy.sum(chain_word_vecs, axis=0)
        return vectors

    def get_chain_word_vectors(self, chains):
        for chain_num, chain_words in enumerate(self.extract_chain_word_lists(chains)):
            # Get vector representations of each of the words (events) in the chain
            # Skip words that we've not learned a representation for
            # If the event has multiple features, sum up their vector representations
            yield [self._features_to_vector(event_features) for event_features in chain_words]

    def _features_to_vector(self, feats):
        # Get a vector for the predicate (or a zero vector)
        vector = self.word2vec.syn0norm[self.word2vec.vocab[feats[0]].index] \
            if feats[0] in self.word2vec.vocab else self._zero_vec
        if self.model_options["args"]:
            # If requested, also include arg vectors: sum them up
            arg_vector = sum([self.word2vec.syn0norm[self.word2vec.vocab[word].index]
                              for word in feats[1:] if word in self.word2vec.vocab], self._zero_vec)
            if self.model_options["concat"]:
                # Concatenate pred vector with args vector
                return numpy.concatenate((vector, arg_vector))
            else:
                # Just sum the two
                return vector + arg_vector
        else:
            return vector

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        """
        Vector space models have a common implementation of the scoring function that makes use of the
        projection defined by the specific model.

        """
        if not self.model_options["cosmul"]:
            # Do some custom scoring if we're using the cosmul comparison
            # Otherwise just do the usual projections
            return super(MikolovVerbNarrativeChainModel, self)._score_choices(entities, contexts, choice_lists,
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
            choice_projections = choice_projections.filled(0.)
            # Perform a comparison between each of the candidate projections and the corresponding context projection
            # Equation 4 of Levy & Goldberg "Linguistic Regularities...", distances shifted to [0,1] per footnote 7
            # Like Gensim's implementation, except we have no negative terms
            pos_dists = [((1. + numpy.dot(choice_projections, term)) / 2.) for term in word_vectors]
            scores[i] = numpy.prod(pos_dists, axis=0)

        return scores

    def extract_chain_word_lists(self, chains):
        """
        Chains should be given as a list of (entity, event list) pairs.
        Produces a list of string representations of the events of each chain, which will be treated as
        the words of a sentence.

        """
        for entity, chain in chains:
            yield [self.get_event_repr(entity, ev) for ev in chain]

    def get_event_repr(self, entity, event):
        verb = event.verb_lemma.lower().replace("+", "_")

        if self.model_options["args"]:
            # Include the verb as a feature
            features = [verb]
            # A word for each argument, if it's filled
            for arg_num, arg in enumerate([event.subject, event.object,
                                           event.iobject[1] if event.iobject is not None else None]):
                if arg is None:
                    # Arg is not filled: leave this out of the representation
                    continue
                elif type(arg) is Entity:
                    # Try getting a headword for the entity
                    arg_word = arg.get_head_word()
                    if arg_word is None:
                        continue
                else:
                    # Otherwise it must be a string -- the headword of an NP
                    arg_word = str(arg)
                # Mark what arg position this is filling
                features.append(arg_word.lower())
            return features
        else:
            # Just use the verb lemma to represent events: we can look up a vector for this
            return [verb]

    @property
    def _description(self):
        return """\
Mikolov Word2Vec-based model
Vector size:        %d
Total vector size:  %d
Predicates:         %d
Metric:             %s""" % (
            self.word2vec.layer1_size,
            self.vector_size,
            len(self.word2vec.vocab),
            self.metric_name,
        )

    @classmethod
    def load(cls, model_name, **kwargs):
        # Override to ignore model name
        model_options = kwargs.get('model_options', None)
        if model_options is None:
            raise ModelLoadError("word2vec-verb model must be loaded with model options to specify vector path")
        if "vectors" not in model_options:
            raise ModelLoadError("you must specify a path to the binary vector file when loading a word2vec-verb model")

        # Load the binary file
        word2vec = Word2Vec.load_word2vec_format(model_options["vectors"], binary=True)
        return cls(word2vec, **kwargs)

    def save(self, model_name, human_name=None):
        # Override to disable
        raise NotImplementedError("you cannot save a word2vec-verb model")