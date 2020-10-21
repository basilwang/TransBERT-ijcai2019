import numpy
import os
from whim.entity_narrative import predicate_relation
from whim.entity_narrative import NarrativeChainModel
from whim.entity_narrative import BigramTrainer
from whim_common.utils.probability.markov.bigram.model import BigramModel
from gensim.corpora.dictionary import Dictionary


class BigramNarrativeChainModel(NarrativeChainModel):
    MODEL_TYPE_NAME = "bigram"
    MODEL_OPTIONS = dict(NarrativeChainModel.MODEL_OPTIONS, **{})
    TRAINER_CLASS = BigramTrainer

    def __init__(self, bigram_model, dictionary, **kwargs):
        self.dictionary = dictionary
        self.bigram_model = bigram_model
        NarrativeChainModel.__init__(self, **kwargs)

    @property
    def _description(self):
        return """\
Bigram chain prediction model
Unigram backoff threshold: %d
Laplace smoothing:         %.2g""" % (
            self.bigram_model.backoff_threshold,
            self.bigram_model.laplace_smoothing
        )

    @staticmethod
    def chain_to_ids(chain, dictionary, include_unknown=False):
        entity, events = chain
        preds = [predicate_relation(entity, e) for e in events]

        if include_unknown:
            # Put in Nones where we've not seen a predicate
            return [dictionary.token2id[pred] if pred in dictionary.token2id else None for pred in preds]
        else:
            return [dictionary.token2id[pred] for pred in preds if pred in dictionary.token2id]

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        scores = numpy.zeros((len(entities), max(len(l) for l in choice_lists)), dtype=numpy.float64)

        for context_num, (entity, context, choices) in enumerate(zip(entities, contexts, choice_lists)):
            # Get IDs for the context
            ids = BigramNarrativeChainModel.chain_to_ids((entity, context), self.dictionary)

            # Get a probability distribution over the following predicates, averaging over the distributions
            #  conditioned on each context predicate
            if len(ids):
                # Only interested in the last one (Markov model)
                event_dist = numpy.mean([self.bigram_model.probability_dist(context) for context in ids], axis=0)
            else:
                event_dist = self.bigram_model.probability_dist(None)

            # Get a score for each candidate next event
            candidate_ids = BigramNarrativeChainModel.chain_to_ids((entity, choices), self.dictionary, include_unknown=True)

            for i, candidate_id in enumerate(candidate_ids):
                if candidate_id is not None:
                    # Score according to the prob dist over next events
                    # basilwang 2020-10-05 maybe bugs, index 2 is out of bounds for axis 0 with size 1
                    # basilwang 2020-10-07 only one row ATTENTION!!!
                    #scores[context_num, i] = event_dist[candidate_id]
                    scores[context_num, i] = event_dist[0,candidate_id]
        return scores

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        dictionary = Dictionary.load(os.path.join(directory, "dictionary"))
        bigram_model = BigramModel.load(os.path.join(directory, "model"))
        return BigramNarrativeChainModel(bigram_model, dictionary, **kwargs)

    def _save_to_directory(self, directory, model_name, human_name=None):
        self.dictionary.save(os.path.join(directory, "dictionary"))
        self.bigram_model.save(os.path.join(directory, "model"))
