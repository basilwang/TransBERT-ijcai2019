import cPickle as pickle
import numpy
import os
import math

from cam.whim.entity_narrative.chains.document import predicate_relation
from cam.whim.entity_narrative.models.base.model import NarrativeChainModel
from cam.whim.entity_narrative.models.candj.shell import CandjShell
from cam.whim.entity_narrative.models.candj.train import CandjTrainer
from whim_common.utils.base import str_to_bool
from whim_common.utils.files import pickle_attrs


class CandjNarrativeChainModel(NarrativeChainModel):
    MODEL_TYPE_NAME = "candj"
    TRAINER_CLASS = CandjTrainer
    MODEL_OPTIONS = dict(NarrativeChainModel.MODEL_OPTIONS, **{
        "ppmi": {
            "type": str_to_bool,
            "help": "Use positive PMI - i.e. zero out any negative PMIs (default: True)",
            "default": True,
        }
    })
    SHELL_TYPE = CandjShell

    def __init__(self, event_counts, pair_counts, **kwargs):
        super(CandjNarrativeChainModel, self).__init__(**kwargs)
        self.pair_counts = pair_counts
        self.event_counts = event_counts

        self.log_total_pairs = math.log(sum(pair_counts.values()))
        self.log_total_events = math.log(sum(event_counts.values()))

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        scores = numpy.zeros((len(entities), max(len(choices) for choices in choice_lists)), dtype=numpy.float64)

        context_event_lists = list(self.extract_chain_feature_dicts(zip(entities, contexts)))
        choice_event_lists = list(self.extract_chain_feature_dicts(zip(entities, choice_lists)))

        for chain_num, (context, choices) in enumerate(zip(context_event_lists, choice_event_lists)):
            for choice_num, choice in enumerate(choices):
                scores[chain_num, choice_num] = sum([self.pmi(context_event, choice) for context_event in context], 0.)
        return scores

    def pmi(self, event0, event1):
        """
        Compute the PMI between two events.

        If ppmi is given, it overrides the model's option forcing positive (or not) PMI.

        """
        raw_count0 = self.event_counts[event0]
        raw_count1 = self.event_counts[event1]

        # If we've not seen one of the events before, it's unclear what the PMI should be
        if raw_count0 == 0 or raw_count1 == 0:
            return 0.0

        log_event0_count = math.log(raw_count0)
        log_event1_count = math.log(raw_count1)

        pair_count = self.pair_counts[(event0, event1)]
        # If this is a zero-count, we can give the result a bit quicker
        if pair_count == 0:
            # If we've never seen a pair before, don't try to estimate a PMI
            # Even with smoothing, we end up giving very low PMIs to anything that's not been seen,
            # which outweighs everything else
            return 0.0

        pair_count = float(pair_count)
        pmi = math.log(pair_count) - self.log_total_pairs - log_event0_count - log_event1_count + \
              2 * self.log_total_events

        if self.model_options["ppmi"] and pmi <= 0.0:
            return 0.0

        # Now we apply a discounting factor due to Pantel and Ravichandan
        # This is slightly different to P&R, since we want order to not matter
        # NB: We don't implement the sum directly, since when the pairwise co-occurrences are extracted from the chains,
        # we over-count events that occur in long chains. Instead, we use counts[x]
        discount_min = min(raw_count0, raw_count1)
        discount = pair_count / (pair_count + 1.0) * float(discount_min) / float(discount_min + 1.0)
        return discount * pmi

    @staticmethod
    def extract_chain_feature_dicts(chains):
        """
        Chains should be given as a list of (entity, event list) pairs.

        """
        for entity, chain in chains:
            yield [predicate_relation(entity, ev) for ev in chain]

    @property
    def _description(self):
        return """\
Pair counts:      %d
Event vocab:      %d""" % (
            len(self.pair_counts),
            len(self.event_counts)
        )

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        # Kwargs are the same as the attr names
        with open(os.path.join(directory, "model"), 'r') as input_file:
            data = pickle.load(input_file)
        data.update(kwargs)
        return CandjNarrativeChainModel(**data)

    def _save_to_directory(self, directory, model_name, human_name=None):
        pickle_attrs(self,
                     ["event_counts", "pair_counts"],
                     os.path.join(directory, "model"))
