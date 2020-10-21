import cPickle as pickle
import os

import numpy

from cam.whim.entity_narrative.chains.document import predicate_relation
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel
from cam.whim.entity_narrative.models.dist_vecs.shell import DistVecsShell
from cam.whim.entity_narrative.models.dist_vecs.train import DistributionalVectorsTrainer
from whim_common.utils.files import pickle_attrs


class DistributionalVectorsNarrativeChainModel(VectorSpaceNarrativeChainModel):
    """
    Use SVD on co-occurrence counts between predicates to derive a vector representation for each.
    Model referred to in the paper as Dist-vecs.

    """
    MODEL_TYPE_NAME = "dist-vecs"
    TRAINER_CLASS = DistributionalVectorsTrainer
    MODEL_OPTIONS = dict(VectorSpaceNarrativeChainModel.MODEL_OPTIONS, **{})
    SHELL_TYPE = DistVecsShell

    def __init__(self, dictionary, vectors, only_verb=False, adjectives=False, **kwargs):
        super(DistributionalVectorsNarrativeChainModel, self).__init__(**kwargs)
        self.adjectives = adjectives
        self.only_verb = only_verb
        self.vectors = vectors
        self.dictionary = dictionary

    def project_chains(self, chains, additional_features=None, progress=False):
        # Get a feature id list for each chain
        chain_ids = list(self.extract_feature_ids(chains))
        vectors = numpy.zeros((len(chain_ids), self.vectors.shape[1]), dtype=numpy.float64)
        chain_vector = numpy.zeros(self.vectors.shape[1], dtype=numpy.float64)
        for c, chain in enumerate(chain_ids):
            # Filter out any empty chains
            if len(chain):
                # Use these as indices into the event vector matrix
                # Sum the vectors within the chain
                chain_vector[:] = self.vectors[chain, :].sum(axis=0)
            else:
                chain_vector[:] = 0.
            if chain_vector.sum() == 0.:
                # TODO Is this the right thing to do? No, I think
                vectors[c] = 1.
            else:
                vectors[c] = chain_vector
        return vectors

    @staticmethod
    def extract_chain_feature_lists(chains, only_verb=False, adjectives=False):
        """
        Chains should be given as a list of (entity, event list) pairs.

        """
        for entity, chain in chains:
            if only_verb:
                yield [event.verb_lemma for event in chain]
            else:
                yield [predicate_relation(entity, ev, handle_predicative=adjectives) for ev in chain]

    def extract_feature_ids(self, chains):
        for features in DistributionalVectorsNarrativeChainModel.extract_chain_feature_lists(chains, only_verb=self.only_verb,
                                                                                      adjectives=self.adjectives):
            yield [self.dictionary.token2id[f] for f in features if f in self.dictionary.token2id]

    @property
    def _description(self):
        return """\
Vector size:        %d
Predicate vocab:    %d
SVD:                %s
Only verb:          %s""" % (
            self.vectors.shape[1],
            len(self.dictionary),
            "%d -> %d" % (self.training_metadata["svd from"], self.training_metadata["svd"])
                if "svd" in self.training_metadata else "not used",
            self.only_verb,
        )

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        # Kwargs are the same as the attr names
        with open(os.path.join(directory, "model"), 'r') as input_file:
            data = pickle.load(input_file)
        data.update(kwargs)
        return DistributionalVectorsNarrativeChainModel(**data)

    def _save_to_directory(self, directory, model_name, human_name=None):
        pickle_attrs(self,
                     ["dictionary", "vectors", "only_verb", "adjectives"],
                     os.path.join(directory, "model"))
