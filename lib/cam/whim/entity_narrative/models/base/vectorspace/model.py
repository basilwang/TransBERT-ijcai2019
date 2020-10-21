import warnings

import numpy

from cam.whim.entity_narrative.models.base.model import NarrativeChainModel
from whim_common.utils.base import str_to_bool
from whim_common.utils.progress import get_progress_bar
from whim_common.utils.vectors import magnitude


class VectorSpaceNarrativeChainModel(NarrativeChainModel):
    """
    Abstract superclass for narrative chain models that measure the plausibility of the next event by constructing
    a vector representation of the context (previous events and any other contextual features) and measuring the
    impact of possible next events on the vector projection.

    Notes on metric implementations
    ===============================
    Metrics can be provided by adding a method metric_<metric-name>() and making sure that metric-name is
    in the subclass' METRICS list (by overriding it).
    The method should take two arguments, which are Numpy arrays, v0 and v1. These may be two 1D vectors, in which
    case a single similarity score is expected. If the first is 2D and the second 1D, the result should be a
    vector containing the similarities of each of the rows of v0 and v1.

    """
    MODEL_OPTIONS = dict(NarrativeChainModel.MODEL_OPTIONS, **{
        "metric": {
            "help": "Similarity metric to use to compare vectors. Available values depend on model type's METRICS "
                    "attribute",
            "default": "default",
        },
        "check": {
            "help": "Perform some checks on the model's projections. Slows things down a bit; mainly useful for "
                    "debugging",
            "type": str_to_bool,
            "default": False,
        },
        "pairwise": {
            "help": "Instead of getting the chain projection of the context and comparing to each candidate, get the "
                    "event projection of each context event and sum the similarity to each candidate over the context "
                    "events",
            "type": str_to_bool,
            "default": False,
        },
    })
    # Metrics are provided as metric_*() methods
    # Subclasses may wish to override this list to add to it or exclude metrics provided by default
    METRICS = ["dot", "cosine", "euclid", "norm_euclid", "cross_entropy", "manhattan"]
    DEFAULT_METRIC = "cosine"

    def __init__(self, **kwargs):
        super(VectorSpaceNarrativeChainModel, self).__init__(**kwargs)
        self._check = bool(self.model_options["check"])

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        raise NotImplementedError("vector space model should implement _load_from_directory()")

    def _save_to_directory(self, directory, model_name, human_name=None):
        raise NotImplementedError("vector space model should implement _save_to_directory()")

    def project_chains(self, chains, progress=False):
        """
        Produce the low-dimensional vector representation of each of a list of event chains.
        """
        raise NotImplementedError("vector space model should implement project_chains()")

    def project_events(self, events, progress=False):
        """
        Produce the low-dimensional vector representation of each of a list of individual events
        (out of the context of a chain). Each event should be an (entity, event) pair.

        Default implementation treats each event as a 1-event chain and calls project_chains().

        """
        return self.project_chains([(entity, [e]) for (entity, e) in events],
                                   progress=progress)

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        """
        Vector space models have a common implementation of the scoring function that makes use of the
        projection defined by the specific model.

        """
        pairwise = self.model_options["pairwise"]

        pbar = None
        if progress:
            pbar = get_progress_bar(len(contexts), title="Scoring choices")

        scores = numpy.zeros((len(contexts), max(len(choices) for choices in choice_lists)), dtype=numpy.float64)

        if pairwise:
            for i, (entity, choice_list, context) in enumerate(zip(entities, choice_lists, contexts)):
                # Project each of the context events
                context_projections = self.project_events([(entity, event) for event in context], progress=progress)

                if numpy.all(context_projections == 0.):
                    # We have no representation of the chain, so can't score the candidates
                    # Give them all equal scores
                    scores[i, :] = 1.
                    continue

                scores[i, :] = numpy.sum(
                    [self._candidates_similarity(context_projections[context_i], entity, choice_list)
                     for context_i in range(context_projections.shape[0])],
                    axis=0
                )

                if pbar:
                    pbar.update(i)
        else:
            # Perform projection of context chains into vector space
            projection = self.project_chains(list(zip(entities, contexts)), progress=progress)
            for i, (entity, choice_list) in enumerate(zip(entities, choice_lists)):
                if numpy.all(projection[i] == 0.):
                    # We have no representation of the chain, so can't score the candidates
                    # Give them all equal scores
                    scores[i, :] = 1.
                    continue
                scores[i, :] = self._candidates_similarity(projection[i], entity, choice_list)

                if pbar:
                    pbar.update(i)

        if pbar:
            pbar.finish()

        return scores

    def _candidates_similarity(self, base_vector, entity, candidates):
        scores = numpy.zeros(len(candidates), dtype=numpy.float64)
        # Project each of the choices into the vector space, as if it's a whole chain
        choice_projections = self.project_events([(entity, choice) for choice in candidates])
        # Mask out any candidate projections that are all zero
        zero_vectors = numpy.all(choice_projections == 0., axis=1)
        choice_projections = numpy.ma.masked_where(
            numpy.ones_like(choice_projections)*zero_vectors[:, numpy.newaxis],
            choice_projections
        )
        scores[:] = self.metric(choice_projections, base_vector)
        # Where the candidates where zero vectors, give a -inf score
        scores[zero_vectors] = float('-inf')

        if self._check:
            # Make sure there's the right number of projections
            if choice_projections.shape[0] != len(candidates):
                warnings.warn("model's project() produced wrong number of projections: %d, should be %d" %
                              (choice_projections.shape[0], len(candidates)))
            if numpy.any(numpy.isnan(choice_projections)):
                warnings.warn("project() produced NaNs in its representation")
            if numpy.any(numpy.isnan(scores)):
                warnings.warn("NaN scores from metric %s" % self.model_options["metric"])
        return scores

    @property
    def metric(self):
        # Metric name has been provided as a model option
        metric_name = self.metric_name
        if metric_name not in self.METRICS:
            raise ValueError("metric %s is not valid for vector space model %s. Available choices: %s" %
                             (metric_name, self.MODEL_TYPE_NAME, ", ".join(self.METRICS)))
        try:
            return nan_to_minus_inf(getattr(self, "metric_%s" % metric_name))
        except AttributeError:
            raise ValueError("metric %s is listed as a metric for model type %s, but no implementation has "
                             "been provided in the form of a method metric_%s()" %
                             (metric_name, self.MODEL_TYPE_NAME, metric_name))

    @property
    def metric_name(self):
        if self.model_options["metric"] is None or self.model_options["metric"] == "default":
            return self.DEFAULT_METRIC
        else:
            return self.model_options["metric"]

    def metric_dot(self, v0, v1):
        return numpy.dot(v0, v1)

    def metric_cosine(self, v0, v1):
        return numpy.dot(v0, v1) / magnitude(v0) / magnitude(v1)

    def metric_euclid(self, v0, v1):
        return -magnitude(v1 - v0)

    def metric_norm_euclid(self, v0, v1):
        """ Should be proportional to cosine (when inverted, as it is here). Only really included for testing. """
        v0 = (v0.T / magnitude(v0)).T
        v1 = (v1.T / magnitude(v1)).T
        return -magnitude(v1 - v0)

    def metric_cross_entropy(self, v0, v1):
        # Take the log of the second distribution
        # Pointwise multiply by the first distribution(s)
        # Sum resulting vectors
        return numpy.dot(v0, numpy.log(v1))

    def metric_manhattan(self, v0, v1):
        return -numpy.sum(numpy.abs(v0 - v1), axis=-1)


def nan_to_minus_inf(fn):
    """ Decorator to replace all nans in a Numpy result with -inf """
    def _dec(*args):
        result = fn(*args)
        if numpy.isscalar(result):
            return numpy.INF if numpy.isnan(result) else result
        else:
            # Replace nans in the result with -inf
            result[numpy.isnan(result)] = numpy.NINF
            return result
    return _dec