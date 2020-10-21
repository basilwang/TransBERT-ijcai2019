import numpy

from cam.whim.entity_narrative.models.base.model import NarrativeChainModel
from whim_common.utils.progress import get_progress_bar


class LanguageModelNarrativeChainModel(NarrativeChainModel):
    """
    Abstract superclass for narrative chain models that measure the plausibility of the next event using something
    like a language model. It simply assigns a probability (or other score) to any given event chain
    P(e_0, ..., e_i-1).

    No assumption is made about the dependencies in the model. Alternative next events are compared by computing
    the LM score for each full chain (including the next event).

    """
    MODEL_OPTIONS = dict(NarrativeChainModel.MODEL_OPTIONS, **{})

    def __init__(self, **kwargs):
        super(LanguageModelNarrativeChainModel, self).__init__(**kwargs)

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        raise NotImplementedError("language model should implement _load_from_directory()")

    def _save_to_directory(self, directory, model_name, human_name=None):
        raise NotImplementedError("language model should implement _save_to_directory()")

    def score_chains(self, chains, additional_features=None):
        raise NotImplementedError("language model should implement score_chains()")

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        """
        Language model-based models have a common implementation of the scoring function that makes use of the
        scoring function defined by the specific model.

        """
        pbar = None
        if progress:
            pbar = get_progress_bar(len(contexts), title="Scoring choices")

        scores = numpy.zeros((len(contexts), max(len(choices) for choices in choice_lists)), dtype=numpy.float64)
        # Process each context in turn
        for i, (entity, context, choice_list) in enumerate(zip(entities, contexts, choice_lists)):
            # Make up full chains by adding the completion event onto each context
            chains = [(entity, context + [completion]) for completion in choice_list]
            # Get the model to score the completed chains
            scores[i] = self.score_chains(chains)

            if pbar:
                pbar.update(i)

        if pbar:
            pbar.finish()
        return scores

