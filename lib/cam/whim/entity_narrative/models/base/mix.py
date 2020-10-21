from cam.whim.entity_narrative.models.base.languagemodel.model import LanguageModelNarrativeChainModel
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel
from whim_common.utils.base import choose_from_list


class VectorSpaceAndLanguageModelNarrativeChainModel(VectorSpaceNarrativeChainModel, LanguageModelNarrativeChainModel):
    """
    Superclass for models that implement both a vector space projection and a chain scoring function.
    Subclasses should implement score_chains() and project().

    """
    MODEL_OPTIONS = dict(
        VectorSpaceNarrativeChainModel.MODEL_OPTIONS.items() +
        LanguageModelNarrativeChainModel.MODEL_OPTIONS.items(), **{
            "scorer": {
                "type": choose_from_list(["vs", "lm"]),
                "help": "Choose chain completion scoring technique. 'vs' scores by projecting into a vector space "
                        "and comparing; 'lm' scores by getting the language model to assign a score to each completion",
                "default": "lm",
            }
        }
    )

    def _score_choices(self, *args, **kwargs):
        if self.model_options["scorer"] == "vs":
            # Use vector space scorer
            return VectorSpaceNarrativeChainModel._score_choices(self, *args, **kwargs)
        else:
            # Use language model scorer
            return LanguageModelNarrativeChainModel._score_choices(self, *args, **kwargs)