from cam.whim.entity_narrative.models.base.model import ModelLoadError

"""
 All types of models should be added to this list.
 Instead of importing the classes, they're imported dynamically when we need them, so that we don't slow
  down the start-up of every script using this module.
 The type names should correspond to those declared by the class' MODEL_TYPE_NAME.

"""
MODEL_TYPES = {
    "candj":   "cam.whim.entity_narrative.models.candj.model.CandjNarrativeChainModel",
    "dist-vecs": "cam.whim.entity_narrative.models.dist_vecs.model.DistributionalVectorsNarrativeChainModel",
    "word2vec": "cam.whim.entity_narrative.models.word2vec.model.Word2VecNarrativeChainModel",
    "mikolov-verb": "cam.whim.entity_narrative.models.mikolov_verb.model.MikolovVerbNarrativeChainModel",
    "bigram": "cam.whim.entity_narrative.models.bigram.model.BigramNarrativeChainModel",
    "arg-comp": "cam.whim.entity_narrative.models.arg_comp.model.ArgumentCompositionNarrativeChainModel",
    "event-comp": "cam.whim.entity_narrative.models.event_comp.model.EventCompositionNarrativeChainModel",
}


def get_model_class(type_name):
    """
    Main way of loading a model type dynamically by name. This should always be used, for example, where the
    model type is specified as a command line argument.

    """
    if type_name not in MODEL_TYPES:
        raise ModelLoadError("no such model type: %s" % type_name)
    cls_path = MODEL_TYPES[type_name]
    # Split up into module and class
    mod_name, __, cls_name = cls_path.rpartition(".")
    # Import the class from the module
    mod = __import__(mod_name, fromlist=[cls_name])
    # Retrieve and return the class object
    return getattr(mod, cls_name)