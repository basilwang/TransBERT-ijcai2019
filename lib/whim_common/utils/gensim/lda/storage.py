import os
from whim_common.utils.gensim import get_model_dir
from gensim.models.ldamulticore import LdaMulticore


def save_lda_model(lda, model_name):
    """
    Save an LdaMulticore model.

    """
    model_dir = get_model_dir("lda", model_name)
    model_path = os.path.join(model_dir, "lda_model")
    lda.save(model_path)


def load_lda_model(model_name):
    model_dir = get_model_dir("lda", model_name)
    model_path = os.path.join(model_dir, "lda_model")
    return LdaMulticore.load(model_path)