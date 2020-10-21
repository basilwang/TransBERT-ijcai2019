import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "..", "..",
                         "models", "gensim")


def get_model_dir(model_type, model_name):
    model_dir = os.path.join(MODEL_DIR, model_type, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir