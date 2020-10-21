import argparse
import os
import shutil
import sys

from whim_common.data.compression import detect_tarred_corpus
from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus
from whim_common.utils.logging import get_console_logger


class NarrativeChainModelTrainer(object):
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def get_training_temp_dir(self, model_name, create=True, remove=True):
        from whim_common.utils.local import LOCAL_CONFIG
        # Check for a locally-defined temp dir
        tmp_dir = LOCAL_CONFIG.get("train", "temp_dir")
        if tmp_dir:
            training_dir = os.path.join(tmp_dir, "chain_model", self.model_cls.MODEL_TYPE_NAME, model_name)
        else:
            training_dir = os.path.join(self.model_cls.get_model_directory(model_name), "training")
        # Remove the directory if it exists: good to make sure we're making a fresh start
        if remove and os.path.exists(training_dir):
            shutil.rmtree(training_dir)
        # Create the directory if it doesn't already exist
        if create and not os.path.exists(training_dir):
            os.makedirs(training_dir)
        return training_dir

    def prepare_arguments(self, parser):
        """
        Add arguments to an argparse parser for training.

        Default method does nothing. Overriding classes should add the arguments and argument groups
        they want.

        """
        pass

    def train(self, model_name, corpus, log, opts, chain_features=None):
        raise NotImplementedError("model type %s did not provide a train method for use by the generic trainer. "
                                  "Perhaps it provides its own separate training scripts" %
                                  self.model_cls.MODEL_TYPE_NAME)

    def train_from_cmd_line(self):
        """
        Use Argparse to process command line args, using the model class' own parameters and
        begin model training.

        """
        parser = argparse.ArgumentParser(description="Train a %s model" % self.model_cls.MODEL_TYPE_NAME)
        # Add some standard options that all trainers use
        parser.add_argument("model_type", help="Type of model to train")
        data_grp = parser.add_argument_group("Event chain data")
        data_grp.add_argument("corpus_dir", help="Directory to read in chains from for training data")
        data_grp.add_argument("model_name", help="Name under which to store the model")
        # Add the model type's arguments
        self.prepare_arguments(parser)

        # Parse cmd line args
        opts = parser.parse_args()

        log = get_console_logger("%s train" % self.model_cls.MODEL_TYPE_NAME)

        # Load event chain data
        tarred = detect_tarred_corpus(opts.corpus_dir)
        if tarred:
            log.info("Loading tarred dataset")
        else:
            log.info("Loading raw (untarred) dataset")
        corpus = RichEventDocumentCorpus(opts.corpus_dir, tarred=tarred)

        log.info("Counting corpus size")
        num_docs = len(corpus)
        if num_docs == 0:
            log.error("No documents in corpus")
            sys.exit(1)

        log.info("Training model '%s' on %d documents" % (opts.model_name, num_docs))
        try:
            self.train(opts.model_name, corpus, log, opts)
        except ModelTrainingError, e:
            log.error("Error training model: %s" % e)


class ModelTrainingError(Exception):
    pass


if __name__ == "__main__":
    # Uncomment this for debugging segfaults coming from Theano
    # import faulthandler
    # faulthandler.enable()
    #
    from cam.whim.entity_narrative.models import MODEL_TYPES, get_model_class
    # First argument always has to be model type, so we can get the other args
    if len(sys.argv) < 2 or sys.argv[1] not in MODEL_TYPES:
        print "First argument to training script must be a model type"
        print "Available types:"
        print "  %s" % ", ".join(MODEL_TYPES.keys())
        sys.exit(1)

    model_cls = get_model_class(sys.argv[1])
    trainer = model_cls.get_trainer()
    # Hand over to the training method to deal with the rest of the cmd line args
    trainer.train_from_cmd_line()
