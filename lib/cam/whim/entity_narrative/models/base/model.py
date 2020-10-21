import os
import shutil
import sys
from textwrap import wrap
import cPickle as pickle

from cam.whim.entity_narrative.models.base.train import NarrativeChainModelTrainer
from cam.whim.entity_narrative.shell.commands import ModelShell
from tabulate import tabulate

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", "..", "..", "..", "..", "..",
                         "models", "entitychains")


class NarrativeChainModel(object):
    """
    Abstract base class for narrative chain models. This is an extension of the idea of narrative cloze predictions
    from Chambers and Jurafsky's work, implemented in cam.whim.narrative.

    """
    SHELL_TYPE = ModelShell
    MODEL_OPTIONS = {}

    def __init__(self, model_options=None, training_metadata=None, human_name=None):
        self.human_name = human_name
        self._prepositions = None
        self._dependency_types = None

        self.model_options = {}
        self.set_default_model_options()
        if model_options is not None:
            self.update_model_options(model_options)

        if training_metadata is None:
            training_metadata = {}
        self.training_metadata = training_metadata

        self._nearest_neighbour_finder = None

    @classmethod
    def handle_command_line_options(cls, opt_string):
        if opt_string is None:
            return None
        if opt_string == "help":
            # Display option help for this special value
            print "Options for model type %s" % cls.MODEL_TYPE_NAME
            print "=" * (23 + len(cls.MODEL_TYPE_NAME))
            table = []
            widths = [30, 50]
            for optname, optdef in cls.MODEL_OPTIONS.items():
                hlp = optdef["help"] if "help" in optdef else ""
                table.append([optname, hlp])
            # Wrap columns
            table = [[wrap(cell, width=width) for (cell, width) in zip(row, widths)] for row in table]
            table_split = []
            for row in table:
                subrows = max(len(cell) for cell in row)
                new_row = [cell + [""] * (subrows - len(cell)) for cell in row]
                table_split.extend(zip(*new_row))
                # Add a blank line
                table_split.append([""] * len(table_split[0]))
            print tabulate(table_split[:-1], tablefmt="orgtbl")
            sys.exit(0)
        try:
            opts = cls.parse_model_options(opt_string)
        except OptionParseError, e:
            print >>sys.stderr, "Error in model options: %s" % e
            sys.exit(1)
        return opts

    @staticmethod
    def model_option_string_to_dict(opt_string):
        options = {}
        if opt_string:
            for name_val in opt_string.split(":"):
                if "=" not in name_val:
                    raise OptionParseError("options should be in the form 'name=value': got '%s'" % name_val)
                name, __, value = name_val.partition("=")
                options[name] = value
        return options

    @classmethod
    def parse_model_options(cls, opt_string):
        """
        Read an option string and parse the options in it, checking that they're valid for this model type.

        :param opt_string: string of options, in the form x=y, separated by colons
        :return: dict of options
        """
        opt_dict = NarrativeChainModel.model_option_string_to_dict(opt_string)
        options = {}
        for name, value in opt_dict.items():
            # Postprocess the option value
            opt_config = cls.MODEL_OPTIONS[name]
            if "type" in opt_config:
                try:
                    value = opt_config["type"](value)
                except Exception, e:
                    raise OptionParseError("error processing option value '%s' for %s option: %s" % (value, name, e))
                options[name] = value
            else:
                options[name] = value
        return options

    def update_model_options(self, options):
        """
        Set model options from a processed dict, the result of parsing command line options.

        :param options: dict of option values
        """
        self.model_options.update(options)

    def set_default_model_options(self):
        """
        Set all model options to their defined defaults. Used when no options are provided when
        the model is loaded.

        """
        self.model_options.update(self.prepare_default_model_options())

    @classmethod
    def prepare_default_model_options(cls):
        model_options = {}
        for optname, optdef in cls.MODEL_OPTIONS.items():
            if "default" in optdef:
                # We have a default value defined for this option
                model_options[optname] = optdef["default"]
            else:
                # No default defined: set to None instead
                model_options[optname] = None
        return model_options

    def score_choices(self, entity, context, choices):
        """
        Public interface for multiple choice narrative cloze task.

        Given a sequence of context event (those that have happened previously in a chain), assign a
        score to each of a list of possible next events. Returns a list of scores, in the same order
        that the options were given in.

        Additional chain-level contextual features may be given as an AdditionalEventChainCorpusFeatures
        object. Those features that were made available to the model at training time will be
        used by models that have that capacity.

        (Subclasses should implement _score_options())

        """
        return self.score_choices_bulk([entity], [context], [choices])[0]

    def score_choices_bulk(self, entities, contexts, choice_lists, progress=False):
        """
        Interface for performing many predictions at once.

        Instead of giving a single context and single list of choices, give a list of contexts and
        a list of lists of choices. The additional_features data, if given, should provide additional
        features for all of the queries in one array.

        If progress=True, display a progress bar on stdout if it's possible to monitor progress. It
        may not always be possible -- e.g. if the model is vectorize and performs all predictions
        simultaneously.

        """
        if len(contexts) != len(choice_lists):
            raise ModelQueryError("wrong number of choice lists (%d) for given contexts (%d)" %
                                  (len(choice_lists), len(contexts)))

        # Get scores from the subclass
        return self._score_choices(entities, contexts, choice_lists,
                                   progress=progress)

    @classmethod
    def get_model_type_directory(cls):
        return os.path.abspath(os.path.join(MODEL_DIR, cls.MODEL_TYPE_NAME))

    @classmethod
    def get_model_directory(cls, model_name):
        return os.path.join(cls.get_model_type_directory(), model_name)

    @classmethod
    def list_models(cls):
        model_dir = cls.get_model_type_directory()
        if os.path.exists(model_dir):
            # Models each have a subdirectory of the models directory
            return os.listdir(model_dir)
        else:
            return []

    @classmethod
    def list_models_with_names(cls):
        model_dir = cls.get_model_type_directory()
        if os.path.exists(model_dir):
            # Models each have a subdirectory of the models directory
            named_models = []
            # Try loading a name for each one
            for model_name in os.listdir(model_dir):
                human_name = None
                if os.path.exists(os.path.join(model_dir, model_name, "name")):
                    with open(os.path.join(model_dir, model_name, "name"), 'r') as f:
                        human_name = f.read().strip("\n ")
                named_models.append((model_name, human_name))
            return named_models
        else:
            return []

    @classmethod
    def delete_model(cls, model_name):
        """ Permanently remove the model from disk. """
        model_dir = cls.get_model_directory(model_name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    @classmethod
    def load(cls, model_name, **kwargs):
        directory = cls.get_model_directory(model_name)
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ModelLoadError("trained model '%s' does not exist for model type '%s' (looked in %s)" %
                                 (model_name, cls.MODEL_TYPE_NAME, directory))

        if os.path.exists(os.path.join(directory, "training_metadata")):
            with open(os.path.join(directory, "training_metadata"), 'r') as f:
                kwargs["training_metadata"] = pickle.load(f)

        if os.path.exists(os.path.join(directory, "name")):
            with open(os.path.join(directory, "name")) as f:
                kwargs["human_name"] = f.read().strip("\n ")

        return cls._load_from_directory(directory, **kwargs)

    def save(self, model_name, human_name=None):
        directory = type(self).get_model_directory(model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, "training_metadata"), 'w') as f:
            pickle.dump(self.training_metadata, f)

        if self.human_name is None:
            # Make sure no name is stored
            try:
                os.remove(os.path.join(directory, "name"))
            except OSError:
                pass
        else:
            with open(os.path.join(directory, "name"), "w") as f:
                f.write(self.human_name)

        self._save_to_directory(directory, model_name, human_name=human_name)

    @staticmethod
    def load_type(model_type):
        from cam.whim.entity_narrative.models import get_model_class
        return get_model_class(model_type)

    @staticmethod
    def load_by_type(model_type, model_name, **kwargs):
        cls = NarrativeChainModel.load_type(model_type)
        # Proceed to load the model
        return cls.load(model_name, **kwargs)

    @property
    def long_description(self):
        """ A detailed human-readable text containing info about the model. """
        return ""

    @classmethod
    def get_trainer(cls):
        return cls.TRAINER_CLASS(cls)
    
    @property
    def description(self):
        return """\
%s

Training details
================
%s

Options
=======
%s\
""" % (
            self._description,   # Start with model-type-specific stuff
            # Add some stuff we show for all models
            "\n".join("%s: %s" % (name.ljust(16), val) for (name, val) in self.training_metadata.items()),
            "\n".join("%s: %s" % (name.ljust(16), val) for (name, val) in self.model_options.items())
        )

    # ################ Abstract methods ########################
    # The following should be overridden by every subclass
    # You might also want to override:
    #  - long_description()

    # A short name used to identify the type of model and name directories, etc
    MODEL_TYPE_NAME = 'UNKNOWN-MODEL-TYPE'

    # Subclasses will want to override this with a class that overrides the base trainer
    TRAINER_CLASS = NarrativeChainModelTrainer

    @property
    def _description(self):
        """ Abstract. Return a human-readable description of the model. Included in description property. """
        return ""

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        """
        Subclass's routine for scoring choices.

        """
        raise NotImplementedError("subclasses should implement prediction")

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        raise NotImplementedError("%s does not implement _load_from_directory()" % cls.__name__)

    def _save_to_directory(self, directory, model_name, human_name=None):
        raise NotImplementedError


def cmd_line_model_options(model_type, option_string):
    model_cls = NarrativeChainModel.load_type(model_type)
    return model_cls.handle_command_line_options(option_string)


class ModelLoadError(Exception):
    pass


class OptionParseError(Exception):
    pass


class ModelQueryError(Exception):
    pass
