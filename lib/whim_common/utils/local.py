"""
Local config loader.

Since the common library can be placed anywhere in the project, we need to be able to specify a
project-specific location for certain directories. This goes in a file called local.conf in the root of
the whim-common package. Paths may be specified relative to the location of the file.
Valid settings for this file: LIB_DIR, MODELS_DIR, LOCAL_DIR.

This is only for basic config. Project-specific configuration should be put on the config file in
the "local" directory (which is located by the procedure above).

"""
import ConfigParser
import os


CONFIG_DEFAULTS = {
    "temp_dir": None,
}

BASIC_CONFIG_FILENAME = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local.conf"))


def get_local_config(config_file):
    config = ConfigParser.ConfigParser(defaults=CONFIG_DEFAULTS)
    if os.path.exists(config_file):
        config.read(config_file)
    return config

def config_rel_path(path):
    # Resolve so that this may be relative to config file
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(os.path.join(os.path.dirname(BASIC_CONFIG_FILENAME), path))


def check_for_basic_config():
    if not os.path.exists(BASIC_CONFIG_FILENAME):
        raise ConfigError("no basic config file found to get project filesystem layout from. "
                          "Please create %s" % BASIC_CONFIG_FILENAME)


if os.path.exists(BASIC_CONFIG_FILENAME):
    # A local config has been created: import straight away
    basic_config = ConfigParser.ConfigParser()
    basic_config.read(BASIC_CONFIG_FILENAME)

    LOCAL_DIR = config_rel_path(basic_config.get("DEFAULT", "LOCAL_DIR"))
    MODELS_DIR = config_rel_path(basic_config.get("DEFAULT", "MODELS_DIR"))
    LIB_DIR = config_rel_path(basic_config.get("DEFAULT", "LIB_DIR"))
else:
    LOCAL_DIR = MODELS_DIR = None

# Can only load local config file if base directory was given so we know where to look
if LOCAL_DIR:
    LOCAL_CONFIG_FILENAME = os.path.join(LOCAL_DIR, "config")
    LOCAL_CONFIG = get_local_config(LOCAL_CONFIG_FILENAME)
else:
    LOCAL_CONFIG_FILENAME = LOCAL_CONFIG = None


def load_config(config_name, defaults={}):
    """
    Loads a config file named <config_name>.conf in the project's local dir.

    """
    check_for_basic_config()
    config = ConfigParser.ConfigParser(defaults=defaults)
    config_filename = os.path.join(LOCAL_DIR, "%s.conf" % config_name)
    if os.path.exists(config_filename):
        config.read(config_filename)
    return config


class ConfigError(Exception):
    pass
