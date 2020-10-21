from __future__ import absolute_import


def module_options_to_django_fields(opt_def_dict):
    """
    Produce a dict of Django form fields suitable for setting the options from the given
    module option definitions. Keys are the names of the fields.

    """
    from django.forms.fields import CharField, IntegerField, FloatField, BooleanField
    from cam.whim.utils.base import str_to_bool

    fields = {}

    for optname, optdef in opt_def_dict.items():
        opt_type = optdef.get("type", None)
        field_kwargs = {"required": False}
        # Try to work out what field type is appropriate for this option type
        if opt_type is None or (type(opt_type) is type and issubclass(opt_type, basestring)):
            # Simple string option
            field_class = CharField
            # Set the max length nice and high
            field_kwargs["max_length"] = 4096
        elif opt_type is int:
            field_class = IntegerField
        elif opt_type is float:
            field_class = FloatField
        elif opt_type is str_to_bool or opt_type is bool:
            field_class = BooleanField
        else:
            raise OptionFormBuildError("don't know of a suitable Django field type for option type '%s'" %
                                       opt_type.__name__)
        # Add help_text to the field if a help string is given (which it usually is)
        if "help" in optdef:
            field_kwargs["help_text"] = optdef["help"]
        # Add a default value if one was given
        if "default" in optdef:
            field_kwargs["initial"] = optdef["default"]
        # Instantiate the Django field
        fields[optname] = field_class(**field_kwargs)

    return fields


class OptionFormBuildError(Exception):
    pass
