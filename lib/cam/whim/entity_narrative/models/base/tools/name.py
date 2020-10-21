import argparse
from cam.whim.entity_narrative.models.base.model import NarrativeChainModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View/edit the human-readable name of a model")
    parser.add_argument("model_type", help="Model type")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("--set", help="Specify a new name (otherwise just prints existing name)", nargs="+")
    opts = parser.parse_args()

    model = NarrativeChainModel.load_by_type(opts.model_type, opts.model_name)

    if opts.set:
        new_name = " ".join(opts.set)
        print "Renaming model %s: %s -> %s" % (opts.model_name, model.human_name or "UNNAMED", new_name)
        model.human_name = new_name
        model.save(opts.model_name)
    else:
        if model.human_name is None:
            print "No name set"
        else:
            print model.human_name