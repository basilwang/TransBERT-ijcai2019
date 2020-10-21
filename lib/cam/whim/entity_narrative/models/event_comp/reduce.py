import argparse
from collections import Counter
from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus, predicate_relation
from cam.whim.entity_narrative.models.event_comp.model import EventCompositionNarrativeChainModel
from whim_common.utils.progress import get_progress_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut down the vocab size of a model and resave under a new name")
    parser.add_argument("corpus_dir", help="Directory to read in corpus from to apply threshold (you probably want "
                                           "to use a training set or a subset of one)")
    parser.add_argument("old_model_name", help="Name of model to load")
    parser.add_argument("new_model_name", help="Name to save the reduced model under")
    parser.add_argument("top_preds", help="Size to limit predicate vocab to", type=int)
    parser.add_argument("top_args", help="Size to limit argument vocab to", type=int)
    parser.add_argument("--tarred", action="store_true", help="The corpus is tarred")
    opts = parser.parse_args()

    print "Loading source model"
    model = EventCompositionNarrativeChainModel.load(opts.old_model_name)
    existing_models = EventCompositionNarrativeChainModel.list_models()
    if opts.new_model_name in existing_models:
        print "Model %s already exists: delete it first if you want to overwrite it" % opts.new_model_name
        print "  Model dir: %s" % EventCompositionNarrativeChainModel.get_model_directory(opts.new_model_name)

    print "Loading corpus from %s" % opts.corpus_dir
    corpus = RichEventDocumentCorpus(opts.corpus_dir, tarred=opts.tarred)

    # Run over the dataset to count up predicates and arg words so we know what our new vocabs look like
    predicates = Counter()
    arguments = Counter()
    print "Counting event slot words to apply threshold"
    pbar = get_progress_bar(len(corpus), title="Counting")
    for doc in pbar(corpus):
        for entity, events in doc.get_chains():
            # Collect the predicate of each event
            predicates.update([predicate_relation(entity, event) for event in events])
            # Collect all np args from the events
            args = sum([event.get_np_argument_words() for event in events], [])
            arguments.update(args)

    # Get just the most common words
    print "Getting most common predicates and arguments"
    predicates = [val for (val, cnt) in predicates.most_common(opts.top_preds)]
    arguments = [val for (val, cnt) in arguments.most_common(opts.top_args)]

    print "Filtering vocabulary of model"

    print predicates
    print arguments
