"""
Deprecated.

Builds an on-disk index of a dataset for finding nearest neighbours in the event vector space.
This could still be useful, but I've really gone over to use a database index for fast querying
and easy use in the web interface.

You probably want to use index_neighbours_file.py to create an index on disk ready for fast
MySQL import, then run the Django management script indexnn to proceed with the database import.

"""
import argparse
from collections import Counter
import copy
from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus, predicate_relation
from cam.whim.entity_narrative.models.base.vectorspace.neighbours import NearestNeighbourFinder
from whim_common.utils.logging import get_console_logger
from whim_common.utils.progress import get_progress_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce vector projections of all the chains/events in a corpus "
                                                 "and store them along with the model for nearest neighbour search")
    parser.add_argument("corpus_dir", help="Path to rich event chain corpus to project into vector space")
    parser.add_argument("model_type", help="Model type to load for projection")
    parser.add_argument("model_name", help="Name of model to load for projection")
    parser.add_argument("--tarred", action="store_true", help="The corpus is tarred")
    parser.add_argument("--events", action="store_true", help="Project each individual event, not whole chains")
    parser.add_argument("--redis-port", type=int, default=6379, help="Port that Redis server is running on")
    parser.add_argument("--hash", type=int, default=10, help="Number of binary hash bits to use")
    parser.add_argument("--threshold", type=int,
                        help="Threshold to apply to counts of predicates and arguments. Events with rare predicates "
                             "or arguments are simply filtered out and not projected")
    parser.add_argument("--replace-entities", action="store_true",
                        help="Replace all entities with their headword (other than the chain protagonist) and treat "
                             "them as NP args")
    opts = parser.parse_args()

    log = get_console_logger("project")

    project_events = opts.events

    log.info("Loading corpus from %s" % opts.corpus_dir)
    corpus = RichEventDocumentCorpus(opts.corpus_dir, tarred=opts.tarred)
    # Doing this caches the corpus length, which we're going to need anyway
    num_docs = len(corpus)
    if project_events:
        log.info("Projecting events from %d documents" % num_docs)
    else:
        log.info("Projecting chains from %d documents" % num_docs)

    # Prepare a filter to apply to each event
    if opts.replace_entities:
        log.info("Replacing entities other than protagonist with NP headwords")
        def _filter_events(chain_entity, events):
            for event in events:
                # Shallow copy: don't copy the entities, but make sure we don't modify events that will come up again
                event = copy.copy(event)
                for event_entity in event.get_entities():
                    # Don't replace the chain entity
                    if event_entity is not chain_entity:
                        # Others get replaced by their headword (or None if a headword can't be found)
                        event.substitute_entity(event_entity, event_entity.get_head_word())
                yield event
    else:
        _filter_events = lambda entity, events: events

    if opts.threshold is not None:
        # Run over the dataset to count up predicates and arg words so we know what to filter out
        predicates = Counter()
        arguments = Counter()
        log.info("Counting event slot words to apply threshold")
        pbar = get_progress_bar(len(corpus), title="Counting")
        for doc in pbar(corpus):
            for entity, events in doc.get_chains():
                events = list(_filter_events(entity, events))
                # Collect the predicate of each event
                predicates.update([predicate_relation(entity, event) for event in events])
                # Collect all np args from the events
                args = sum([event.get_np_argument_words() for event in events], [])
                arguments.update(args)
        pbar.finish()
        # Get just the most common words
        predicates = [p for (p, cnt) in predicates.items() if cnt >= opts.threshold]
        arguments = [a for (a, cnt) in arguments.items() if cnt >= opts.threshold]
        log.info("Predicate set of %d, argument set of %d" % (len(predicates), len(arguments)))

        # Prepare a filter to get rid of any events with rare words
        def _filter_chains(chains):
            filtered_chains = []
            for entity, events in chains:
                filtered_events = [
                    event for event in _filter_events(entity, events) if
                    predicate_relation(entity, event) in predicates and
                    all(word in arguments for word in event.get_np_argument_words())
                ]
                if len(filtered_events):
                    filtered_chains.append((entity, filtered_events))
            return filtered_chains
    else:
        def _filter_chains(chains):
            filtered_chains = []
            for entity, events in chains:
                events = list(_filter_events(entity, events))
                if len(events):
                    filtered_chains.append((entity, events))
            return filtered_chains

    finder = NearestNeighbourFinder.build_from_document_corpus(
        corpus, opts.model_type, opts.model_name, progress=True, include_events=True,
        hash_size=opts.hash, log=log, project_events=opts.events, redis_port=opts.redis_port,
        filter_chains=_filter_chains,
    )
