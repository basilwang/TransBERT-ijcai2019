"""
Builds an on-disk index of a dataset, which can easily be imported into a MySQL database. The
database  can then be used for finding nearest neighbours in the event vector space.

Use this to create an index on disk ready for fast MySQL import, then run the Django management
script indexnn to proceed with the database import.

"""
import argparse
from collections import Counter
import copy
import cPickle as pickle
import os

from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus, predicate_relation, Event
from cam.whim.entity_narrative.models.base.vectorspace.project import VectorCorpus
from cam.whim.entity_narrative.models.event_comp.model import EventCompositionNarrativeChainModel
from whim_common.utils.base import remove_duplicates
from whim_common.utils.logging import get_console_logger
from whim_common.utils.progress import get_progress_bar
from nearpy.hashes.randombinaryprojections import RandomBinaryProjections
import unicodecsv


def event_to_text(entity, event):
    # First prepare a list of entities used by the event
    entities = event.get_entities()
    # Make sure the chain entity is at the beginning
    entities = [entity] + [e for e in entities if e is not entity]
    # Get the text representation of the entities
    #entity_texts = [e.to_text() for e in entities]
    # Get the text representation of the event
    event_text = event.to_text(entities)
    # Since the entities take up a lot of text space and we don't actually generally need them later, we don't
    #  store them. The event repr contains a representation of where they occurred, but we can't recover who they are.
    # We do know that entity0 is the protagonist
    return event_text


def event_from_text(text, entities):
    # Build the entity objects
    #entities = [Entity.from_text(e) for e in entity_texts]
    # Build the event object using these entities
    event = Event.from_text(text, entities)
    # The first entity in the list is the chain entity
    return entities[0], event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index nearest neighbours in a format ready for importing "
                                                 "into a database. Specific to event-comp model type")
    parser.add_argument("corpus_dir", help="Path to rich event chain corpus to project into vector space")
    parser.add_argument("model_name", help="Name of model to load for projection")
    parser.add_argument("output_base", help="Base filename to output to. This file will contain the index, but "
                                            "data will be stored in a CSV in the same directory")
    parser.add_argument("--tarred", action="store_true", help="The corpus is tarred")
    parser.add_argument("--hash", type=int, default=10, help="Number of binary hash bits to use")
    parser.add_argument("--threshold", type=int,
                        help="Threshold to apply to counts of predicates and arguments. Events with rare predicates "
                             "or arguments are simply filtered out and not projected")
    parser.add_argument("--threshold-sample", type=int,
                        help="To work out the predicate and arg words to keep (according to the threshold) look "
                             "only at this many of the first events in the corpus")
    parser.add_argument("--replace-entities", action="store_true",
                        help="Replace all entities with their headword (other than the chain protagonist) and treat "
                             "them as NP args")
    opts = parser.parse_args()

    log = get_console_logger("project")

    # Create the output directory
    output_dir = os.path.dirname(opts.output_base)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log.info("Loading corpus from %s" % opts.corpus_dir)
    corpus = RichEventDocumentCorpus(opts.corpus_dir, tarred=opts.tarred)
    # Doing this caches the corpus length, which we're going to need anyway
    num_docs = len(corpus)
    log.info("Projecting events from %d documents" % num_docs)

    model_name = opts.model_name
    hash_size = opts.hash
    tarred = opts.tarred
    threshold = opts.threshold
    replace_entities = opts.replace_entities

    # Prepare a filter to apply to each event
    if replace_entities:
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

    if threshold is not None:
        # Run over the dataset to count up predicates and arg words so we know what to filter out
        predicates = Counter()
        arguments = Counter()
        log.info("Counting event slot words to apply threshold")
        max_events = opts.threshold_sample
        if max_events is not None:
            log.info("Limiting to first %d events in corpus" % max_events)
            pbar = get_progress_bar(max_events, title="Counting")
            events_seen = 0
            for doc in corpus:
                for entity, events in doc.get_chains():
                    events = list(_filter_events(entity, events))
                    events_seen += len(events)
                    # Collect the predicate of each event
                    predicates.update([predicate_relation(entity, event) for event in events])
                    # Collect all np args from the events
                    args = sum([event.get_np_argument_words() for event in events], [])
                    arguments.update(args)
                # Stop once we've seen enough
                if events_seen >= max_events:
                    break
                pbar.update(events_seen)
            pbar.finish()
        else:
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
        predicates = [p for (p, cnt) in predicates.items() if cnt >= threshold]
        arguments = [a for (a, cnt) in arguments.items() if cnt >= threshold]
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

    # Wrap the chain filter in another filter to remove duplicate events
    def dup_key(event):
        # Key for deciding whether two events are duplicates
        return event.to_string_entity_text()
    def _filter_chains2(chains):
        filtered_chains = []
        for entity, events in _filter_chains(chains):
            events = remove_duplicates(events, key=dup_key)
            if len(events):
                filtered_chains.append((entity, events))
        return filtered_chains

    log.info("Preparing neighbour search hash")
    # Create binary hash
    binary_hash = RandomBinaryProjections("event-comp:%s_binary_hash" % model_name, hash_size)
    model = EventCompositionNarrativeChainModel.load(model_name)
    binary_hash.reset(model.vector_size)

    log.info("Creating index file in %s" % opts.output_base)
    with open(opts.output_base, "w") as f:
        pickle.dump({
            "model_name": model_name,
            "binary_hash": binary_hash,
            "dataset": opts.corpus_dir,
            "threshold": threshold,
        }, f, -1)

    data_filename = "%s_data.csv" % opts.output_base
    log.info("Projecting (storing in %s)" % data_filename)
    with open(data_filename, "w") as f:
        writer = unicodecsv.writer(f)
        for (vector, source, (entity, event)) in VectorCorpus.project_from_docs(corpus,
                                                                                "event-comp",
                                                                                model_name,
                                                                                progress=True,
                                                                                buffer_size=5000,
                                                                                project_events=True,
                                                                                filter_chains=_filter_chains2):
            # Get the model's indices for this event
            indices = model.get_event_input_indices(entity, event)
            if indices is None:
                # Event not recognised by the model -- don't add to the index
                continue
            # Use hash function to get NN key
            hash_key = binary_hash.hash_vector(vector, querying=True)[0]
            # Convert the event itself to text for neat storage
            event_text = event_to_text(entity, event)
            writer.writerow([hash_key] + [str(i) for i in indices] + [event_text])
