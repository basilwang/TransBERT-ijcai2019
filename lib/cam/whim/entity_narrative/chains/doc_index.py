from itertools import groupby
from operator import itemgetter
import os
import cPickle as pickle
import sys

from cam.whim.entity_narrative.chains.document import RichEventDocument, rich_event_doc_files, RichEventDocumentCorpus
from whim_common.utils.progress import get_progress_bar


class RichDocumentVerbIndex(object):
    def __init__(self, verb_index, corpus):
        self.verb_index = verb_index
        self.corpus = corpus

    def __len(self):
        return len(self.verb_index)

    def __getitem__(self, item):
        return self.verb_index[item]

    def keys(self):
        return self.verb_index.keys()

    def load_document(self, doc_filename, archive_name=None):
        return self.corpus.get_document(doc_filename, archive_name=archive_name)

    @staticmethod
    def build_for_corpus(corpus, progress=True, limit=None):
        doc_dir = corpus.directory
        ext = "tar.index" if corpus.tarred else "index"
        if limit is not None:
            output_filename = os.path.join(doc_dir, "verbs-%d.%s" % (limit, ext))
        else:
            output_filename = os.path.join(doc_dir, "verbs.%s" % ext)

        # Read in each rich document in the directory in turn
        num_docs = len(corpus)
        if limit and num_docs > limit:
            num_docs = limit
        pbar = None
        if progress:
            print "Indexing %d documents" % num_docs
            pbar = get_progress_bar(num_docs, title="Indexing documents", counter=True)

        # Go through every event in all the documents, indexing the verb lemmas
        verb_index = {}
        for doc_num, (archive_name, filename, doc) in enumerate(corpus.archive_iter()):
            if limit and doc_num >= limit:
                break

            # Add an entry to the index for each event
            for event_num, event in enumerate(doc.events):
                verb_index.setdefault(event.verb_lemma, []).append((archive_name, filename, event_num))

            if pbar:
                pbar.update(doc_num)

        if pbar:
            pbar.finish()

        if progress:
            print "Indexed %d verb types" % len(verb_index)
            print "Outputting index to %s" % output_filename
        with open(output_filename, 'w') as output_file:
            pickle.dump(verb_index, output_file)

    @staticmethod
    def load(filename, index_tars=True):
        doc_dir = os.path.dirname(filename)
        tarred = filename.endswith("tar.index")

        # Load the index
        with open(filename, 'r') as infile:
            verb_index = pickle.load(infile)

        # Prepare the corpus wrapper
        corpus = RichEventDocumentCorpus(doc_dir, tarred=tarred, index_tars=index_tars)
        return RichDocumentVerbIndex(verb_index, corpus)


class RichDocumentVerbChainIndex(object):
    """
    An index like RichDocumentVerbIndex, but only includes instances where the verb features
    in an event chain of a given minimum length.

    """
    def __init__(self, verb_index, corpus, min_length):
        self.verb_index = verb_index
        self.corpus = corpus
        self.min_length = min_length

    @property
    def directory(self):
        return self.corpus.directory

    def __len(self):
        return len(self.verb_index)

    def __getitem__(self, item):
        return self.verb_index[item]

    def keys(self):
        return self.verb_index.keys()

    def load_document(self, doc_filename, archive_name=None):
        return self.corpus.get_document(doc_filename, archive_name=archive_name)

    def fill_cache(self, targets):
        for archive_name, filenames in groupby(sorted(targets), key=itemgetter(0)):
            self.corpus.fill_cache([filename for (archive, filename) in filenames], archive_name)

    def random_chains_for_verbs(self, verbs):
        import random
        choices = [random.choice(self[verb]) for verb in verbs]
        # Load the documents in one go, if they're stored in such a way that that will help
        self.fill_cache([(archive, filename) for (archive, filename, event_num) in choices])
        random_chains = []
        for archive, filename, event_num in choices:
            # Load the document
            document = self.load_document(filename, archive_name=archive)
            target_event = document.events[event_num]
            # Pick a chain that leads to the randomly selected event
            chains = document.chains_for_event(target_event)
            chains = [(e, c) for (e, c) in chains if len(c) >= self.min_length]
            # If there are no chains, something went wrong with our indexing
            # If there are multiple chains still, pick one randomly
            entity, chain = random.choice(chains)
            # See where the target event occurs in the chain
            target_index = (i for (i, event) in enumerate(chain) if event is target_event).next()
            random_chains.append((entity, target_index, chain, document))
        return random_chains

    def random_chain_for_verb(self, verb):
        import random
        archive_name, doc_filename, event_num = random.choice(self[verb])
        # Load the document
        document = self.load_document(doc_filename, archive_name=archive_name)
        target_event = document.events[event_num]
        # Pick a chain that leads to the randomly selected event
        chains = document.chains_for_event(target_event)
        chains = [(e, c) for (e, c) in chains if len(c) >= self.min_length]
        # If there are no chains, something went wrong with our indexing
        # If there are multiple chains still, pick one randomly
        entity, chain = random.choice(chains)
        # See where the target event occurs in the chain
        target_index = (i for (i, event) in enumerate(chain) if event is target_event).next()
        return entity, target_index, chain, document

    @staticmethod
    def build_for_corpus(corpus, min_length, progress=True, limit=None):
        doc_dir = corpus.directory
        ext = "tar.index" if corpus.tarred else "index"
        if limit is not None:
            output_filename = os.path.join(doc_dir, "verb_chains_%d-%d.%s" % (min_length, limit, ext))
        else:
            output_filename = os.path.join(doc_dir, "verb_chains_%d.%s" % (min_length, ext))

        # Read in each rich document in the directory in turn
        num_docs = len(corpus)
        if limit and num_docs > limit:
            num_docs = limit
        pbar = None
        if progress:
            print "Indexing %d documents" % num_docs
            pbar = get_progress_bar(num_docs, title="Indexing documents", counter=True)

        # Go through every event in all the documents, indexing the verb lemmas
        verb_index = {}
        for doc_num, (archive_name, filename, doc) in enumerate(corpus.archive_iter()):
            if limit and doc_num >= limit:
                break

            # Build an index of the event chains in this document
            chains = [doc.find_events_for_entity(entity) for entity in doc.entities]
            chains = [chain for chain in chains if len(chain) >= min_length]
            events = list(set(sum(chains, [])))

            # Add an entry to the index for each event
            for event in events:
                verb_index.setdefault(event.verb_lemma, []).append((archive_name, filename, doc.events.index(event)))

            if pbar:
                pbar.update(doc_num)

        if pbar:
            pbar.finish()

        if progress:
            print "Indexed %d verb types" % len(verb_index)
            print "Outputting index to %s" % output_filename
        with open(output_filename, 'w') as output_file:
            pickle.dump({
                "index": verb_index,
                "min_length": min_length
            }, output_file)

    @staticmethod
    def load(filename):
        doc_dir = os.path.dirname(filename)
        tarred = filename.endswith("tar.index")
        with open(filename, 'r') as infile:
            data = pickle.load(infile)

        # Prepare the corpus wrapper
        corpus = RichEventDocumentCorpus(doc_dir, tarred=tarred)
        return RichDocumentVerbChainIndex(data["index"], corpus, data["min_length"])
