from collections import Counter
from itertools import islice
from operator import itemgetter
import os
import random
import re
import warnings
from whim_common.candc.dependency import DependencyGraph
from whim_common.candc.tags import Tags
from whim_common.data.compression import SynchronizedTarredCorpora, TarredCorpus
from whim_common.data.coref import Entity, unescape, escape
from whim_common.utils.base import remove_duplicates as remove_dups
from whim_common.utils.files import split_sections

DEFAULT_STOPVERBS = [
    "say", "do"
]


def _entity_id(arg, entity_list, tolerate_missing_entities=False):
    if arg is None:
        return "None"
    elif isinstance(arg, str):
        # Word itself
        # Use same escaping as is used for coref output
        return escape(arg)
    elif type(arg) is Entity:
        if entity_list is None:
            # Just show the entity's repr itself
            return arg.to_text()
        else:
            try:
                # Look for the entity (by identity) in the entity list
                entity_id = (index for index, entity in enumerate(entity_list) if entity is arg).next()
            except StopIteration:
                if tolerate_missing_entities:
                    return arg.to_text()
                else:
                    raise ValueError("event uses an entity arg (%s) that's not in the given entity list for "
                                     "text formatting" % arg)
            return "entity-%d" % entity_id
    raise ValueError("unknown event argument type: %s (%s)" % (type(arg).__name__, arg))


def _entity_from_id(string, entity_list):
    if string == "None":
        return None
    elif string.startswith("entity-"):
        return entity_list[int(string[7:])]
    else:
        # The word itself, presumably
        return unescape(string)


event_re = re.compile(r'(?P<verb>[^/]*) / (?P<verb_lemma>[^/]*) / '
                      r'verb_pos=\((?P<sentence_num>\d+),(?P<word_index>\d+)\) / '
                      r'type=(?P<type>[^/]*) / subj=(?P<subj>[^/]*) / obj=(?P<obj>[^/]*) / '
                      r'iobj=(?P<iobj>[^/]*)')


# Remove all characters from verbs (and verb lemmas) that are not either alphanumeric or "-"
verb_cleanup_re = re.compile(r'[^\w-]')
# None of these characters should appear in prepositions
illegal_preposition_re = re.compile(r'[/\\{},\+]')


class Event(object):
    def __init__(self, verb, verb_lemma, verb_position=None, type="normal", subject=None, object=None, iobject=None,
                 transitivity=None):
        self.transitivity = transitivity
        self.verb = verb
        self.verb_lemma = verb_lemma
        if verb_position is None:
            self.verb_position = (0, 1)
        else:
            self.verb_position = verb_position
        # Should be "normal" (default) or "predicative"
        self.type = type
        self.subject = subject
        self.object = object
        # If given, should be a pair (preposition, object)
        self.iobject = iobject

    def get_entities(self):
        entities = []
        if type(self.subject) is Entity:
            entities.append(self.subject)
        if type(self.object) is Entity:
            entities.append(self.object)
        if self.iobject is not None and type(self.iobject[1]) is Entity:
            entities.append(self.iobject[1])
        return entities

    def substitute_entity(self, old_entity, new_entity):
        if type(old_entity) is int:
            # Treat as an argument index
            if old_entity == 0:
                self.subject = new_entity
            elif old_entity == 1:
                self.object = new_entity
            elif old_entity == 2 and self.iobject is not None:
                self.iobject = (self.iobject[0], new_entity)
        else:
            # Otherwise, old_entity should be the entity we're replacing
            if self.subject is old_entity:
                self.subject = new_entity
            if self.object is old_entity:
                self.object = new_entity
            if self.iobject is not None and self.iobject[1] is old_entity:
                self.iobject = (self.iobject[0], new_entity)

    def get_entity_position(self, entity):
        if self.subject is entity:
            return 0
        elif self.object is entity:
            return 1
        elif self.iobject is not None and self.iobject[1] is entity:
            return 2
        else:
            return None

    def to_string_entity_text(self, entity_mapping={}):
        def _map_entity(e):
            return entity_mapping.get(e, str(e))

        s = "%s(%s" % (self.verb_lemma, _map_entity(self.subject) if self.subject is not None else "--")
        if self.object or self.iobject:
            s = "%s, %s" % (s, _map_entity(self.object) if self.object is not None else "--")
            if self.iobject:
                s = "%s, %s %s" % (s, self.iobject[0], _map_entity(self.iobject[1]))
        if self.type != "normal":
            s = "%s | %s" % (s, self.type)
        s += ")"
        return s

    def get_argument_texts(self, exclude_entities=False):
        """
        Collect all the texts associated with arguments of this event. Some may contribute multiple
        texts: entities include texts for all of their mentions.

        """
        texts = []
        for arg_num, arg in enumerate([self.subject, self.object,
                                       self.iobject[1] if self.iobject is not None else None]):
            if arg is None:
                texts.append([])
            elif type(arg) is Entity:
                if not exclude_entities:
                    # If it's an entity, include all the mentions
                    texts.append(arg.non_pronoun_mention_texts)
            else:
                # Otherwise it must be a string
                texts.append([arg])
        return texts

    def get_argument_words(self, exclude_entitites=False):
        words = [sum([text.split() for text in arg_texts], []) for arg_texts in
                 self.get_argument_texts(exclude_entities=exclude_entitites)]
        words = [[w.lower().strip() for w in arg_words] for arg_words in words]
        words = [set([w for w in arg_words if w]) for arg_words in words]
        return words

    def get_np_argument_words(self):
        """
        :return: a list of the head word arguments, leaving out any empty arguments or entities
        """
        args = []
        if type(self.subject) is str:
            args.append(self.subject)
        if type(self.object) is str:
            args.append(self.object)
        if self.iobject is not None and type(self.iobject[1]) is str:
            args.append(self.iobject[1])
        return args

    def to_text(self, entity_list=None, tolerate_missing_entities=False):
        return "%s / %s / verb_pos=(%d,%d) / type=%s / subj=%s / obj=%s / iobj=%s" % (
            self.verb, self.verb_lemma, self.verb_position[0], self.verb_position[1],
            self.type,
            _entity_id(self.subject, entity_list, tolerate_missing_entities=tolerate_missing_entities),
            _entity_id(self.object, entity_list, tolerate_missing_entities=tolerate_missing_entities),
            "%s,%s" % (self.iobject[0], _entity_id(self.iobject[1], entity_list,
                                                   tolerate_missing_entities=tolerate_missing_entities))
                if self.iobject is not None else "None"
        )

    @staticmethod
    def from_text(text, entity_list):
        match = event_re.match(text)
        if not match:
            raise ValueError("could not parse event string: %s" % text)
        groups = match.groupdict()

        if groups["iobj"] == "None":
            iobj = None
        else:
            iobj_bits = groups["iobj"].split(",")
            iobj = (iobj_bits[0], _entity_from_id(iobj_bits[1], entity_list))

        return Event(
            groups["verb"], groups["verb_lemma"], (int(groups["sentence_num"]), int(groups["word_index"])),
            type=groups["type"],
            subject=_entity_from_id(groups["subj"], entity_list),
            object=_entity_from_id(groups["obj"], entity_list),
            iobject=iobj
        )

    @staticmethod
    def from_dependency_event(verb, dependency, entity):
        subj = obj = iobj = None
        if dependency == "subj":
            subj = entity
        elif dependency == "obj":
            obj = entity
        elif dependency.startswith("prep"):
            iobj = (dependency[:4], dependency[5:])
        elif dependency.startswith("pp"):
            iobj = (dependency[:2], dependency[3:])
        else:
            raise ValueError("unknown dependency type %s" % dependency)
        return Event(verb, verb, subject=subj, object=obj, iobject=iobj)

    def __str__(self):
        return self.to_string_entity_text()

    def __repr__(self):
        return str(self)

    @staticmethod
    def _cmp_args(arg0, arg1):
        if arg0 is None:
            # Require both to be None if one is
            return arg1 is None
        elif arg1 is None:
            return False
        elif type(arg0) is Entity:
            # Require identity of entities
            return arg0 is arg1
        elif type(arg0) is tuple:
            # Iobj style tuple
            return arg0[0] == arg1[0] and Event._cmp_args(arg0[1], arg1[1])
        else:
            # Just a string: require them to match
            return arg0 == arg1

    def __eq__(self, other):
        return type(other) is Event and other.type == self.type and \
            self.verb_lemma == other.verb_lemma and \
            Event._cmp_args(self.subject, other.subject) and \
            Event._cmp_args(self.object, other.object) and \
            Event._cmp_args(self.iobject, other.iobject)


class RichEventDocument(object):
    def __init__(self, doc_name, entities, events):
        self.entities = entities
        self.doc_name = doc_name
        self.events = events

    def find_events_for_entity(self, entity, remove_duplicates=False, remove_duplicate_verbs=False):
        """
        Extract an event chain by looking up all the events that include a particular entity as one of
        their args.

        """
        chain = [
            event for event in self.events
                if event.subject is entity or event.object is entity or
                (event.iobject is not None and event.iobject[1] is entity)
        ]
        if remove_duplicate_verbs:
            chain = remove_dups(chain, key=lambda e: e.verb_lemma)
        elif remove_duplicates:
            chain = remove_dups(chain)
        return chain

    def chains_for_event(self, target_event, remove_duplicates=False, remove_duplicate_verbs=False):
        # Pick a chain that involves the event
        # There could be multiple
        possible_target_entities = []
        if type(target_event.subject) is Entity:
            possible_target_entities.append(target_event.subject)
        if type(target_event.object) is Entity:
            possible_target_entities.append(target_event.object)
        if target_event.iobject is not None and type(target_event.iobject[1]) is Entity:
            possible_target_entities.append(target_event.iobject[1])

        return [(entity, self.find_events_for_entity(entity, remove_duplicates=remove_duplicates,
                                                     remove_duplicate_verbs=remove_duplicate_verbs))
                for entity in possible_target_entities]

    def get_chains(self, stoplist=None):
        """
        Return all event chains in the document: one for each entity that's represented at least once
        in an event.

        """
        chains = [(e, self.find_events_for_entity(e)) for e in self.entities]
        if stoplist:
            chains = [
                (entity, [ev for ev in chain if predicate_relation(entity, ev) not in stoplist])
                for (entity, chain) in chains
            ]
        chains = [(e, c) for (e, c) in chains if len(c) > 0]
        return chains

    def to_text(self):
        """
        String representation of the document with event information. Designed for outputting to
        files.

        """
        # Build the entity section
        # They're given in order, so we don't strictly need the int indices, but it makes the file easier to read
        entities_text = "\n".join("%d: %s" % (e_num, e.to_text()) for e_num, e in enumerate(self.entities))
        # Build the event representations
        events_text = "\n".join(event.to_text(self.entities) for event in self.events)
        return "%s\n\nEntities:\n%s\n\nEvents:\n%s\n" % (self.doc_name, entities_text, events_text)

    @staticmethod
    def from_text(text):
        if len(text.splitlines()) == 1:
            # Empty document: error in event extraction
            return RichEventDocument(text.strip(), [], [])

        sections = split_sections((l.strip() for l in text.splitlines()), ["Entities:", "Events:"])
        # First comes the doc name
        doc_name = sections[0][0].strip()
        # Then a whole section giving the entities, with numbers at the beginning
        entity_lines = [line.partition(": ")[2] for line in sections[1] if line]
        entities = [Entity.from_text(line) for line in entity_lines]
        # Then another giving the events themselves
        events = [Event.from_text(line, entities) for line in sections[2] if line]

        return RichEventDocument(doc_name, entities, events)

    @staticmethod
    def build_documents(text_corpus, coref_dir, deps_dir, pos_dir, stopverbs=None, skip_files=None):
        """
        Generator to build rich event documents from raw data.

        """
        if skip_files is None:
            skip_files = []
        # Synchronized read from corefs and deps
        tar_reader = SynchronizedTarredCorpora([coref_dir, deps_dir, pos_dir], index=False)
        # Read each document's entities
        for filename, (entity_filename, deps_filename, pos_filename) in tar_reader:
            if filename in skip_files:
                continue
            with open(entity_filename, 'r') as entity_file:
                entities = Entity.read_entity_file(entity_file)
            # Extract the document name
            doc_name = os.path.splitext(os.path.basename(entity_filename))[0]
            # Load the document text
            doc_text = text_corpus.get_document_by_name(doc_name, text=True)
            # Load the dependency graph
            dep_graphs = DependencyGraph.from_file(deps_filename)
            # Load tagged sentences
            tagged_sentences = Tags.from_file(pos_filename)

            try:
                yield build_document(doc_name, doc_text, entities, dep_graphs, tagged_sentences, stopverbs=stopverbs)
            except EventExtractionError as err:
                warnings.warn("error extracting document %s: %s" % (doc_name, err))
                yield doc_name, err


class RichEventDocumentCorpus(object):
    """
    Simple tool for iterating over the documents in a corpus dir.

    """
    def __init__(self, directory, tarred=False, index_tars=True):
        self.directory = directory
        self._length = None

        if tarred:
            self.tarred_corpus = TarredCorpus(directory, index=index_tars)
        else:
            self.tarred_corpus = None

    @property
    def tarred(self):
        return self.tarred_corpus is not None

    @property
    def filenames(self):
        if self.tarred_corpus is not None:
            # Iterate over the files within the tar archives, extracting each archive one at a time
            return filter_non_doc_files(self.tarred_corpus, path=True)
        else:
            return rich_event_doc_files(self.directory)

    def list_filenames(self):
        """
        Just list the filenames. Yields (archive,filename) pairs. The files might not actually exist.

        """
        if self.tarred_corpus is not None:
            # Iterate over the files within the tar archives, extracting each archive one at a time
            return filter_non_doc_files(self.tarred_corpus.list_archive_iter(), path=True, key=itemgetter(1))
        else:
            return ((None, fn) for fn in rich_event_doc_files(self.directory))

    def fill_cache(self, filenames, archive_name=None):
        if self.tarred_corpus:
            if archive_name is None:
                raise ValueError("tried to extract a document from a tarred corpus without specifying the name of the "
                                 "tar archive")
            self.tarred_corpus.fill_cache(archive_name, filenames)
        # Don't do anything if we're not using a tarred corpus: it's only for tar archives that we need the speedup

    def get_document(self, filename, archive_name=None):
        if self.tarred_corpus is not None:
            if archive_name is None:
                raise ValueError("tried to extract a document from a tarred corpus without specifying the name of the "
                                 "tar archive")
            # Extract a file from the tar archive and parse it
            text = self.tarred_corpus.extract_file(archive_name, filename)
            if text is None:
                raise IOError("document %s does not exist in archive %s" % (filename, archive_name))
            return RichEventDocument.from_text(text)
        else:
            # Just load the file and parse it
            with open(os.path.join(self.directory, filename), 'r') as doc_file:
                return RichEventDocument.from_text(doc_file.read())

    def __len__(self):
        # This can take a long time
        # Once it's done, we store the count in a file
        len_filename = os.path.join(self.directory, "length.meta")
        if os.path.exists(len_filename):
            with open(len_filename, 'r') as len_file:
                self._length = int(len_file.read().strip())
        elif self._length is None:
            # Count and cache the length
            self._length = sum(1 for fn in self.list_filenames())
            # Store it for next time
            with open(len_filename, 'w') as len_file:
                len_file.write("%d" % self._length)
        return self._length

    def __iter__(self):
        for filename in self.filenames:
            with open(os.path.join(self.directory, filename), 'r') as doc_file:
                yield RichEventDocument.from_text(doc_file.read())

    def archive_iter(self, subsample=None, start=0):
        for archive_name, filename, file_path in self.filename_archive_iter(subsample=subsample, start=start):
            with open(file_path, 'r') as doc_file:
                try:
                    yield archive_name, filename, RichEventDocument.from_text(doc_file.read())
                except Exception as err:
                    raise IOError("error reading document %s (%s): %s" % (filename, archive_name, err))

    def filename_archive_iter(self, subsample=None, start=0):
        if not self.tarred:
            # Not a tarred corpus: just iterate over the files, setting the archive_name to None
            for filename in islice(self.filenames, start, None):
                # If subsampling, decide whether to extract this file
                if subsample is not None and random.random() > subsample:
                    # Reject this file
                    continue
                yield None, filename, os.path.abspath(os.path.join(self.directory, filename))
        else:
            for archive_name, filename, tmp_filename in \
                    filter_non_doc_files(self.tarred_corpus.archive_iter(subsample=subsample, start=start),
                                         key=itemgetter(1)):
                yield archive_name, filename, tmp_filename

    def list_archive_iter(self):
        """
        Like filename_archive_iter, but doesn't extract the archives to a temporary location, just lists
        their contents.
        """
        if not self.tarred:
            for filename in self.filenames:
                yield None, filename
        else:
            for archive_name, filename in filter_non_doc_files(self.tarred_corpus.list_archive_iter(),
                                                               key=itemgetter(1)):
                yield archive_name, filename


def word_to_char_mapping(doc_text):
    maps = []
    for line in doc_text.splitlines():
        map = {}
        char_count = 0
        word_num = -1
        for word_num, word in enumerate(line.split()):
            map[word_num] = char_count
            char_count += len(word) + 1
        # Add an extra imaginary word, so we can get the chars at the end of the sentence
        map[word_num+1] = char_count
        maps.append(map)
    return maps


def char_to_word_mapping(doc_text):
    maps = []
    for line in doc_text.splitlines():
        map = {}
        char_count = 0
        for word_num, word in enumerate(line.split()):
            for char_num in range(char_count, char_count+len(word)+1):
                map[char_num] = word_num
            char_count += len(word) + 1
        maps.append(map)
    return maps


def find_entity_mentions(entities, word_index_list, word_to_char_map, sentence_num):
    found_entities = [
        find_entity_mention(entities,
                            word_to_char_map[sentence_num][word_index],
                            word_to_char_map[sentence_num][word_index+1]-1,
                            sentence_num) for word_index in word_index_list
    ]
    return [found_entity if found_entity is not None else word_index
            for (found_entity, word_index) in zip(found_entities, word_index_list)]


def find_entity_mention(entities, start_char, end_char, sentence_num):
    for entity in entities:
        if any(mention.sentence_num == sentence_num and
               mention.head_span[0] <= start_char and
               mention.head_span[1] >= end_char for mention in entity.mentions):
            return entity
    return None


def replace_indices_with_words(tagged_sentence, index_list):
    for index in index_list:
        if type(index) is int:
            yield tagged_sentence.lemmas[index]
        elif type(index) is tuple:
            yield "-".join(tagged_sentence.lemmas[i] for i in index)
        else:
            yield index


class InvalidPredicativeAdjective(Exception):
    pass


def predicate_relation(entity, event, handle_predicative=False):
    """
    :param entity: protagonist
    :param event: event object
    :param handle_predicative: by default, no special treatment is applied to predicative events.
        If True, they are represented as "adj:<verb>:<adj>", where <verb> is the verb that governed the
        adjective ("be" or "become") and <adj> is the predicative adjective.
    :return: string representation of predicate-GR
    """
    # Special handling of predicative adjective events
    if handle_predicative and event.type == "predicative":
        if not isinstance(event.object, str):
            # Object of predicative event should always be a string, but this error can occur because of an
            # old bug in the extraction code. If so, you're probably using old data
            raise InvalidPredicativeAdjective(
                "predicative adjective with object of type %s: %s" % (type(event.object).__name__, event))
        return "adj:%s:%s" % (event.verb_lemma, event.object)

    return "%s:%s" % (event.verb_lemma, dependency_relation(entity, event))


def dependency_relation(entity, event):
    if event.subject is entity:
        return "subj"
    elif event.object is entity:
        return "obj"
    elif event.iobject is not None and event.iobject[1] is entity:
        return "prep_%s" % event.iobject[0]
    else:
        raise KeyError("entity %s was not found as any argument of event %s" % (entity, event))


def transitivity_class(event):
    """
    If the event has a defined transitivity class, just return that.
    Otherwise, classify the given event as intransitive, transitive, reflexive, passive, or noargs,
    based on which of its subject and object slots are filled.

    :param event: event (note that the entity is not required)
    :return: string denoting class
    """
    if event.transitivity is not None:
        return event.transitivity
    subject, object = (event.subject is not None, event.object is not None)
    if subject and object:
        # Check whether they're the same entity, in which case it's reflexive
        if isinstance(event.subject, Entity) and event.subject is event.object:
            return "reflexive"
        else:
            return "transitive"
    elif subject:
        # Subject only: intransitive
        return "intransitive"
    elif object:
        # Object only: passive
        return "passive"
    else:
        # No direct args at all: odd case, but possible
        return "noargs"


def predicate_relation_with_transitivity(entity, event, handle_predicative=False):
    """
    Like predicate_relation, but also includes a marker of the transitivity of the predicate.

    """
    return "%s[%s]" % (predicate_relation(entity, event, handle_predicative=handle_predicative),
                       transitivity_class(event))


def rich_event_doc_files(directory, filename_only=False):
    """
    Iterate over the rich event documents in a directory, excluding certain files that should
    always be ignored (indices and the like).

    """
    for dir, dirs, files in os.walk(directory):
        reldir = os.path.relpath(dir, directory)
        for fn in filter_non_doc_files(files):
            if filename_only:
                yield fn
            else:
                yield os.path.join(reldir, fn)


def filter_non_doc_files(iterator, path=False, key=lambda x:x):
    for obj in iterator:
        fn = key(obj)

        if path:
            basename = os.path.basename(fn)
        else:
            basename = fn

        if not basename.endswith(".index") and not basename.startswith("README") and not basename.endswith(".meta"):
            yield obj


def build_document(doc_name, text, entities, dep_graphs, tagged_sentences, stopverbs=None):
    if stopverbs is None:
        # Use the default set
        stopverbs = DEFAULT_STOPVERBS

    if len(dep_graphs) != len(tagged_sentences):
        if len(tagged_sentences) == 1:
            # When there's one sentence and it can't be parsed, end up getting no dep graphs
            # Just put an empty one in, since that's what it should be
            dep_graphs = [DependencyGraph()]
        else:
            raise EventExtractionError("got %d dep graphs with %d tagged sentences for doc %s" %
                                       (len(dep_graphs), len(tagged_sentences), doc_name))

    # Look for all the verbs in the document: they will give us our events
    verbs = sum([
        [(sentence_num, word_num) for (word_num, pos) in enumerate(sentence_tags.pos_tags) if pos.startswith("VB")]
        for (sentence_num, sentence_tags) in enumerate(tagged_sentences)
    ], [])

    # Filter out some verbs that should never be used to build events
    verbs = [(s, w) for (s, w) in verbs if tagged_sentences[s].lemmas[w] not in stopverbs]

    # Construct a map to get from word numbers to char numbers
    word_to_char_maps = word_to_char_mapping(text)
    # Construct a map to go the other way
    char_to_word_maps = char_to_word_mapping(text)

    # Add NE types to entities using the C&C tags
    add_entity_types(entities, char_to_word_maps, [sentence.ne_tags for sentence in tagged_sentences])

    events = []
    for sentence_num, word_index in verbs:
        tagged_sentence = tagged_sentences[sentence_num]
        word = tagged_sentence.words[word_index]
        lemma = tagged_sentence.lemmas[word_index]

        # Some substitutions to clean up the verbs a bit
        word = verb_cleanup_re.sub("", word)
        lemma = verb_cleanup_re.sub("", lemma)
        lemma = lemma.lower()
        # If the verb was entirely punctuation/whitespace, don't use it
        if len(lemma) == 0:
            continue

        # Honestly, why doesn't this get lemmatized correctly?
        if lemma == "'s" or lemma == "s":
            # NB Could cause a problem if it's really "have", but it'll only be included if it's got predicative
            #  arguments anyway
            lemma = "be"

        # First check whether this verb satisfies the conditions for being included as an event
        right_deps = dep_graphs[sentence_num].get_by_arg1(word_index)

        # If the verb is arg1 of an aux dep, it's a modal verb and shouldn't form an event
        if any(dep.dep_type == "aux" for dep in right_deps):
            continue
        # Check whether this verb dominates another verb via an xcomp (e.g. hurry to leave)
        if any(right_dep.dep_type == "xcomp" and len(right_dep.args) > 2 and
                       tagged_sentence.pos_tags[right_dep.args[2][1]].startswith("VB") for right_dep in right_deps):
            # Dominates another verb: skip this one, just allowing the subordinate verb to be used
            continue

        # Work out what this verb's arguments are
        left_deps = dep_graphs[sentence_num].get_by_arg0(word_index)

        subjects = []
        objects = []
        iobject_preps = []
        iobjects = []
        object2s = []
        unknown_deps = []
        passive = False
        event_type = "normal"

        for dep in left_deps:
            if len(dep.args) < 2 or tagged_sentence.pos_tags[dep.args[1][1]].startswith("VB"):
                # Skip any arcs that, for some reason, attach this verb to another verb
                continue
            if dep.dep_type == "ncsubj":
                # Subject of verb
                subjects.append(dep.args[1][1])
                if len(dep.args) > 2 and dep.args[2][0] == "obj" and tagged_sentence.pos_tags[word_index] == "VBN":
                    # Indicator of a passive: the syntactic subj is marked as being a semantic obj
                    # Swap them round later
                    # Only use this where the verb is in past participle form (VBN): otherwise it's probably a
                    #  subordinate verb of a passive verb and not passive at all
                    passive = True
            elif dep.dep_type == "dobj":
                # Object
                objects.append(dep.args[1][1])
            elif dep.dep_type == "iobj":
                # Indirect object: the relation is to the preposition
                prep_index = dep.args[1][1]
                prep = tagged_sentence.lemmas[prep_index]
                # Check that this is a vaguely sensible-looking preposition
                if illegal_preposition_re.search(prep) is not None:
                    # Nasty preposition: reject (probably corrupted text)
                    continue
                # Look up the preposition to find what it attaches to
                prep_deps = dep_graphs[sentence_num].get_by_arg0(prep_index)
                # If a preposition attaches to a verb, but has no dependent, don't include it as a direct object
                if len(prep_deps) == 0:
                    continue
                # There can be multiple objects this preposition attaches to via a conjunction
                for prep_dep in prep_deps:
                    # Include them as seperate indirect objects
                    prep_arg = prep_dep.args[1][1]
                    iobject_preps.append(prep)
                    iobjects.append(prep_arg)
            elif dep.dep_type == "obj2":
                # These will be used if available in passive constructions
                object2s.append(dep.args[1][1])
            else:
                unknown_deps.append(dep)

        # Make passives into actives if possible
        if passive:
            if objects:
                # There are various reasons why this happens, some legitimate, but most due to parser errors
                # Since it's fairly rare anyway and needs to be handled in different ways, just skip these cases
                continue
            # In a passive construction the agent ("by") is supplied as an obj2
            # Where this is available, include it as the subject
            objects = subjects
            subjects = object2s

        # Handle predicative adjectives
        if lemma in ["be", "become"]:
            # Check whether this gives us a predicative adjective
            for right_dep in right_deps:
                if right_dep.dep_type == "xcomp" and len(right_dep.args) > 2 and \
                        tagged_sentence.pos_tags[right_dep.args[2][1]].startswith("J"):
                    # Predicative complement: is ADJ
                    # Subjects should have been found: if not, doesn't make sense, skip
                    if subjects:
                        # Put the adjective in the object position
                        objects = [right_dep.args[2][1]]
                        iobjects = []
                        event_type = "predicative"
                        break
            else:
                # No predicative arguments were found: don't use this verb
                # Apart from predicative adjectives, we want to throw away "be"
                continue

        # Check for any particles that attach to the verb
        for right_dep in right_deps:
            if right_dep.dep_type == "ncmod" and len(right_dep.args) > 2 and \
                            tagged_sentence.pos_tags[right_dep.args[2][1]] == "RP":
                # Need to do the same cleanup on the particles as on the verbs to avoid dodgy chars
                word_particle = verb_cleanup_re.sub("", tagged_sentence.words[right_dep.args[2][1]])
                lemma_particle = verb_cleanup_re.sub("", tagged_sentence.lemmas[right_dep.args[2][1]])
                # Maybe it was only dodgy chars
                if len(lemma_particle):
                    # Combine the particle with the verb
                    word = "%s+%s" % (word, word_particle)
                    lemma = "%s+%s" % (lemma, lemma_particle)
                    # Only do this once
                    break

        # Skip any verbs that have no arguments at all
        if not subjects and not objects and not iobjects:
            continue

        # Look for where the args correspond to entities: where they appear in the head NP
        subjects = find_entity_mentions(entities, subjects, word_to_char_maps, sentence_num)
        # In predicative adjective cases, the object is the adjective, so don't try to find entities including it
        if event_type == "normal":
            objects = find_entity_mentions(entities, objects, word_to_char_maps, sentence_num)
            iobjects = find_entity_mentions(entities, iobjects, word_to_char_maps, sentence_num)
        # In other cases, just use the lemmas themselves (head words)
        subjects = list(replace_indices_with_words(tagged_sentence, subjects))
        objects = list(replace_indices_with_words(tagged_sentence, objects))
        iobjects = list(replace_indices_with_words(tagged_sentence, iobjects))

        # Multiply out multiple args in each position: these generally correspond to conjunctions
        # so should expand into multiple events
        if not subjects:
            subjects = [None]
        arg_sets = [[subj] for subj in subjects]
        if not objects:
            objects = [None]
        arg_sets = [args + [obj] for obj in objects for args in arg_sets]
        if iobjects:
            arg_sets = [args + [(prep, iobj)] for (prep, iobj) in zip(iobject_preps, iobjects) for args in arg_sets]
        else:
            arg_sets = [args + [None] for args in arg_sets]

        for subj, obj, iobj in arg_sets:
            events.append(Event(word, lemma, (sentence_num, word_index), type=event_type,
                                subject=subj, object=obj, iobject=iobj))

    return RichEventDocument(doc_name, entities, events)


def add_entity_types(entities, char_to_words, doc_ne_tags):
    for entity in entities:
        entity_type = None
        if (entity.gender == "male" or entity.gender == "female") and entity.gender_prob >= 0.9:
            # This entity has gender, so is presumably a person (this is English)
            # Only use this if it's really confident
            entity_type = "person"
        else:
            # Try to set the entity type by looking at NEs in the mentions
            ne_tag_counter = Counter()
            for mention in entity.mentions:
                try:
                    mention_head_start = char_to_words[mention.sentence_num][mention.head_span[0]]
                except KeyError:
                    # This is bad, but happens very rarely, so just skip this mention
                    continue
                try:
                    mention_head_end = char_to_words[mention.sentence_num][mention.head_span[1]] + 1
                except KeyError:
                    # If the end goes past the end, don't worry, just go to the end
                    mention_head_end = None
                # Look at the NE tags of the head NP
                ne_tags = doc_ne_tags[mention.sentence_num][mention_head_start:mention_head_end]
                # Count up how many we've seen of each
                ne_tag_counter.update(ne_tags)

            # If they're all the same (and not O), we have a NE type for this mention
            # Otherwise, take the most common NE type
            if "O" in ne_tag_counter:
                del ne_tag_counter["O"]
            if len(ne_tag_counter):
                entity_type = ne_tag_counter.most_common(1)[0][0]
                entity_type = CANDC_ENTITY_TYPES[entity_type[2:]]

        if entity_type is not None:
            # The tags are of the form I-*
            # Drop the I and look up the corresponding entity type
            entity.entity_type = entity_type


CANDC_ENTITY_TYPES = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "DAT": "date",
    "PCT": "percentage",
    "TIM": "time",
    "MON": "money",
}


class EventExtractionError(Exception):
    pass
