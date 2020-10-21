import random
import traceback

from whim.entity_narrative.chains.document import RichEventDocument, Event
from whim_common.utils.base import remove_duplicates
from whim_common.utils.files import split_sections


class MultipleChoiceQuestion(object):
    def __init__(self, context_events, choices, target_choice, entity, document=None):
        self.context_events = context_events
        self.choices = choices
        self.target_choice = target_choice
        self.entity = entity
        self.document = document

    def to_text(self):
        # XXX This doesn't handle when the document isn't given
        return """\
Entity:
%d
%s

Context:
%s

Choices:
%s

Target:
%d

Document:
%s""" % (
            self.document.entities.index(self.entity),
            self.entity,
            "\n".join([e.to_text(self.document.entities) for e in self.context_events]),
            "\n".join([e.to_text(self.document.entities) for e in self.choices]),
            self.target_choice,
            self.document.to_text(),
        )

    @staticmethod
    def from_text(text):
        # Split up the headed sections
        sections = split_sections(text.splitlines(),
                                  ["Entity:", "Context:", "Choices:", "Target:", "Document:"])
        # The first section should always be empty
        sections = sections[1:]
        # Remove the blank line from the end of each
        sections = [lines[:-1] if len(lines) and len(lines[-1]) == 0 else lines for lines in sections]
        # Entity number is all we need: it identifies the entity in the list
        entity_number = int(sections[0][0])
        # Next read in the full document, which is at the end
        document = RichEventDocument.from_text("\n".join(sections[4]))
        entity = document.entities[entity_number]
        # Read context events
        # basilwang 2020-10-21 strip E<>
        # context_events = [Event.from_text(line, document.entities) for line in sections[1]]
        context_events = ''
        for line in sections[1]:
            context_event = Event.from_text(line, document.entities).to_string_entity_text().replace('E<','').replace('>','')
            predicate = context_event[0:context_event.index('(')]
            left_part = context_event[context_event.index('('):]
            print(left_part)
            try:
                insert_index = left_part.index(',')
            except ValueError as ve:
                insert_index = len(left_part)

            context_event = left_part[0:insert_index] + ',' + predicate + left_part[insert_index:]
            context_event = context_event.replace('(','').replace(')','').replace('--','')
            context_events += context_event + ','
        context_events = context_events.replace(',',' ')
        # Multiple choices
        #choices = [Event.from_text(line, document.entities) for line in sections[2]]
        choices = []
        for line in sections[2]:
            choice = Event.from_text(line, document.entities).to_string_entity_text().replace('E<','').replace('>','')
            predicate = choice[0:choice.index('(')]
            left_part = choice[choice.index('('):]
            print(left_part)
            try:
                insert_index = left_part.index(',')
            except ValueError as ve:
                insert_index = len(left_part)

            choice = left_part[0:insert_index] + ',' + predicate + left_part[insert_index:]
            choice = choice.replace('(', '').replace(')', '').replace('--', '')
            choices.append(choice)
        # Target choice index
        target = int(sections[3][0])
        return MultipleChoiceQuestion(context_events, choices, target, entity, document=document)

    @staticmethod
    def generate_random_unbalanced(rich_doc_corpus, min_context=4, truncate_context=None, choices=5, subsample=None,
                                   stoplist=None):
        """
        Generate a multiple choice question by randomly selecting an instance from a corpus index.
        Not balanced on verb, just selected by uniformly picking a document from the corpus, then
        uniformly picking one of its chains.

        Any selections made in which the context from the preceding events is smaller than
        min_context are discarded and a new selection is made.

        If the context is larger than min_context, it's all included, unless truncate_context is
        given. In this case, only the most recent truncate_context events are included. To
        specify an exact context window size, set min_context=truncate_context.

        The choices are randomly ordered.

        """
        if subsample == 1. or subsample == 0.:
            subsample = None
        # Don't preload documents as we do in the balanced sample case
        # Here we're just iterating over the corpus in order, so there's no need
        initial_chains = iter([])
        corpus_iter = rich_doc_corpus.archive_iter(subsample=subsample)

        def chain_gen(it, min_chain_length=1, min_target_index=0):
            # Pull the next doc out of the corpus
            for archive, filename, doc in it:
                chains = doc.get_chains(stoplist=stoplist)
                # Filter out chains that are too short
                chains = [c for c in chains if len(c[1]) >= min_chain_length]
                if chains:
                    entity, chain = random.choice(chains)
                    # Pick an event at random from the chain to use as the target
                    target_index = random.randint(min_target_index, len(chain)-1)
                    yield entity, target_index, chain, doc
        # Filter out short chains here, before picking one at random, so we don't reject them unnecessarily
        chain_iter = chain_gen(corpus_iter, min_chain_length=min_context+1, min_target_index=min_context)
        # This will be called to get each chain
        chain_getter = chain_iter.next

        # To get distractors, we need to be selecting from another part of the corpus
        # We subsample at a low rate to avoid any obvious relations between consecutive distractors, but
        #  otherwise iterate over the corpus, since it's the most efficient way to pull things out of it
        def distractor_chain_gen():
            while True:
                # Pick a random spot in the corpus to start
                # We should select from the whole corpus really, but if it's a long way in it takes too long to seek
                start = random.randint(0, min(len(rich_doc_corpus)-1, 50000))
                for archive, filename, doc in rich_doc_corpus.archive_iter(subsample=0.02, start=start):
                    yield archive, filename, doc

        distractor_chain_iter = chain_gen(distractor_chain_gen())

        def distractor_getter(main_event, num_distractors):
            given = 0
            while given < num_distractors:
                entity, target_index, chain, doc = distractor_chain_iter.next()
                if chain[target_index].verb_lemma != main_event.verb_lemma:
                    yield entity, target_index, chain, doc
                    given += 1

        return MultipleChoiceQuestion._generate_random(initial_chains, chain_getter, distractor_getter,
                                                       min_context=min_context, truncate_context=truncate_context,
                                                       choices=choices)

    @staticmethod
    def generate_random_balanced_on_verb(verbs_index, min_context=4, truncate_context=None, choices=5, preload=1,
                                         verb_subset=None):
        """
        Generate a multiple choice question by randomly selecting an instance from a corpus index.
        First pick a uniform random predicate, then uniformly selects an instance of that in a
        document in the corpus. Then (choices-1) distractors are selected from the corpus,
        following the same procedure, but replacing any entities they have as arguments with
        randomly chosen entities from the document context of the target.

        Any selections made in which the context from the preceding events is smaller than
        min_context are discarded and a new selection is made.

        If the context is larger than min_context, it's all included, unless truncate_context is
        given. In this case, only the most recent truncate_context events are included. To
        specify an exact context window size, set min_context=truncate_context.

        The choices are randomly ordered.

        """
        # Start by asking the index to cache a large bundle of chains, so that it's faster to load the documents
        if verb_subset:
            verb_vocab = verb_subset
        else:
            verb_vocab = verbs_index.keys()

        # Generally best to load more targets than we expect to need at this stage, as some of them will be rejected
        # and it will be slower to load more later
        target_verbs = [random.choice(verb_vocab) for i in range(int(preload))]
        # Note that looking all of these up at once allows faster bulk lookup for tarred corpora
        initial_chains = iter(verbs_index.random_chains_for_verbs(target_verbs))

        def chain_getter():
            # This will be called to get more chains if the above list runs out
            extra_verb = random.choice(verb_vocab)
            return verbs_index.random_chain_for_verb(extra_verb)

        def distractor_getter(main_event, num_distractors):
            # Select some other events at random to serve as distractors
            possible_distractor_verbs = list(verbs_index.keys())
            # Don't include the target itself
            possible_distractor_verbs.remove(main_event.verb_lemma)
            distractor_verbs = random.sample(possible_distractor_verbs, num_distractors-1)
            for verb in distractor_verbs:
                yield verbs_index.random_chain_for_verb(verb)

        return MultipleChoiceQuestion._generate_random(initial_chains, chain_getter, distractor_getter,
                                                       min_context=min_context, truncate_context=truncate_context,
                                                       choices=choices)

    @staticmethod
    def _generate_random(initial_chains, chain_getter, distractor_getter, min_context=4, truncate_context=None,
                         choices=5):
        # Just keep generating as long as the caller wants more
        rejected_too_short = 0

        try:
            while True:
                try:
                    # Randomly select an instance of an event using this verb and load the document/chain
                    entity, target_index, chain, document = initial_chains.next()
                except StopIteration:
                    # We've used up everything we loaded
                    # Add a new data point one by one
                    entity, target_index, chain, document = chain_getter()

                chain_to_target = chain[:target_index+1]
                # Get rid of any duplicate events, taking care to keep the target event
                chain_to_target = list(reversed(remove_duplicates(reversed(chain_to_target),
                                                                  key=lambda e: e.verb_lemma)))
                if len(chain_to_target)-1 < min_context:
                    # Not enough context: pick again
                    rejected_too_short += 1
                    continue
                context = chain_to_target[:-1]
                if truncate_context:
                    context = context[-truncate_context:]
                target_event = chain_to_target[-1]

                # Get the non-protagonist entities from the document ready to
                # substitute in for non-protags in distractors
                non_protagonist_entities = [e for e in document.entities if e is not entity]
                if len(non_protagonist_entities) == 0:
                    # Only one entity in the document! We'll have to use that for all distractor entities
                    non_protagonist_entities = [entity]

                # Find event instances of these
                options = [target_event]
                for dis_chain_entity, dis_index, dis_chain, dis_document in distractor_getter(target_event, choices-1):
                    distractor_event = dis_chain[dis_index]

                    # Replace any entities in the event with entities from the current document
                    distractor_entities = distractor_event.get_entities()
                    for distractor_entity in distractor_entities:
                        if distractor_entity is dis_chain_entity:
                            # This is the chain's protagonist: replace with our fictional protagonist
                            distractor_event.substitute_entity(dis_chain_entity, entity)
                        else:
                            # Not a protagonist originally: replace with a non-protagonist
                            distractor_event.substitute_entity(distractor_entity, random.choice(non_protagonist_entities))

                    options.append(distractor_event)

                # Shuffle to make sure we can't infer target from ordering!
                random.shuffle(options)
                # Look up the new index of the target event
                target_index = (i for (i, event) in enumerate(options) if event is target_event).next()

                yield MultipleChoiceQuestion(context, options, target_index, entity, document=document)
        except StopIteration:
            # Shouldn't happen
            traceback.print_exc()
            print("rejected %d short chains" % rejected_too_short)
            raise ValueError("random generation stopped!")