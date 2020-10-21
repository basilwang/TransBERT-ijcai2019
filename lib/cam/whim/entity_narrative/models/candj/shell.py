from operator import itemgetter

from cam.whim.entity_narrative.chains.document import predicate_relation
from cam.whim.entity_narrative.shell.commands import ModelShell


class CandjShell(ModelShell):
    def do_neighbours(self, line, **kwargs):
        entities, events, line = ModelShell.parse_event_context(line)
        if line.strip():
            print "Ignoring remainder of input: %s" % line

        preds = [predicate_relation(entities[0], event) for event in events]

        # Score all events in the vocabulary
        pmis = list(reversed(sorted(
            [(vocab_ev, sum(self.model.pmi(vocab_ev, context_ev) for context_ev in preds))
             for vocab_ev in self.model.event_counts.keys()], key=itemgetter(1))))

        for event, score in pmis[:10]:
            if score == 0.:
                break
            print event, score
