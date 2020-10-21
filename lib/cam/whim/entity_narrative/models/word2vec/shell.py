import numpy
from cam.whim.entity_narrative.models.base.predict.shell import NarrativePredictorShell
from cam.whim.entity_narrative.shell.commands import ModelShell


class Word2VecShell(ModelShell, NarrativePredictorShell):
    def do_repr(self, line, **kwargs):
        entities, events, line = ModelShell.parse_event_context(line)
        # Get string representation of events
        chain_words = list(self.model.word_generator([(entities[0], events)]))[0]

        for word in chain_words:
            if word in self.model.word2vec:
                print word
            else:
                print "Not in vocab: %s" % word

    def do_neighbours(self, line, **kwargs):
        limit = 10
        entities, events, line = ModelShell.parse_event_context(line, allow_arg_only=True)
        if line.strip():
            print "Ignoring remainder of input: %s" % line
        predicates_only = "pred" in kwargs

        arg_words = ["arg:%s" % event[1] for event in events if type(event) is tuple]
        events = [e for e in events if type(e) is not tuple]

        # Get string representation of events
        chain_words = list(self.model.word_generator([(entities[0], events)]))[0] + arg_words
        if self.model.model_options["cosmul"]:
            dists = self.model.word2vec.most_similar_cosmul(positive=chain_words, topn=None)
        else:
            dists = self.model.word2vec.most_similar(positive=chain_words, topn=None)

        best = numpy.argsort(dists)[::-1]
        returned = 0
        for index in best:
            word = self.model.word2vec.index2word[index]

            if word in chain_words:
                # Ignore (don't return) words from the input
                continue
            if predicates_only and word.startswith("arg:"):
                continue

            print "%s  (%g)" % (word, dists[index])
            returned += 1
            if returned >= limit:
                break