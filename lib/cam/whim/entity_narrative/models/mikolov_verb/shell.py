from cam.whim.entity_narrative.shell.commands import ModelShell


class Word2VecVerbShell(ModelShell):
    def do_repr(self, line, **kwargs):
        from cam.whim.entity_narrative.models.mikolov_verb.model import MikolovVerbNarrativeChainModel

        entities, events, line = ModelShell.parse_event_context(line)
        # Get string representation of events
        chain_words = list(MikolovVerbNarrativeChainModel.extract_chain_word_lists([(entities[0], events)]))[0]

        for word in chain_words:
            if word in self.model.word2vec:
                print word
            else:
                print "Not in vocab: %s" % word

    def do_neighbours(self, line, **kwargs):
        from cam.whim.entity_narrative.models.mikolov_verb.model import MikolovVerbNarrativeChainModel

        entities, events, line = ModelShell.parse_event_context(line)
        if line.strip():
            print "Ignoring remainder of input: %s" % line

        # Get string representation of events
        chain_words = list(MikolovVerbNarrativeChainModel.extract_chain_word_lists([(entities[0], events)]))[0]
        if self.model.model_options["cosmul"]:
            neighbours = self.model.word2vec.most_similar_cosmul(positive=chain_words)
        else:
            neighbours = self.model.word2vec.most_similar(positive=chain_words)

        for predicate, sim in neighbours:
            print "%s  (%g)" % (predicate, sim)