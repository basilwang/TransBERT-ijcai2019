
class Tags(object):
    def __init__(self, words, lemmas, pos_tags, chunk_tags, ne_tags, supertags):
        self.words = words
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.chunk_tags = chunk_tags
        self.ne_tags = ne_tags
        self.supertags = supertags

    @staticmethod
    def from_text(line):
        line = line.strip("\n")
        # Split up the tags
        tokens = line.split()

        # We need a special case for where the word itself is '|' and we end up with double |s
        # No idea why anyone would use this in a text, but it comes up
        tokens = [t.replace("||", "|") for t in tokens]
        # Split up the tags
        # If there are too many |s, they get interpreted as part of the word (which they probably are)
        tokens = [t.rsplit("|", 5) for t in tokens]

        if not all(len(t) == 6 for t in tokens):
            incorrect = [t for t in tokens if len(t) != 6]
            raise TagsReadError("could not split up tags correctly in line: %s. Problem with words: %s" %
                                (line, " and ".join("|".join(toks) for toks in incorrect)))

        words, lemmas, pos_tags, chunk_tags, ne_tags, supertags = zip(*tokens)
        return Tags(words, lemmas, pos_tags, chunk_tags, ne_tags, supertags)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as tags_file:
            return [Tags.from_text(line) for line in tags_file]

    def __str__(self):
        return "Tags<%s>" % " ".join("%s|%s" % (word, tag) for (word, tag) in zip(self.words, self.pos_tags))


class TagsReadError(Exception):
    pass