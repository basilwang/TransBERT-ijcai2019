from whim_common.utils.language import strip_punctuation, load_stopwords


stopwords = load_stopwords()


def escape(text, space_slashes=False):
    """
    Escaping used for entity/mention storage and also some other things.
    Default behaviour is slightly more generic than the usage for entities.
        ';'  ->  '@semicolon@'
        '/'  ->  '@slash@'
        ','  ->  '@commma@'

    Entity-style (enforced by space_slashes=True) only replaces /s surrounded by spaces
    and has a special double-slash replacement.

    """
    # Various substitutions made for output
    text = text.replace(";", "@semicolon@")
    text = text.replace(",", "@comma@")
    if space_slashes:
        text = text.replace(" / ", "@slash@")
        text = text.replace(" // ", "@slashes@")
    else:
        text = text.replace("/", "@slash@")
    return text


def unescape(text, space_slashes=False):
    # Reverse various substitutions made on output
    text = text.replace("@semicolon@", ";")
    text = text.replace("@comma@", ",")
    if space_slashes:
        text = text.replace("@slash@", " / ")
        text = text.replace("@slashes@", " // ")
    else:
        text = text.replace("@slash@", "/")
    return text


class Mention(object):
    def __init__(self, char_span, text, np_sentence_position, np_doc_position, nps_in_sentence, sentence_num,
                 head_span):
        self.char_span = char_span
        self.text = text
        self.np_sentence_position = np_sentence_position
        self.np_doc_position = np_doc_position
        self.nps_in_sentence = nps_in_sentence
        self.sentence_num = sentence_num
        self.head_span = head_span

    def to_text(self):
        return "(%d,%d);%s;%d;%d;%d;%d;(%d,%d)" % (
            self.char_span[0], self.char_span[1],
            escape(self.text, space_slashes=True),
            self.np_sentence_position,
            self.np_doc_position,
            self.nps_in_sentence,
            self.sentence_num,
            self.head_span[0], self.head_span[1]
        )

    @staticmethod
    def from_text(string):
        parts = [p.strip() for p in string.split(";")]
        if len(parts) != 7:
            raise EntityReadError("expected 7 parts, separated by ;, got %d: %s" % (len(parts), string))
        # First part is (start,end) char span of mention
        start, end = parts[0].strip("()").split(",")
        char_span = int(start), int(end)
        # Next is the mention text
        text = unescape(parts[1], space_slashes=True)
        # Next come various integer indices and such
        np_sentence_position = int(parts[2])
        np_doc_position = int(parts[3])
        nps_in_sentence = int(parts[4])
        # The index of the sentence the mention appears in
        sentence_num = int(parts[5])
        # And another span, this time that of the head
        start, end = parts[6].strip("()").split(",")
        head_span = int(start), int(end)

        return Mention(char_span, text, np_sentence_position, np_doc_position, nps_in_sentence, sentence_num, head_span)

    @staticmethod
    def from_mention_text(string):
        return Mention((0, len(string)), string, 0, 0, 1, 0, (0, len(string)))

    def __str__(self):
        return "M<(%d,%d):%s>" % (self.char_span[0], self.char_span[1], self.text)

    def __repr__(self):
        return str(self)


class Entity(object):
    def __init__(self, mentions, category=None, gender=None, gender_prob=None, number=None, number_prob=None,
                 entity_type=None):
        self.entity_type = entity_type
        self.mentions = mentions
        self.category = category
        self.gender = gender
        self.gender_prob = gender_prob
        self.number = number
        self.number_prob = number_prob

        # Pick a text that we'll use to represent this entity
        mention_texts = [m.text for m in self.mentions]
        # Filter out pronouns
        mention_texts = [m for m in mention_texts if m.lower() not in PRONOUNS]
        if len(mention_texts) == 0:
            # No non-pronouns, not much we can do
            self.representative_text = self.mentions[0].text
        else:
            # If we have non-pronoun text, take the shortest mention (within reason)
            if not any(len(t) > 3 for t in mention_texts):
                # Only very short ones, take the longest
                self.representative_text = max(mention_texts, key=lambda t: len(t))
            else:
                self.representative_text = min([t for t in mention_texts if len(t) > 3], key=lambda t: len(t))
        # Finally, some hackery on the text to make it look nicer
        # Often we end up with possessives among the mentions: try to get rid of these
        if self.representative_text.endswith("'s"):
            self.representative_text = self.representative_text[:-2].strip()
        elif self.representative_text.endswith("'"):
            self.representative_text = self.representative_text[:-1].strip()

        self.non_pronoun_mention_texts = mention_texts

    def to_text(self):
        return "category=%s // gender=%s // genderProb=%f // number=%s // numberProb=%f // mentions=%s%s" % (
            self.category,
            self.gender,
            self.gender_prob,
            self.number,
            self.number_prob,
            " / ".join(m.to_text() for m in self.mentions),
            " // type=%s" % self.entity_type if self.entity_type is not None else "",
        )

    @staticmethod
    def from_text(string):
        """
        Build an entity from its textual representation (usually a single line of the input file).

        """
        parts = [p.strip() for p in string.split(" // ")]
        if len(parts) < 6 or len(parts) > 7:
            raise EntityReadError("expected 6 or 7 parts, separated by //, got %d: %s" % (len(parts), string))
        # Category field is never filled, for some reason
        category = field_value("category", parts[0])
        gender = field_value("gender", parts[1])
        gender_prob = float(field_value("genderProb", parts[2]))
        number = field_value("number", parts[3])
        number_prob = float(field_value("numberProb", parts[4]))
        mentions_text = field_value("mentions", parts[5])
        # Now split up the mentions
        mentions = [
            Mention.from_text(men_text) for men_text in mentions_text.split(" / ")
        ]
        if len(parts) == 7:
            # There's an entity type
            entity_type = field_value("type", parts[6])
        else:
            entity_type = None
        return Entity(mentions,
                      category=category, gender=gender, gender_prob=gender_prob,
                      number=number, number_prob=number_prob, entity_type=entity_type)

    @staticmethod
    def read_entity_file(f):
        try:
            return [Entity.from_text(line) for line_num, line in enumerate(f)]
        except EntityReadError as err:
            raise EntityReadError("error reading entity file (line %d): %s [whole line: %s]" % (line_num, err, line))

    def __str__(self):
        return "E<%s>" % self.representative_text

    def __repr__(self):
        return str(self)

    def get_head_word(self):
        """
        Retrieve a head word from the entity's mentions if possible. Returns None if no suitable head
        word can be found: e.g., if all mentions are pronouns.

        """
        entity_head_words = set()
        # Gather a head word, if possible, from each mention
        for mention in self.mentions:
            mention_start = mention.char_span[0]
            mention_head = mention.text[mention.head_span[0]-mention_start:mention.head_span[1]-mention_start].lower()
            # Process the head phrase a bit
            # Remove punctuation
            mention_head = strip_punctuation(mention_head)
            # Get rid of words that won't help us: stopwords and pronouns
            head_words = mention_head.split()
            head_words = [w for w in head_words if w not in stopwords and w not in PRONOUNS]
            # Don't use any 1-letter words
            head_words = [w for w in head_words if len(w) > 1]
            # If there are no words left, we can't get a headword from this mention
            # If there are multiple (a minority of cases), use the rightmost, which usually is the headword
            if head_words:
                entity_head_words.add(head_words[-1])
        # If we've ended up with multiple possible head words (minority, but not uncommon), we've no way to choose
        # We could just pick one randomly
        # Take the lexicographic first, just to be consistent
        if len(entity_head_words):
            return list(sorted(entity_head_words))[0]
        else:
            return None

    def get_short_mention(self):
        """
        Pick out a mention text by first filtering out all that are only pronouns and
        then taking the shortest remaining. Returns None if there are no non-pronoun mentions.
        """
        # Filter out any mentions that are entirely pronouns
        mentions = [
            m for m in self.mentions if not all(w.lower() in PRONOUNS for w in m.text.split())
        ]
        # Take the remaining mention with the fewest words
        if len(mentions):
            return min(mentions, key=lambda m: len(m.text.split()))
        else:
            return None


def field_value(expected_name, text):
    if not text.startswith("%s=" % expected_name):
        raise EntityReadError("expected to find a %s-field in place of: %s" % (expected_name, text))
    return text[len(expected_name)+1:]


class EntityReadError(Exception):
    pass


# Handy list of pronouns so we can pick out a meaningful text to represent an entity
PRONOUNS = [
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "myself", "yourself", "himself", "herself", "itself", "ourself", "ourselves", "themselves",
    "my", "your", "his", "its", "it's", "our", "their",
    "mine", "yours", "ours", "theirs",
    "this", "that", "those", "these"
]