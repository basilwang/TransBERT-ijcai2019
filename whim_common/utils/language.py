import os
import re
import string


def load_stopwords():
    stopwords_filename = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_stopwords.txt"))
    with open(stopwords_filename, 'r') as stopwords_file:
        return stopwords_file.read().splitlines()


RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation))


def strip_punctuation(s, split_words=True):
    if split_words:
        return RE_PUNCT.sub(" ", s)
    else:
        return RE_PUNCT.sub("", s)