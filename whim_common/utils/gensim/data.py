from glob import iglob
from itertools import chain
from operator import itemgetter
import os
import string
from whim_common.utils.base import find_duplicates
from whim_common.utils.language import load_stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus


class MultiFileTextCorpus(TextCorpus):
    """
    Used for loading, e.g., Gigaword from disk.

    """
    def __init__(self, input=None, dictionary=None, build_dictionary=True):
        self.length = None
        self._trans_table = string.maketrans("", "")
        self.stopwords = load_stopwords()
        if dictionary is not None:
            # Use the given dictionary: don't build it from the corpus
            self.dictionary = dictionary
            super(MultiFileTextCorpus, self).__init__(None)
            self.input = input
        elif not build_dictionary:
            # Don't build the dictionary now: start with an empty one
            self.dictionary = Dictionary()
            super(MultiFileTextCorpus, self).__init__(None)
            self.input = input
        else:
            super(MultiFileTextCorpus, self).__init__(input)

        # Build an index of documents so we can look up their texts
        doc_to_filename = [
            (self._filename_to_document_name(filename), filename) for filename in self.get_input_filenames()
        ]
        self._doc_index = dict(doc_to_filename)
        # Make sure we've not got multiple files with the same doc name
        if len(doc_to_filename) > len(self._doc_index):
            # Find the duplicates
            duplicate_docs = find_duplicates(doc_to_filename, key=itemgetter(0))
            raise ValueError("duplicate document names in corpus: %s" %
                             ", ".join("%s (%s)" % doc_fn for doc_fn in duplicate_docs))
        self.length = len(self._doc_index)

    def get_input_filenames(self):
        if type(self.input) is list:
            dirs = self.input
        else:
            dirs = [self.input]

        globs = []
        for dir in dirs:
            # Input is a directory
            # Look for text files in several levels of subdir
            globs.append(iglob(os.path.join(dir, "*", "*", "*.txt")))
            globs.append(iglob(os.path.join(dir, "*", "*.txt")))
            globs.append(iglob(os.path.join(dir, "*.txt")))
        # Iterate over all txt files in the directory
        return chain(*globs)

    def getstream(self):
        # Iterate over all txt files in the directory
        for filename in self.get_input_filenames():
            with open(filename, 'r') as infile:
                # Yield to whole file as a document
                yield infile.read()

    @staticmethod
    def _filename_to_document_name(filename):
        return os.path.splitext(os.path.basename(filename))[0]

    def get_document_names(self):
        """
        Document names are the basenames of the input files, without extension.

        """
        for filename in self.get_input_filenames():
            yield self._filename_to_document_name(filename)

    def get_document_words(self, text):
        # Preprocessing
        text = text.translate(self._trans_table,  string.punctuation)
        text = text.lower()
        # Split words on spaces (tokenization should already be done)
        words = text.split()
        # Remove stopwords and 1- or 2-letter words
        words = [w for w in words if w not in self.stopwords and len(w) > 2]
        return words

    def get_texts(self):
        text_num = -1
        texts = self.getstream()
        # Each file contains a single text
        for text_num, text in enumerate(texts):
            words = self.get_document_words(text)

            if self.metadata:
                yield words, (text_num,)
            else:
                yield words
        self.length = text_num + 1  # will be 0 if loop never executes

    def get_document_by_name(self, name, text=False):
        """
        Look a document up in the index and load its text.

        If text=True, return the string content. Otherwise, standard processing is applied to
        get a list of words.

        """
        if name.endswith(".txt"):
            name = name[:-4]

        filename = self.get_document_filename(name)
        with open(filename, 'r') as infile:
            data = infile.read()

        if text:
            return data
        else:
            return self.get_document_words(data)

    def get_document_filename(self, name):
        if name not in self._doc_index:
            raise KeyError("document %s not found in corpus" % name)
        return self._doc_index[name]
