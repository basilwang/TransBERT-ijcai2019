import os
from scipy.sparse.csr import csr_matrix
from scipy.sparse.base import issparse
import numpy
import cPickle as pickle


class BigramModel(object):
    """
    Non-Bayesian, non-hidden Markov model, plain and simple.

    """
    def __init__(self, initial_counts, counts, laplace_smoothing=1.0, backoff_threshold=0):
        self.counts = counts
        self.initial_counts = initial_counts
        self.vocab_size = initial_counts.shape[0]
        self.laplace_smoothing = laplace_smoothing
        if backoff_threshold is None:
            backoff_threshold = 0
        self.backoff_threshold = backoff_threshold

        # Precompute the conditional probabilities
        self.conditional_probs = (counts.astype(numpy.float64) + laplace_smoothing) / \
                                 (counts.sum(axis=1) + laplace_smoothing * self.vocab_size)

        # Compute the distribution for the first elements
        self.initial_probs = initial_counts.astype(numpy.float64)
        self.initial_probs /= self.initial_probs.sum()

        # Compute the unigram distribution
        self.unigram_probs = counts.astype(numpy.float64).sum(axis=0)
        self.unigram_probs += laplace_smoothing
        self.unigram_probs /= self.unigram_probs.sum()

        # Sum up how many times each event has featured as the context
        self.context_counts = counts.sum(axis=1)

    def probability_dist(self, context=None, override_backoff_threshold=None):
        if override_backoff_threshold is None:
            backoff_threshold = self.backoff_threshold
        else:
            backoff_threshold = override_backoff_threshold

        if context is None:
            return self.initial_probs
        if self.context_counts[context] < backoff_threshold:
            return self.unigram_probs
        else:
            return self.conditional_probs[context, :]

    @staticmethod
    def train(sequences, vocab_size, **kwargs):
        """
        Train a model on an iterable containing lists of IDs. These should be integer IDs of
        items in the vocab, so the maximum number should be the size of the vocab - 1.

        """
        initial_counts = numpy.zeros(vocab_size, dtype=numpy.int)
        counts = numpy.zeros((vocab_size, vocab_size), dtype=numpy.int)

        # Get counts from every sequence in the corpus
        for sequence in sequences:
            if len(sequence) > 0:
                # Count the first element
                initial_counts[sequence[0]] += 1

            for i, word0 in enumerate(sequence[:-1]):
                # Count the transition
                counts[word0, sequence[i+1]] += 1
        return BigramModel(initial_counts, counts, **kwargs)

    @staticmethod
    def train_on_pairs(pairs, vocab_size, **kwargs):
        """
        Train a model on an iterable containing lists of bigrams as pairs of IDs.
        These should be integer IDs of items in the vocab, so the maximum number
        should be the size of the vocab - 1.

        The unigram distribution is used as the initial distribution in this case.

        """
        counts = numpy.zeros((vocab_size, vocab_size), dtype=numpy.int)
        # Get counts from all the pairs
        for w0, w1 in pairs:
            # Count the pair
            counts[w0, w1] += 1
        return BigramModel(counts.sum(axis=0), counts, **kwargs)

    def save(self, filename):
        # Store counts as a sparse matrix so they don't take up tonnes of space
        if issparse(self.counts):
            sparse_counts = self.counts
        else:
            sparse_counts = csr_matrix(self.counts)
        with open("%s-sparse-counts" % filename, "w") as f:
            pickle.dump(sparse_counts, f, -1)

        numpy.save("%s-init.npy" % filename, self.initial_counts)
        with open("%s.params" % filename, 'w') as params_file:
            pickle.dump({
                "laplace_smoothing": self.laplace_smoothing,
                "backoff_threshold": self.backoff_threshold,
            }, params_file)

    @staticmethod
    def load(filename):
        if not os.path.exists("%s-sparse-counts" % filename):
            # Allow non-sparse loading for backwards compatibility
            counts = numpy.load("%s-counts.npy" % filename)
        else:
            with open("%s-sparse-counts" % filename, "r") as f:
                counts = pickle.load(f)
            counts = counts.todense()

        initial_counts = numpy.load("%s-init.npy" % filename)
        with open("%s.params" % filename, 'r') as params_file:
            params = pickle.load(params_file)
        return BigramModel(initial_counts, counts,
                           laplace_smoothing=params['laplace_smoothing'], backoff_threshold=params['backoff_threshold'])