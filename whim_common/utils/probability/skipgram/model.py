import numpy
import cPickle as pickle


class SkipBigramModel(object):
    def __init__(self, counts, vocab_size, laplace_smoothing=1.0):
        self.counts = counts
        self.vocab_size = vocab_size
        self.laplace_smoothing = laplace_smoothing

        # Precompute the conditional probabilities
        self.context_counts = counts.sum(axis=1)
        self.conditional_probs = (counts.astype(numpy.float64) + laplace_smoothing) / \
                                 (self.context_counts + laplace_smoothing * vocab_size)

        # Compute the prior distribution for when no context is available
        self.prior = self.context_counts.sum(axis=0).astype(numpy.float64)
        self.prior /= self.prior.sum()

    def predict_from_context(self, context_words, average=False):
        """
        If average=True, averages the conditional probability conditioned on each context event.

        Otherwise, sums counts from all the context events. This is probably a better thing to do, since it
        gives more weight to things we have more counts for.

        """
        # Skip any context words that have 0 denominators
        context_words = [w for w in context_words if self.context_counts[w] > 0]
        if average:
            return numpy.mean(self.conditional_probs[context_words, :], axis=0)
        else:
            return (self.counts[context_words, :].astype(numpy.float64) + self.laplace_smoothing).sum(axis=0) / \
                   (self.context_counts[context_words] + self.laplace_smoothing * self.vocab_size).sum()

    def probability(self, word, context):
        return self.conditional_probs[context, word]

    @staticmethod
    def train(sequences, vocab_size, **kwargs):
        """
        Train a model on an iterable containing lists of IDs. These should be integer IDs of
        items in the vocab, so the maximum number should be the size of the vocab - 1.

        """
        counts = numpy.zeros((vocab_size, vocab_size), dtype=numpy.int)

        # Get counts from every sequence in the corpus
        for sequence in sequences:
            for i, word0 in enumerate(sequence[:-1]):
                # Count all the pairs that begin with this word
                for word1 in sequence[i+1:]:
                    counts[word0, word1] += 1
        return SkipBigramModel(counts, vocab_size, **kwargs)

    def save(self, filename):
        numpy.save("%s.npy" % filename, self.counts)
        with open("%s.params" % filename, 'w') as params_file:
            pickle.dump({
                "vocab_size": self.vocab_size,
                "laplace_smoothing": self.laplace_smoothing,
            }, params_file)

    @staticmethod
    def load(filename):
        counts = numpy.load("%s.npy" % filename)
        with open("%s.params" % filename, 'r') as params_file:
            params = pickle.load(params_file)
        return SkipBigramModel(counts, params['vocab_size'], laplace_smoothing=params['laplace_smoothing'])