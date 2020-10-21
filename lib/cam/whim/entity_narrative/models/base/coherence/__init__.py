class CoherenceScorer(object):
    """
    Abstract superclass for narrative chain models that provide a coherence score.

    These models need to be able to assign a coherence score to a given narrative chain that is
    comparable across different chains (i.e. normalized for chain length, etc).

    """
    def chain_coherence(self, entity, events):
        raise NotImplementedError("CoherenceScorer type %s should implement chain_coherence()" % type(self).__name__)

    def chain_coherences(self, chains, batch_size=1000):
        """
        Default implementation just iterates over chains, calling chain_coherence() for each. May be
        overridden if the subclass has a better way to do this -- for example, if it's faster to do
        it in batches.

        Chains may be any iterable, including a generator. Result is a generator that yields a float
        score for each chain.

        Each value in chains should be of the form ((entity, events), data), where data is any other data
        that will be yielded along with the coherence score to identify the chain.

        batch_size may or may not make a difference, depending on the implementation

        """
        for (entity, events), data in chains:
            yield self.chain_coherence(entity, events), data
