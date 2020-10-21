import logging
from cam.whim.entity_narrative.models.base.train import NarrativeChainModelTrainer
from gensim.models.word2vec import Word2Vec


class Word2VecTrainer(NarrativeChainModelTrainer):
    def prepare_arguments(self, parser):
        """ Add arguments to an argparse parser for training. """
        parser.add_argument("--vector-size", type=int, default=100,
                            help="Size of vector representation to learn (default: 100)")
        parser.add_argument("--threshold", type=int, default=20,
                            help="Minimum number of counts for a predicate to be included (default: 20)")
        parser.add_argument("--window", type=int, default=5,
                            help="Context window size (default: 5)")
        parser.add_argument("--with-args", action="store_true",
                            help="Include arguments in the chains just as if they're other words")
        parser.add_argument("--adj", action="store_true",
                            help="Handle predicative adjective events properly, i.e. as distinct predicates. "
                                 "Only not the default for backwards compatibility")
        parser.add_argument("--trans", action="store_true",
                            help="Include transitivity markers in predicate representation")

    def train(self, model_name, corpus, log, opts, chain_features=None):
        from cam.whim.entity_narrative.models.word2vec.model import Word2VecNarrativeChainModel

        training_metadata = {
            "data": corpus.directory,
        }
        # Set up logging so the word2vec trainer outputs useful info
        logging.basicConfig(format='%(asctime)s - word2vec internal - %(levelname)s - %(message)s', level=logging.INFO)

        # Start the training algorithm
        log.info("Training word2vec")
        corpus_it = CorpusIterator(corpus, with_args=opts.with_args,
                                   predicative_adjectives=opts.adj,
                                   transitivity=opts.trans)
        if opts.with_args:
            log.info("Including arguments as words")
        word2vec = Word2Vec(corpus_it, size=opts.vector_size, min_count=opts.threshold)

        log.info("Saving model: %s" % model_name)
        model = Word2VecNarrativeChainModel(word2vec, training_metadata=training_metadata,
                                            with_args=opts.with_args,
                                            predicative_adjectives=opts.adj,
                                            transitivity=opts.trans)
        model.save(model_name)
        return model


class CorpusIterator(object):
    """
    Iterator to prepare a sentence for each chain in a corpus.

    """
    def __init__(self, corpus, with_args=False, predicative_adjectives=False, transitivity=False):
        self.transitivity = transitivity
        self.predicative_adjectives = predicative_adjectives
        self.with_args = with_args
        self.corpus = corpus

    def __iter__(self):
        from cam.whim.entity_narrative.models.word2vec.model import Word2VecNarrativeChainModel

        if self.with_args:
            word_gen = Word2VecNarrativeChainModel.extract_chain_word_lists_with_args
        else:
            word_gen = Word2VecNarrativeChainModel.extract_chain_word_lists

        for doc in self.corpus:
            for chain_words in word_gen(doc.get_chains(),
                                        predicative_adjectives=self.predicative_adjectives,
                                        transitivity=self.transitivity):
                yield chain_words
