from whim.entity_narrative import predicate_relation
from whim.entity_narrative.models.base.train import NarrativeChainModelTrainer
from whim_common.utils.probability.markov.bigram.model import BigramModel
from whim_common.utils.progress import get_progress_bar
from gensim.corpora.dictionary import Dictionary


class BigramTrainer(NarrativeChainModelTrainer):
    def prepare_arguments(self, parser):
        """ Add arguments to an argparse parser for training. """
        parser.add_argument("--backoff-threshold", type=int,
                            help="Threshold count of context for backing off to unigram dist", default=0)
        parser.add_argument("--laplace", type=float, help="Laplace smoothing amount (default: 0.1)", default=0.1)
        parser.add_argument("--threshold", type=int,
                            help="Threshold to apply to infrequent events during training (default: 10)", default=10)

    def train(self, model_name, corpus, log, opts, chain_features=None):
        from whim.entity_narrative.models.bigram.model import BigramNarrativeChainModel

        log.info("Building dictionary")
        corpus_it = CorpusIterator(corpus, progress="Counting")
        # Filter out infrequent events from the vocabulary
        # basilwang todo should uncomment this where use large corpus
        corpus_it.dictionary.filter_extremes(no_below=opts.threshold, no_above=1.0, keep_n=None)

        log.info("Training bigram model")
        corpus_it.progress = "Training"
        bigram_model = BigramModel.train(corpus_it, len(corpus_it.dictionary),
                                         laplace_smoothing=opts.laplace,
                                         backoff_threshold=opts.backoff_threshold)
        log.info("Bigram model training complete")

        log.info("Saving as %s" % model_name)
        model = BigramNarrativeChainModel(bigram_model, corpus_it.dictionary)
        model.save(model_name)
        return model


class CorpusIterator(object):
    def __init__(self, corpus, progress=None):
        self.progress = progress
        self.corpus = corpus
        # Go once over the corpus to build the dictionary
        self.dictionary = Dictionary(self.feature_iter())

    def feature_iter(self):
        pbar = None
        if self.progress:
            pbar = get_progress_bar(len(self.corpus), title=self.progress)

        try:
            for doc_num, document in enumerate(self.corpus):
                if pbar:
                    pbar.update(doc_num)

                for entity, events in document.get_chains():
                    yield [predicate_relation(entity, e) for e in events]
        finally:
            if pbar:
                pbar.finish()

    def __iter__(self):
        for features in self.feature_iter():
            yield [self.dictionary.token2id[pred] for pred in features if pred in self.dictionary.token2id]