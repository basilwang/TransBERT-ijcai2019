import argparse

from whim_common.utils.gensim.data import MultiFileTextCorpus
from whim_common.utils.gensim.lda.storage import save_lda_model
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LDA model using Gensim")
    parser.add_argument("model_name", help="Name to store the LDA model under")
    parser.add_argument("text_archives", nargs="+",
                        help="Directories containing tokenized text to get the original documents from")
    parser.add_argument("--topics", type=int, default=20,
                        help="Number of topics to use (default: 20)")
    parser.add_argument("--processes", type=int, default=1,
                        help="Number of processes to use during LDA training (default: 1)")
    opts = parser.parse_args()

    print "Building dictionary for dataset"
    print "Reading data from\n  %s" % "\n  ".join(opts.text_archives)
    corpus = MultiFileTextCorpus(opts.text_archives)
    if len(corpus) == 0:
        raise ValueError("no texts found in %s: looking for *.txt files in archive dir and subdirs" %
                         opts.text_archives)
    print "  %d texts" % len(corpus)

    print "Training gensim LDA model..."
    lda = LdaMulticore(corpus, num_topics=opts.topics, workers=opts.processes, id2word=corpus.dictionary)
    print "Saving model %s" % opts.model_name
    save_lda_model(lda, opts.model_name)
