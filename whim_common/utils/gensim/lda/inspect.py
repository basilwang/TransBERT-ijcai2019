import argparse
import sys

from whim_common.utils.gensim.lda.storage import load_lda_model
from whim_common.utils.gensim.lda.utils import format_topics


def inspect_lda_model(lda, topic=None, prior_reduce=0.75, topics=True):
    if topic is not None and topic >= lda.num_topics:
        print "Topic %d out of range" % opts.topic
        sys.exit(1)

    if topic is not None:
        print "Topic %d: %s" % (topic, format_topics(lda, [topic], prior_reduce=prior_reduce)[0])

    if topics:
        print "\n".join(
            "Topic %d: %s" % (t, topic) for (t, topic) in enumerate(format_topics(lda, list(range(lda.num_topics)),
                                                                                  prior_reduce=prior_reduce))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examine a trained LDA model")
    parser.add_argument("model_name", help="Name to store the LDA model under")
    parser.add_argument("--topic", type=int, help="Print out a topic's word distribution")
    parser.add_argument("--topics", action="store_true", help="Print out all topics' distributions")
    parser.add_argument("--prior-reduce", type=float, default=0.75,
                        help="Weight down words that have a high prior probability when representing topics "
                             "(default: 0.75)")
    opts = parser.parse_args()

    # Load the LDA model
    lda = load_lda_model(opts.model_name)
    print "Loading LDA model with %d topics" % lda.num_topics

    inspect_lda_model(lda, topic=opts.topic, prior_reduce=opts.prior_reduce, topics=opts.topics)