import numpy


def get_important_words(lda, topic_nums, prior_reduce=0.75):
    """
    Use a mutual information-type statistic to pull out the most important words from
    a topic. The statistic itself is rather hacky, but it's only really meant for choosing
    words to display to represent a topic and seems to work quite nicely for that.

    Returns an array with weights for the words.

    """
    # Get the full set of topic distributions
    dists = lda.state.get_lambda()
    # Normalize each distribution
    dists = dists.T
    dists /= dists.sum(axis=0)
    dists = dists.T
    # Average over topics to get a kind of prior
    prior_dist = dists.sum(axis=0)

    dist = numpy.log(dists[topic_nums])
    prior_dist = numpy.log(prior_dist)

    # Divide each topic's probability by the word's prior prob to get a mutual information-type statistic
    # Scale down (root) the prior a bit - arbitrary scaling factor, but avoids ending up with only really rare words
    word_mi = dist - prior_reduce*prior_dist
    return word_mi


def format_topics(lda, topic_nums, top_n=10, prior_reduce=0.75):
    word_mi = get_important_words(lda, topic_nums, prior_reduce=prior_reduce)
    lines = []
    for i in range(word_mi.shape[0]):
        # Pull out the most important words
        bestn = numpy.argsort(word_mi[i])[::-1][:top_n]
        lines.append(", ".join(lda.id2word[id] for id in bestn))
    return lines