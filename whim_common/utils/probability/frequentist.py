import numpy


def event_prior_from_dictionary(dictionary, laplace=0.0):
    """
    Compute the (frequentist) prior probability distribution over events on the
    basis of the document frequency counts stored in a dictionary.

    Returns a numpy array of (non-log) probabilities.

    """
    dist = numpy.zeros(len(dictionary), dtype=numpy.float64) + laplace
    # Fill the array with document frequencies
    for event, count in dictionary.dfs.items():
        dist[event] = count
    # Normalize
    dist /= dist.sum()
    return dist