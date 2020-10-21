import numpy


def vector_cosine_similarity(v0, v1):
    """
    Compute cosine similarity between vectors v0 and v1.

    """
    return numpy.dot(v0, v1) / magnitude(v0) / magnitude(v1)


def magnitude(v):
    return numpy.sqrt(numpy.sum(v ** 2., axis=-1))
