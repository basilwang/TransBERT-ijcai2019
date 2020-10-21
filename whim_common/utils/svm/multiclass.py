"""
Tools relating to multiclass classification using LibSVM.

"""
import numpy


def count_class_votes(decision_function, num_classes):
    """
    The decision function from multiclass SVM gives us scores for every pairwise classification between
    possible classes.
    Count up the votes for each class from this pairwise classification. This is exactly what LibSVM does,
    choosing the class with most votes. Since we want to rank all the classes, we count up all the
    votes.

    """
    # This is modelled closely on LibSVM's code to make sure we do the same thing
    votes = numpy.zeros(num_classes)
    p = 0
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            if decision_function[p] > 0:
                votes[i] += 1
            else:
                votes[j] += 1
            p += 1
    return votes