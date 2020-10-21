
def hold_one_out(in_set):
    """
    Return a list of all lists made by holding out one of the given input set. Each is
    a tuple of the item held out and the remaining list.

    """
    return [(in_set[hold], in_set[:hold]+in_set[hold+1:]) for hold in range(len(in_set))]


def hold_one_out_forwards(in_set, start=0):
    """
    Like hold_one_out(), but creates pairs (prior, event), where prior is all the events that
    happen *before* event in the sequence.

    If start > 0, splits are skipped where len(prior) < start.

    """
    return [(in_set[hold], in_set[:hold]) for hold in range(start, len(in_set))]