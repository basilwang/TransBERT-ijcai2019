"""
Very simple utilities go in here. You can import then from the containing package.

"""
from itertools import islice
from operator import itemgetter
from random import random
import numpy


class Accumulator(object):
    def __init__(self, initial=0):
        self.value = initial

    def add(self, val):
        self.value += val

    def increment(self):
        self.add(1)

    def __str__(self):
        return "<%d>" % self.value


def comma_int(val):
    return "{:,}".format(val)


def k_thousands(val):
    if val > 0 and val % 1000 == 0:
        return "%dk" % (val / 1000)
    else:
        return str(val)


def median(vals):
    sorted_vals = list(sorted(vals))
    if len(vals) == 0:
        return 0.0
    elif len(vals) % 2 == 0:
        return float(sorted_vals[len(vals)/2-1] + sorted_vals[len(vals)/2]) / 2.0
    else:
        return float(sorted_vals[len(vals)/2])


def group_pairs(lst, none_initial=False):
    previous = None
    for v in lst:
        if previous is not None or none_initial:
            yield (previous, v)
        previous = v


def prepend_lines(text, prepend="  "):
    """
    Prepend the given string to the start of each line of the text.

    """
    return "\n".join("%s%s" % (prepend, line) for line in text.splitlines())


def indent(text, spaces=2):
    return prepend_lines(text, prepend=" "*spaces)


def filled_truncate(lst, size, from_end=False, fill=None):
    """
    Truncate lst to the given size (from the start, or the end if from_end=True) and, if it's not as long
    as the required size, fill it with Nones (or fill).

    """
    if len(lst) < size:
        if from_end:
            return [fill] * (size - len(lst)) + lst
        else:
            return lst + [fill] * (size - len(lst))
    else:
        if from_end:
            return lst[-size:]
        else:
            return lst[:size]


def window(seq, window_size):
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    """
    it = iter(seq)
    result = list(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + [elem]
        yield result


def tuple_window(seq, window_size):
    for result in window(seq, window_size):
        yield tuple(result)


def lookahead(seq, n=1):
    """
    Iterates over a sequence, providing a sneaky lookahead to the upcoming n items in the sequence.
    Similar to window().

    """
    it = iter(seq)
    # Load the first n+1 items into the buffer
    buffer = list(islice(it, n+1))
    # If we reached the end already, start filling out with Nones
    buffer_filled = len(buffer)
    if buffer_filled < n+1:
        buffer.extend([None] * (buffer_filled-n+1))
    yield buffer
    # Add each of the remaining items in the seq one at a time
    for elem in it:
        buffer.pop(0)
        buffer.append(elem)
        yield buffer
    # Keep going, filling with Nones until there are no more elements left
    while buffer_filled > 1:
        buffer.pop(0)
        buffer.append(None)
        yield buffer
        buffer_filled -= 1


def file_lines(filename):
    with open(filename, 'r') as f:
        return sum(1 for x in f)


def first_over(it, threshold, initializer=0):
    """
    Return the index of the first item in iterator it for which the sum of the values seen so
    far (including this item) is greater than threshold.

    Returns a tuple containing the index and the sum up to and including that index
    Returns (None, None) if the sum never reaches the threshold.

    """
    total = initializer
    for i, points in enumerate(it):
        total += points
        if total > threshold:
            return i, total
    return None, None


def list_structure(lst):
    """
    For debugging: take a potentially nest list (or tuple) and return a string that doesn't output the
    full content of the list, but just its structure.

    """
    if type(lst) is list or type(lst) is tuple:
        # Report the contents of the list
        contents = []
        non_lists = 0
        for item in lst:
            if type(item) is list or type(item) is tuple:
                if non_lists:
                    contents.append("..%d.." % non_lists)
                    non_lists = 0
                # Recurse for this one
                contents.append(list_structure(item))
            else:
                # Just report how many non-lists there were
                non_lists += 1
        if non_lists:
            contents.append("..%d.." % non_lists)
        return "%s(%s)" % (type(lst).__name__, ", ".join(contents))
    else:
        return type(lst).__name__


def random_split(lst, proportion=0.5):
    """
    Split the items in the list (or iterable with a len) into two lists. Optionally, split such that the first list
    has (roughly) the given proportion. Implemented so that it can be run on very large lists.

    """
    # Generate booleans to select which list to put the items in
    selectors = [random() < proportion for i in range(len(lst))]
    # Select the items from the list
    left_list = [item for (selector, item) in zip(selectors, lst) if selector]
    right_list = [item for (selector, item) in zip(selectors, lst) if not selector]
    return left_list, right_list


def hold_one_out(lst):
    for i in range(len(lst)):
        yield lst[i], lst[:i] + lst[i+1:]


def rising_factorial(x, n):
    """
    Rising factorial, or Pochhammer symbol.

    prod_{y=0..n-1} (x + y)
      = (x+n-1)! / (x-1)!

    :param x: base of factorial, may be an int, float or even numpy array
    :param n: number to rise
    """
    if n == 0:
        return numpy.ones_like(x)
    return numpy.prod([x + y for y in range(n)], axis=0)


def log_rising_factorial(x, n):
    """
    Rising factorial, or Pochhammer symbol. Returns the log of the result, which can avoid overflow
    for large n.

    prod_{y=0..n-1} (x + y)
      = (x+n-1)! / (x-1)!

    :param x: base of factorial, may be an int, float or even numpy array
    :param n: number to rise
    """
    if n == 0:
        return numpy.ones_like(x)
    return numpy.sum(numpy.log(numpy.array([x + y for y in range(n)])), axis=0)


def str_to_bool(string):
    if string.lower() in ["0", "f", "false", "n", "no"]:
        return False
    else:
        return bool(string)


def choose_from_list(options, name=None):
    """
    Utility for option processors to limit the valid values to a list of possibilities.
    """
    name_text = " for option %s" % name if name is not None else ""

    def _fn(string):
        if string not in options:
            raise ValueError("%s is not a valid value%s. Valid choices: %s" % (string, name_text, ", ".join(options)))
        else:
            return string
    return _fn


def filter_subsample(iterator, sample_indices):
    if sample_indices is None:
        for v in iterator:
            yield v
    else:
        sample_indices = iter(sample_indices)
        next_index = sample_indices.next()

        for i, value in enumerate(iterator):
            if i == next_index:
                # Include this value
                try:
                    next_index = sample_indices.next()
                except StopIteration:
                    # No more indices: fine, we'll skip the rest
                    pass
                yield value


def batch(seq, size):
    # Special case if size == 0: just yield the whole lot in one big batch
    if size == 0 or size is None:
        yield list(seq)
    else:
        it = iter(seq)
        values = []
        try:
            while True:
                values = []
                for n in xrange(size):
                    values.append(it.next())
                yield values
        except StopIteration:
            # Reached the end of the sequence: yield any accumulated values not yet yielded
            if len(values):
                yield values


def bincount(iterable, minlength=None):
    """
    Like Numpy's bincount, but takes an arbitrary iterable of ints as input.

    """
    if minlength is not None:
        bin_counts = [0] * minlength
    else:
        # Fill out the list as we go
        bin_counts = []

    for value in iterable:
        try:
            bin_counts[value] += 1
        except IndexError:
            # Not got a count for this value yet
            while len(bin_counts) <= value:
                bin_counts.append(0)
            bin_counts[value] = 1

    return numpy.array(bin_counts)


def count_dict_to_bin_count(dic, minlength=None):
    """
    Take a dictionary of (int ID -> count) and produce an array in the style of bincount().

    """
    if minlength is not None:
        bin_counts = [0] * minlength
    else:
        # Fill out the list as we go
        bin_counts = []

    for idx, count in dic.items():
        try:
            bin_counts[idx] = count
        except IndexError:
            # Need to extend the list of counts to fit this one in
            while len(bin_counts) <= idx:
                bin_counts.append(0)
            bin_counts[idx] = count

    return numpy.array(bin_counts)


def find_duplicates(lst, key=lambda x:x):
    """
    Find duplicate items in a list. Optionally, takes a key function to use for comparison.
    Items found more than twice will be included multiple times in the resulting list.

    """
    seen = set()
    seen_add = seen.add
    # Adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = [x for x in lst if key(x) in seen or seen_add(key(x))]
    return seen_twice


def remove_duplicates(lst, key=lambda x: x):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if key(x) not in seen and not seen_add(key(x))]


def enum_attr(lst, attr_name="num", start=0):
    for i, obj in enumerate(lst, start=start):
        setattr(obj, attr_name, i)
    return lst


def top_n(it, n, score_fn=lambda x: x):
    """
    Iterates over an iterable, keeping a list of the top N items (by some score available in the
    iterable), without having to load everything into memory at once.

    """
    top = []
    score_threshold = 0.

    for row in it:
        score = score_fn(row)
        if score > score_threshold or len(top) < n:
            # Add this to the highest scores we've got
            top.append((row, score))
            # Remove a relation if it falls out of the top n now
            top.sort(key=itemgetter(1), reverse=True)
            top = top[:n]
            # Update the score threshold for getting into the top N
            score_threshold = top[-1][1]

    # Get rid of the scores
    top = [row for (row, score) in top]
    return top
