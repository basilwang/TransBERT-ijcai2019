import random
import numpy
import numpy.random
import math


def slicesample(initial, nsamples, pdf, logpdf=False, width=10, burnin=0, thin=1, max_iters=200):
    """
    Port of Matlab's slicesample function. Ported very directly from the Matlab implementation.

    :param initial: numpy array or float
    :param nsamples: number of samples to draw
    :param pdf: function to serve as the probability density function
    :param logpdf: if True, output of pdf is assumed to be log probs
    :param width: how wide a slice to explore
    :param burnin: number of burn-in samples
    :param thin: thinning, no idea...
    :param max_iters: maximum number of iterations at various stages of the sampling process
    :return:
    """
    # log density is used for numercial stability
    if not logpdf:
        # Wrap the pdf in a log if it's not already logged
        pdf = lambda x: numpy.log(pdf(x))

    if pdf(initial) == float('-inf'):
        raise SliceSamplingError("initial=%s, not in the domain of the target distribution" % initial)

    # error checks for burnin and thin
    burnin = int(burnin)
    if burnin < 0:
        raise SliceSamplingError("Burn-in parameter must be a non-negative integer value.")
    thin = int(thin)
    if thin <= 0:
        raise SliceSamplingError("Thinning parameter must be a positive integer value")

    # dimension of the distribution
    if type(initial) is float or len(initial.shape) == 0:
        # Either a float or a numpy numeric value
        dim = 1
        initial = numpy.array([initial])
    else:
        dim = initial.shape[0]
    rnd = numpy.zeros((nsamples, dim), dtype=numpy.float64)  # placeholder for the random sample sequence
    neval = nsamples  # one function evaluation is needed for each slice of the density.

    # needed for the vertical position of the slice.
    e = numpy.random.exponential(size=nsamples*thin+burnin)

    RW = numpy.random.rand(nsamples*thin+burnin, dim)  # factors of randomizing the width
    RD = numpy.random.rand(nsamples*thin+burnin, dim)  # uniformly draw the point within the slice
    x0 = numpy.array(initial)  # current value

    # bool function indicating whether the point is inside the slice.
    def inside(x, th):
        return pdf(x) > th

    realmax = numpy.finfo(dtype=numpy.float64).max
    sqrt_max = numpy.sqrt(realmax)

    # update using stepping-out and shrinkage procedures.
    for i in range(-burnin, nsamples*thin):
        # A vertical level is drawn uniformly from (0,f(x0)) and used to define
        # the horizontal "slice".
        z = pdf(x0) - e[i+burnin]

        # An interval [xl, xr] of width w is randomly position around x0 and then
        # expanded in steps of size w until both size are outside the slice.
        # The choice of w is usually tricky.
        r = width * RW[i+burnin, :]  # random width/stepsize
        xl = x0 - r
        xr = xl + width
        iter = 0

        # step out procedure is performed only when univariate samples are drawn.
        if dim == 1:
            # step out to the left.
            while inside(xl,z) and iter< max_iters:
                xl = xl - width
                iter += 1
            if iter >= max_iters or numpy.any(xl < -sqrt_max):
                # It takes too many iterations to step out.
                raise SliceSamplingError("The step-out procedure failed")
            neval = neval + iter
            # step out to the right
            iter = 0
            while inside(xr,z)and iter < max_iters:
                xr = xr + width
                iter += 1
            if iter >= max_iters or numpy.any(xr > sqrt_max):
                # It takes too many iterations to step out
                raise SliceSamplingError("The step-out procedure failed")
            neval = neval + iter

        # A new point is found by picking uniformly from the interval [xl, xr].
        xp = RD[i+burnin, :] * (xr-xl) + xl

        # shrink the interval (or hyper-rectangle) if a point outside the
        # density is drawn.
        iter = 0
        while not inside(xp, z) and iter < max_iters:
            rshrink = (xp > x0)
            xr[rshrink] = xp[rshrink]
            xl[~rshrink] = xp[~rshrink]
            xp = numpy.random.rand(dim) * (xr-xl) + xl # draw again
            iter += 1
        if iter >= max_iters:
            # It takes too many iterations to shrink in.
            raise SliceSamplingError("The shrink-in procedure failed")

        x0 = xp   # update the current value
        if i >= 0 and (i + 1) % thin == 0:
            # burnin and thin
            rnd[i/thin-1, :] = x0
        neval = neval + iter

    # averaged number of evaluations
    neval = neval/(nsamples*thin+burnin)
    return rnd, neval


class SliceSamplingError(Exception):
    pass


def sample_normalized(log_probs, return_prob=False, non_log=False, exclude=[]):
    """
    Samples from normalized multinomial using numpy array of log probs.
    If non_log=True, input probs are non treated as log probs

    :param log_probs:
    :return: the bin in which the random float was found
    """
    # Exp all the probs and normalize
    if non_log:
        probs = log_probs
    else:
        probs = numpy.exp(log_probs - max(log_probs))
    probs /= probs.sum()

    state = None
    while state is None or state in exclude:
        state = numpy.random.choice(len(probs), p=probs)

    if return_prob:
        return state, math.log(probs[state])
    else:
        return state


def random_subsample(iterable, prop, batch_draw=1000):
    # We don't know how many samples we'll need, but draw a load at once for efficiency
    random_choices = iter(numpy.random.random(size=batch_draw) < prop)
    for val in iterable:
        try:
            choice = random_choices.next()
        except StopIteration:
            # We ran out of random draws: generate some more
            random_choices = iter(numpy.random.random(size=batch_draw) < prop)
            choice = random_choices.next()

        # Decide whether to include this value
        if bool(choice):
            yield val


def balanced_array_sample(array, balance_ratio=1.0, min_inclusion=0):
    """
    Subsample the rows of the given 1D array such that the result is roughly balanced on the values. The sample
    will include all rows for the least frequent label, but will undersample the rest to roughly match.

    By default, the non-minimal classes are sampled such that they're expected to include the same number
    of instances as the minimal class. Giving balance_ratio changes this so that, where a class has at least
    min_freq * balance_ratio instances, the expected number of inclusions is that (and otherwise full inclusion).

    If min_inclusion is given, the result is checked to ensure at least that many of every class are included.
    If not, more are sampled until either the min is reached or there aren't any more rows with that label
    left.

    Returns a list of row indices to include. You can get the sample (view) by doing array[result].

    """
    # Count the frequencies of each of the labels
    label_freqs = numpy.bincount(array)
    nonzero_label_freqs = label_freqs[label_freqs.nonzero()]
    # Take as our base the least frequent observed label -- this will be included in full
    min_freq = float(numpy.min(nonzero_label_freqs))
    # Make an array of probabilities with which we sample whether to include each instance, according to label
    label_probs = numpy.zeros(label_freqs.shape[0], dtype=numpy.float64)
    # An unobserved labels are given zero prob (which is never used, since they're not in the data we're sampling)
    label_probs[label_freqs.nonzero()] = min_freq / nonzero_label_freqs
    # Scale up the probs by the balance ratio to scale up the expected number of included instances
    # Where a class doesn't have the resulting number of instances (i.e. scaled prob > 1.), just include all
    label_probs = numpy.minimum(label_probs*balance_ratio, 1.0)

    # Draw a random float sample for every row
    samples = numpy.random.ranf(size=array.shape[0])
    # Pick all the samples that are under the prob for that row's target
    include_indices = numpy.where(samples <= label_probs[array])[0].tolist()

    if min_inclusion > 0:
        # Check we've got enough of each class
        sampled_freqs = numpy.bincount(array[include_indices], minlength=label_freqs.shape[0])
        include_set = set(include_indices)
        # Only include labels where there are some more that we've not yet sampled
        for label in numpy.where(numpy.logical_and(sampled_freqs < min_inclusion, label_freqs > sampled_freqs))[0]:
            # Find out what we've got left to choose from
            unused = list(set(numpy.where(array == label)[0]) - include_set)
            num_needed = min_inclusion - sampled_freqs[label]
            if len(unused) < num_needed:
                # There aren't enough left to reach to minimum: include all remaining examples
                include_indices.extend(unused)
            else:
                # Sample (without replacement) as many as we need from those we haven't already taken
                random.shuffle(unused)
                include_indices.extend(unused[:num_needed])

    return include_indices
