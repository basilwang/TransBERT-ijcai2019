import argparse
# import cPickle as pickle
import pickle
from operator import itemgetter
import sys

import numpy


def load_stoplist(filename):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.read().splitlines()]
    return [predicate for predicate in lines if predicate]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a stoplist of the most common predicates in a corpus")
    subparsers = parser.add_subparsers(dest="command")
    stats_parser = subparsers.add_parser("stats",
                                         help="Output some information about the corpus to help us decide how "
                                              "long the stoplist should be")
    stats_parser.add_argument("predicate_counts", help="Predicate counts for the input corpus, output by "
                                                       "count_predicates")
    filter_parser = subparsers.add_parser("build", help="Actually produce the list")
    filter_parser.add_argument("predicate_counts", help="Predicate counts for the input corpus, output by "
                                                       "count_predicates")
    filter_parser.add_argument("length", type=int, help="Number of most common predicates to include")
    opts = parser.parse_args()

    print >>sys.stderr, "Loading counts..."
    with open(opts.predicate_counts, 'r') as infile:
        count_dict = pickle.load(infile)
    print>>sys.stderr,  "%d verbs in corpus" % len(count_dict)

    print >>sys.stderr, "Excluding be:subj, since it should never be in the stoplist"
    del count_dict["be:subj"]

    if opts.command == "stats":
        print >>sys.stderr, "\nVerb count distribution stats"
        print >>sys.stderr, "============================="
        ordered_verb_counts = list(reversed(sorted(count_dict.items(), key=itemgetter(1))))
        count_cum = numpy.cumsum([c for (v, c) in ordered_verb_counts])
        count_cum_perc = count_cum.astype(numpy.float64) / count_cum[-1] * 100.
        for list_size in [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]:
            print >>sys.stderr, "Top %d predicate types cover %.4f%% of event tokens" % \
                                (list_size, count_cum_perc[list_size-1])

        print >>sys.stderr
        counts_perc = numpy.array([c for (v, c) in ordered_verb_counts], dtype=numpy.float64) / count_cum[-1] * 100.
        for list_size in [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]:
            print >>sys.stderr, "Predicate %d covers %d event tokens (%.3f%%)" % \
                                (list_size-1, ordered_verb_counts[list_size-1][1], counts_perc[list_size-1])

        print >>sys.stderr
        for list_size in [1, 5, 10, 20, 30, 50, 100]:
            print >>sys.stderr, "Predicate %d: %s" % (list_size-1, ordered_verb_counts[list_size-1][0])

        print >>sys.stderr
        print >>sys.stderr, "Top 20 predicates:"
        print >>sys.stderr, "\n".join("%.2d: %s" % (i, v) for i, (v, c) in enumerate(ordered_verb_counts[:20]))
    else:
        ordered_verb_counts = list(reversed(sorted(count_dict.items(), key=itemgetter(1))))
        print >>sys.stderr, "\nOutputting stoplist, top %d predicates, to stdout" % opts.length
        print("\n".join(v for (v, c) in ordered_verb_counts[:opts.length]))
