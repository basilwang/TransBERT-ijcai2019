import argparse
from StringIO import StringIO
import random
import tarfile
import os
import shutil

import numpy
import matplotlib

from whim_common.utils.progress import get_progress_bar


matplotlib.use('PDF', warn=False)
import matplotlib.pyplot as plt

from cam.whim.entity_narrative.chains.doc_index import RichDocumentVerbIndex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read in a corpus of rich event documents and filter out rare events, "
                                                 "outputting to another directory")
    subparsers = parser.add_subparsers(dest="command")
    stats_parser = subparsers.add_parser("stats",
                                         help="Don't do anything: just output some graphs of the verb distribution to "
                                              "help with choosing the threshold")
    stats_parser.add_argument("verbs_index", help="Verb lemma index for the input corpus")
    filter_parser = subparsers.add_parser("filter", help="Actually perform filtering")
    filter_parser.add_argument("verbs_index", help="Verb lemma index for the input corpus")
    filter_parser.add_argument("threshold", type=int, help="Minimum number of times a predicate (verb lemma) must have "
                                                           "been seen for events that use it to be included")
    filter_parser.add_argument("output_dir", help="Directory to output document files to")
    opts = parser.parse_args()

    print "Loading index..."
    index = RichDocumentVerbIndex.load(opts.verbs_index)
    print "%d verbs in corpus" % len(index.verb_index)

    # From the index, we have easy access to the number of occurrences of each verb
    # Get all the counts and plot the distribution
    verb_counts = [(v, len(index.verb_index[v])) for v in index.verb_index]

    if opts.command == "stats":
        print "Verb count distribution stats"
        print "============================="
        count_array = numpy.array([c for (v, c) in verb_counts])
        count_cum = numpy.cumsum(numpy.bincount(count_array), dtype=numpy.float64)
        count_cum /= count_cum[-1]
        count_cum *= 100.
        for max_count in [1, 5, 10, 15, 20, 25, 30]:
            print "%.4g%% of verb types have counts <= %d" % (count_cum[max_count], max_count)

        print
        print "First percentile -> count threshold of %d" % int(numpy.argmax(count_cum >= 1.))
        print "10th percentile -> count threshold of %d" % int(numpy.argmax(count_cum >= 10.))
        print "20th percentile -> count threshold of %d" % int(numpy.argmax(count_cum >= 20.))

        def _sample(start, stop):
            print "\nSample of verbs with %s <= count < %s:" % (start, stop)
            print "  %s" % "\n  ".join(random.sample([v for (v, c) in verb_counts if start <= c < stop], 10))

        _sample(5, 10)
        _sample(15, 20)
        _sample(25, 30)
        _sample(45, 50)
        _sample(55, 60)
        _sample(95, 100)

        graph_filename = "verbs-0-30.pdf"
        print "Outputting verbs with 0-30 counts to %s" % graph_filename
        plt.figure()
        plt.hist([c for (v, c) in verb_counts if c < 30])
        plt.savefig(graph_filename)

        graph_filename = "verbs-0-100.pdf"
        print "Outputting verbs with 0-100 counts to %s" % graph_filename
        plt.figure()
        plt.hist([c for (v, c) in verb_counts if c < 100], bins=100)
        plt.savefig(graph_filename)

        graph_filename = "verbs-100-500.pdf"
        print "Outputting verbs with 100-500 counts to %s" % graph_filename
        plt.figure()
        plt.hist([c for (v, c) in verb_counts if 100 <= c < 500], bins=150)
        plt.savefig(graph_filename)

        graph_filename = "verbs-200-.pdf"
        print "Outputting verbs with 200 or more counts to %s" % graph_filename
        plt.figure()
        plt.hist([c for (v, c) in verb_counts if c >= 200], bins=1000)
        plt.savefig(graph_filename)
    else:
        output_dir = opts.output_dir
        # Clear any existing output and make sure the output dir exists
        if os.path.exists(output_dir):
            print "Clearing up old output..."
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        included_verbs = [v for (v, c) in verb_counts if c >= opts.threshold]
        print "Filter will remove %d verb types, leaving %d" % (len(verb_counts)-len(included_verbs),
                                                                len(included_verbs))
        # Output a list of the verbs we're keeping
        print "Outputting verb list to %s" % os.path.join(output_dir, "predicates.meta")
        with open(os.path.join(output_dir, "predicates.meta"), 'w') as predicates_file:
            predicates_file.write("\n".join(included_verbs))

        print "Counting docs"
        num_docs = len(index.corpus)
        print "Processing %d documents" % num_docs
        pbar = get_progress_bar(num_docs, title="Filtering", counter=True)
        # Do the filtering: read in each document
        current_archive = None
        archives = []
        try:
            for i, (archive_name, filename, doc) in enumerate(index.corpus.archive_iter()):
                pbar.update(i)

                if archive_name != current_archive:
                    # We've moved onto a new archive: create a new archive in the output
                    current_archive = archive_name
                    new_archive = tarfile.open(os.path.join(output_dir, archive_name), 'w')
                    # Add it to the list of archives that will get closed
                    archives.append(new_archive)

                # Filter the document's events
                doc.events = [e for e in doc.events if e.verb_lemma in included_verbs]
                data = doc.to_text()

                # Write out the modified doc
                if current_archive is not None:
                    # We're writing out to tar archives
                    tarinfo = tarfile.TarInfo(filename)
                    tarinfo.size = len(data)
                    archives[-1].addfile(tarinfo, StringIO(data))
                else:
                    # Just write out as text
                    with open(os.path.join(output_dir, filename), 'w') as out_doc_file:
                        out_doc_file.write(data)
        finally:
            for archive in archives:
                archive.close()

        pbar.finish()
