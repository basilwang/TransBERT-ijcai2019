import argparse
import cPickle as pickle
from collections import Counter
import os

from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus, predicate_relation
from whim_common.utils.progress import get_progress_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run over a corpus, counting verb-dependency pairs")
    parser.add_argument("doc_dir", help="Dir containing all rich documents, as output by build_docs")
    parser.add_argument("--tarred", action="store_true", help="The input corpus is tarred")
    opts = parser.parse_args()

    # Prepare a corpus for the input documents
    print "Loading corpus"
    corpus = RichEventDocumentCorpus(opts.doc_dir, tarred=opts.tarred, index_tars=False)

    num_docs = len(corpus)
    print "%d documents" % num_docs
    output_filename = os.path.join(opts.doc_dir, "predicate_counts")
    print "Outputting to %s" % output_filename
    pbar = get_progress_bar(num_docs, title="Counting")

    predicate_counter = Counter()
    for doc_num, doc in enumerate(corpus):
        for entity, events in doc.get_chains():
            predicate_counter.update([predicate_relation(entity, event) for event in events])
        pbar.update(doc_num)

    pbar.finish()

    # Convert to a dict of counts
    count_dict = dict(predicate_counter)

    with open(output_filename, 'w') as outfile:
        pickle.dump(count_dict, outfile)