import argparse
import os
import shutil

from cam.whim.entity_narrative.chains.document import RichEventDocument, RichEventDocumentCorpus
from whim_common.utils.gensim.data import MultiFileTextCorpus
from whim_common.utils.progress import get_progress_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a rich representation of the events in a document")
    parser.add_argument("texts", help="Look for the original documents in the given directory")
    parser.add_argument("coref_dir", help="Look for the output from the coreference resolver in the given directory")
    parser.add_argument("deps_dir", help="Look for the dependency graphs in the given directory")
    parser.add_argument("pos_dir", help="Look for the POS tags in the given directory")
    parser.add_argument("output_dir", help="Directory to output document files to")
    parser.add_argument("--skip-done", action="store_true", help="Skip any files that already exist in the output dir")
    parser.add_argument("--tarred", action="store_true", help="The existing output is tarred: for use with "
                                                              "--skip-done. Any new files will not be added to tar "
                                                              "archives: you must do that manually")
    opts = parser.parse_args()

    output_dir = opts.output_dir
    # Clear any existing output and make sure the output dir exists
    if os.path.exists(output_dir) and not opts.skip_done:
        print "Clearing up old output"
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print "Looking up original texts from %s" % opts.texts
    text_corpus = MultiFileTextCorpus(opts.texts, build_dictionary=False)

    print "Looking up coref output in %s" % opts.coref_dir
    print "Looking up dependency graphs in %s" % opts.deps_dir
    print "Looking up POS tags in %s" % opts.pos_dir
    print "Outputting documents to %s" % output_dir

    if opts.skip_done:
        print "Reading existing corpus in %s" % output_dir
        rich_corpus = RichEventDocumentCorpus(output_dir, tarred=opts.tarred)
        skip_files = [corpus_filename for archive_name, corpus_filename in rich_corpus.list_archive_iter()]
        print "Skipping %d files already found in output corpus" % len(skip_files)
    else:
        skip_files = []

    total_docs = len(text_corpus) - len(skip_files)
    print "Processing %d documents" % total_docs
    if total_docs < 1:
        # If we've counted 0 or negative docs, go on anyway
        # The counting doesn't actually check the same docs exist and we want to be sure of this
        print "  Strange number of docs, but continuing anyway to check none have been missed"
        total_docs = 1
    pbar = get_progress_bar(total_docs, title="Extracting docs", counter=True)
    for doc_num, document in enumerate(RichEventDocument.build_documents(text_corpus,
                                                                         opts.coref_dir,
                                                                         opts.deps_dir,
                                                                         opts.pos_dir,
                                                                         skip_files=skip_files)):
        if type(document) is tuple:
            # There was an error in extraction
            doc_name, err = document
            output_text = doc_name
        else:
            doc_name = document.doc_name
            output_text = document.to_text()

        if doc_name.startswith("NYT_ENG"):
            # Some special behaviour for Gigaword, so that we don't get too many files in one directory
            year = doc_name[8:12]
            file_output_dir = os.path.join(output_dir, year)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)
                # ## End Gigaword hack ## #
        else:
            file_output_dir = output_dir

        with open(os.path.join(file_output_dir, "%s.txt" % doc_name), 'w') as output_file:
            output_file.write(output_text)

        if doc_num < total_docs:
            pbar.update(doc_num)
    pbar.finish()