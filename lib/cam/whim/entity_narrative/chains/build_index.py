import argparse

from cam.whim.entity_narrative.chains.doc_index import RichDocumentVerbIndex, RichDocumentVerbChainIndex
from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents by the verb lemmas in events")
    parser.add_argument("doc_dir", help="Dir containing all rich documents, as output by build_docs")
    parser.add_argument("--tarred", action="store_true", help="The input corpus is tarred")
    parser.add_argument("--limit", type=int,
                        help="Only index the first N documents. Meant for testing only")
    parser.add_argument("--chains", action="store_true", help="Build an index of only those events that feature in "
                                                              "event chains")
    parser.add_argument("--min-chain-length", type=int, default=1,
                        help="Minimum length of chain to count when building chain index")
    opts = parser.parse_args()

    # Prepare a corpus for the input documents
    print "Loading corpus"
    corpus = RichEventDocumentCorpus(opts.doc_dir, tarred=opts.tarred, index_tars=False)

    if opts.chains:
        RichDocumentVerbChainIndex.build_for_corpus(corpus, opts.min_chain_length, progress=True, limit=opts.limit)
    else:
        RichDocumentVerbIndex.build_for_corpus(corpus, progress=True, limit=opts.limit)