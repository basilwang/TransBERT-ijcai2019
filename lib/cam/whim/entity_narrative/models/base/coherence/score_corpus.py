import argparse
import sys
from whim_common.data.compression import detect_tarred_corpus
from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus
from cam.whim.entity_narrative.models.base.coherence import CoherenceScorer
from cam.whim.entity_narrative.models.base.model import NarrativeChainModel, cmd_line_model_options
from whim_common.utils.progress import get_progress_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterate over a narrative chain corpus, assigning a coherence score to each chain")
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument("model_type", help="Type of model to load (must be a type that implements CoherenceScorer)")
    model_grp.add_argument("model_name", help="Name of model to load")
    model_grp.add_argument("--opts", help="Model options (use 'help' for a list)")
    model_grp.add_argument("--batch", type=int, default=1000,
                           help="Batch size to use (for models that provide a batched implementation)")
    data_grp = parser.add_argument_group("Event chain data")
    data_grp.add_argument("corpus_dir", help="Directory to read in chains from for training data")
    output_grp = parser.add_argument_group("Output")
    output_grp.add_argument("output_file", help="File to output scores to")
    opts = parser.parse_args()

    # Load the model
    model_options = cmd_line_model_options(opts.model_type, opts.opts)
    model_cls = NarrativeChainModel.load_type(opts.model_type)
    # Check this is a suitable type of model
    if not issubclass(model_cls, CoherenceScorer):
        print >>sys.stderr, "Model type %s is not a coherence scorer (does not override CoherenceScorer)" % \
                            opts.model_type
        sys.exit(1)
    model = model_cls.load(opts.model_name, **(model_options or {}))

    # Load the corpus
    tarred = detect_tarred_corpus(opts.corpus_dir)
    corpus = RichEventDocumentCorpus(opts.corpus_dir, tarred=tarred)

    # Open output file
    pbar = get_progress_bar(len(corpus), title="Scoring chains", counter=True)
    next_doc_num_report = 0
    with open(opts.output_file, 'w') as output_file:
        for score, (archive_name, doc_num, doc_name, chain_num) in model.chain_coherences(
                (((entity, events), (archive_name, doc_num, doc.doc_name, chain_num))
                 for doc_num, (archive_name, filename, doc) in pbar(enumerate(corpus.archive_iter()))
                 for chain_num, (entity, events) in enumerate(doc.get_chains())), batch_size=opts.batch):
            if score is not None:
                print >>output_file, "%s, %s, %d, %g" % (archive_name, doc_name, chain_num, score)
    pbar.finish()
