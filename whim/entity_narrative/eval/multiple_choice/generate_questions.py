import argparse
import os
import sys

from whim.entity_narrative.chains.doc_index import RichDocumentVerbChainIndex
from whim.entity_narrative.chains.document import RichEventDocumentCorpus
from whim.entity_narrative.chains.stoplist import load_stoplist
from whim.entity_narrative.eval.multiple_choice import MultipleChoiceQuestion
from whim_common.utils.files import nfs_rmtree
from whim_common.utils.logging import get_console_logger
from whim_common.utils.progress import get_progress_bar


def generate_questions(corpus, output_dir,
                       min_context=4, truncate_context=None,
                       samples=1000, alternatives=5, log=None,
                       unbalanced_sample_rate=None, stoplist=None):
    if log is None:
        log = get_console_logger("Multiple choice questions")

    # Prepare the output directory
    log.info("Outputting to %s" % output_dir)
    if os.path.exists(output_dir):
        nfs_rmtree(output_dir)
    os.makedirs(output_dir)

    # Draw samples samples to evaluate on
    if isinstance(corpus, RichEventDocumentCorpus):
        log.info("Generating %d test samples (unbalanced)" % samples)
        if unbalanced_sample_rate is None:
            log.info("Not subsampling")
            if samples > len(corpus):
                log.warn("Trying to generate %d samples, but only %d docs in corpus" % (samples, len(corpus)))
        else:
            log.info("Subsampling at a rate of %.3g" % unbalanced_sample_rate)
            if samples > int(float(len(corpus)) * unbalanced_sample_rate * 0.9):
                log.warn("Trying to generate %d samples, but likely to run out by %d" %
                         float(len(corpus)) * unbalanced_sample_rate)

        questions = MultipleChoiceQuestion.generate_random_unbalanced(corpus,
                                                                      min_context=min_context,
                                                                      truncate_context=truncate_context,
                                                                      choices=alternatives,
                                                                      subsample=unbalanced_sample_rate,
                                                                      stoplist=stoplist)
    else:
        log.info("Generating %d test samples (balanced on verb)" % samples)
        questions = MultipleChoiceQuestion.generate_random_balanced_on_verb(corpus,
                                                                            min_context=min_context,
                                                                            truncate_context=truncate_context,
                                                                            choices=alternatives)

    pbar = get_progress_bar(samples, "Generating")
    filename_fmt = "question%%0%dd.txt" % len("%d" % (samples-1))
    q = 0
    for q, question in enumerate(questions):
        pbar.update(q)
        with open(os.path.join(output_dir, filename_fmt % q), 'w') as q_file:
            q_file.write(question.to_text())
        if q == samples-1:
            # Got enough samples: stop here
            break
    else:
        log.info("Question generation finished after %d samples" % q)
    pbar.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a sample for the evaluation task, so that it's repeatable and comparable across models, "
                    "whilst still being a random sample from the dataset")
    parser.add_argument("corpus_index", help="Verb chain index of corpus to be used as test set. Note that this "
                                             "must be a chain index, not event index")
    parser.add_argument("output", help="Directory to output report to")
    parser.add_argument("--context", type=int, default=4, help="Minimum context to use to make predictions")
    parser.add_argument("--max-context", type=int, default=8, help="Truncate context to use to make predictions")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to draw. Default is 1000")
    parser.add_argument("--unbalanced", action="store_true",
                        help="Don't balance the samples on verb lemmas (default behaviour), just sample randomly. "
                             "In this case, the corpus_index should point to a corpus root, not an index")
    parser.add_argument("--unbalanced-subsample", type=float,
                        help="For unbalanced sampling, subsamples randomly at the given rate (0 < x <= 1). "
                             "If 1, all data points will be used until the number of samples is reached")
    parser.add_argument("--tarred", action="store_true", help="Corpus is tarred (applies only to --unbalanced)")
    parser.add_argument("--stoplist", help="Path to file containing predicate stoplist. Only applies to unbalanced")
    opts = parser.parse_args()

    if opts.unbalanced:
        # Load a rich docs corpus, not an index
        index = RichEventDocumentCorpus(opts.corpus_index, tarred=opts.tarred, index_tars=True)
        print >>sys.stderr, "Counting corpus size"
        # This gets cached so do it now
        len(index)
    else:
        # Prepare dataset: must be a verb chains index
        index = RichDocumentVerbChainIndex.load(opts.corpus_index)

    if opts.stoplist:
        stoplist = load_stoplist(opts.stoplist)
        print >>sys.stderr, "Using stoplist in %s" % opts.stoplist
    else:
        stoplist = None

    generate_questions(index, opts.output,
                       min_context=opts.context,
                       truncate_context=opts.max_context,
                       samples=opts.samples,
                       unbalanced_sample_rate=opts.unbalanced_subsample,
                       stoplist=stoplist)