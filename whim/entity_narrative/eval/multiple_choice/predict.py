import argparse
import os
import sys

import numpy


from whim.entity_narrative import RichEventDocumentCorpus
from whim.entity_narrative import MultipleChoiceQuestion
from whim_common.utils.files import nfs_rmtree
from whim_common.utils.logging import get_console_logger
from whim_common.utils.progress import slice_progress


def read_predictions(prediction_filenames):
    # Read in all the prediction files
    prediction_sets = []
    for filename in prediction_filenames:
        if not filename.endswith(".log"):
            try:
                with open(filename, 'r') as in_file:
                    data = in_file.read()
                    prediction_sets.extend(MultipleChoicePrediction.from_text(t) for t in data.split("\n########\n"))
            except IOError as err:
                print >>sys.stderr, "Error reading predictions file, %s: %s" % (filename, err)
                raise
    return prediction_sets


def read_predictions_dir(dirname):
    """
    Read in all prediction files from a dir. By default, removes any files with a .log extension.

    """
    filenames = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    # Skip log files
    filenames = [f for f in filenames if not f.endswith(".log") and os.path.basename(f).startswith("question")]
    return read_predictions(filenames)


def read_prediction_accuracies(dirname):
    """
    Like running read_predictions_dir() and measuring the accuracy of the predictions, but faster, as
    it doesn't involve loading all predictions, just the necessary bits.

    """
    filenames = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    # Skip log files
    filenames = [f for f in filenames if not f.endswith(".log") and os.path.basename(f).startswith("question")]

    total_questions = 0
    correct_answers = 0
    for filename in filenames:
        # Only need to read the first two lines of the file to work out whether this prediction was correct or not
        with open(filename, "r") as question_file:
            model_choice = int(question_file.readline().partition(": ")[2])
            correct_choice = int(question_file.readline().partition(": ")[2])

            total_questions += 1
            if model_choice == correct_choice:
                correct_answers += 1
    return correct_answers, total_questions


def multiple_choice_eval(model, corpus, output_dir,
                         min_context=8, truncate_context=None,
                         samples=1000, runs=1, alternatives=5, log=None,
                         unbalanced_sample_rate=None, stoplist=None):
    if log is None:
        log = get_console_logger("Multiple choice eval")

    accuracies = numpy.zeros(runs, dtype=numpy.float64)
    run_additional_features = None
    # Do the sample and test multiple times
    for run in range(runs):
        if runs > 1:
            log.info("#### Run %d / %d ####" % (run+1, runs))

        # Prepare the output directory
        run_output_dir = os.path.join(output_dir, "run%.2d" % run)
        log.info("Outputting to %s" % run_output_dir)
        if os.path.exists(run_output_dir):
            nfs_rmtree(run_output_dir)
        os.makedirs(run_output_dir)

        # Draw samples samples to evaluate on
        if isinstance(corpus, RichEventDocumentCorpus):
            if samples is None:
                log.info("Generating test samples for full corpus")
                questions = list(
                    MultipleChoiceQuestion.generate_random_unbalanced(corpus,
                                                                      min_context=min_context,
                                                                      truncate_context=truncate_context,
                                                                      choices=alternatives,
                                                                      subsample=unbalanced_sample_rate,
                                                                      stoplist=stoplist)
                )
            else:
                log.info("Generating %d test samples (unbalanced)" % samples)
                questions = slice_progress(
                    MultipleChoiceQuestion.generate_random_unbalanced(corpus,
                                                                      min_context=min_context,
                                                                      truncate_context=truncate_context,
                                                                      choices=alternatives,
                                                                      subsample=unbalanced_sample_rate,
                                                                      stoplist=stoplist),
                    samples, "Sampling"
                )
        else:
            log.info("Generating %d test samples (balanced on verb)" % samples)
            questions = slice_progress(
                MultipleChoiceQuestion.generate_random_balanced_on_verb(corpus,
                                                                        min_context=min_context,
                                                                        truncate_context=truncate_context,
                                                                        choices=alternatives),
                samples, "Sampling"
            )

        accuracies[run] = evaluate_on_questions(questions, model, run_output_dir, log=log, corpus_dir=corpus.directory)

    # Put the stats together from all runs
    mean_accuracy = float(accuracies.mean())
    std_accuracy = float(accuracies.std())
    print("\nMean: %.2f%%\nStd: %.2f%%" % (mean_accuracy, std_accuracy))


def evaluate_on_questions(questions, model, output_dir, log=None, corpus_dir="unknown"):
    if log is None:
        log = get_console_logger("Multiple choice eval")

    entities = [q.entity for q in questions]
    choice_lists = [q.choices for q in questions]
    contexts = [q.context_events for q in questions]

    log.info("Making predictions")
    # Score the alternatives for every test case
    predictions = model.score_choices_bulk(entities, contexts, choice_lists, progress=True)

    scores = numpy.array(predictions)
    selections = numpy.argmax(scores, axis=1)

    correct_answers = numpy.array([q.target_choice for q in questions])
    num_correct = int(numpy.sum(selections == correct_answers))
    samples = len(questions)
    print("%d / %d correct" % (num_correct, samples))
    accuracy = float(num_correct) / samples * 100.

    # Output a report of how it went
    print("Outputting results to %s" % output_dir)
    with open(os.path.join(output_dir, "report.txt"), 'w') as report_file:
        report_file.write("""\
Model
=====
%s

Read test instances from %s

%d / %d correct choices
Accuracy: %.3f
""" % (
            model.description,
            corpus_dir,
            num_correct, samples,
            float(num_correct) / samples
        ))

    for q_num, (question, q_scores) in enumerate(zip(questions, scores)):
        prediction = MultipleChoicePrediction(question, selections[q_num], q_scores)
        with open(os.path.join(output_dir, "question-%.3d.txt" % q_num), 'w') as question_file:
            question_file.write(prediction.to_text())
    return accuracy


class MultipleChoicePrediction(object):
    def __init__(self, question, choice, scores):
        self.scores = scores
        self.choice = choice
        self.question = question

    def to_text(self):
        return """\
Model's choice: %d
Correct choice: %d
  (%s)

Model's scores: %s

%s""" % (
            self.choice,
            self.question.target_choice,
            "correct" if self.correct else "incorrect",
            ", ".join(["%f" % float(score) for score in self.scores]),
            self.question.to_text()
        )

    @property
    def correct(self):
        return self.choice == self.question.target_choice

    @staticmethod
    def from_text(text):
        try:
            lines = text.splitlines()
            model_choice = int(lines[0].partition("Model's choice: ")[2])
            scores = [float(t) for t in lines[4].partition("Model's scores: ")[2].split(",")]
            question = MultipleChoiceQuestion.from_text("\n".join(lines[6:]))
        except Exception as err:
            raise IOError("could not parse prediction text: %s" % err)
        return MultipleChoicePrediction(question, model_choice, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An evaluation task where the model is required to pick the "
                                                 "next event in a chain from a choice of 5 (including one "
                                                 "observed). Balanced on target predicates.")
    # parser.add_argument("model_type", help="Model type")
    # parser.add_argument("model", help="Model name")
    parser.add_argument("corpus_index", help="Verb chain index of corpus to be used as test set. Note that this "
                                             "must be a chain index, not event index")
    parser.add_argument("output", help="Directory to output report to")
    parser.add_argument("--opts", help="Model options. Use 'help' to get a list")
    parser.add_argument("--context", type=int, default=8, help="Minimum context to use to make predictions")
    parser.add_argument("--max-context", type=int, default=8, help="Truncate context to use to make predictions")
    parser.add_argument("--runs", type=int, help="Run the evaluation multiple times (with different random samples) "
                                                 "and output all the results", default=1)
    parser.add_argument("--samples", type=int,
                        help="Number of samples to draw per run. If using --unbalanced, the default behaviour is "
                             "not to subsample at all, but to use the full dataset. Otherwise, default is 1000")
    parser.add_argument("--unbalanced", action="store_true",
                        help="Don't balance the samples on verb lemmas (default behaviour), just sample randomly. "
                             "In this case, the corpus_index should point to a corpus root, not an index")
    parser.add_argument("--unbalanced-subsample", type=float,
                        help="For unbalanced sampling, subsamples randomly at the given rate (0 < x <= 1). "
                             "If 1, all data points will be used until the number of samples is reached")
    parser.add_argument("--tarred", action="store_true", help="Corpus is tarred (applies only to --unbalanced)")
    parser.add_argument("--stoplist", help="Path to file containing predicate stoplist. Only applies to unbalanced")
    parser.add_argument("--prepared", action="store_true",
                        help="Read in a pre-prepared sample (from generate_questions) instead of reading a corpus "
                             "and drawing samples. All sampling parameters are ignored in this case")
    opts = parser.parse_args()

    if opts.output is not None:
        if os.path.exists(opts.output):
            nfs_rmtree(opts.output)
        os.makedirs(opts.output)



    log = get_console_logger("Multiple choice eval")

    # Read in pre-prepared questions
    log.info("Reading in %d questions from %s" % (len(os.listdir(opts.corpus_index)), opts.corpus_index))
    questions = []
    for filename in os.listdir(opts.corpus_index):
        with open(os.path.join(opts.corpus_index, filename), 'r') as question_file:
            questions.append(MultipleChoiceQuestion.from_text(question_file.read()))

    log.info("Evaluating model")


