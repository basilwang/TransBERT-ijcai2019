import argparse
import os
import sys

import numpy as np

from cam.whim.entity_narrative.eval.multiple_choice.predict import read_predictions_dir


def evaluate(prediction_sets):
    # Compute accuracy
    total = len(prediction_sets)
    hits = sum(1 for cp in prediction_sets if cp.correct == cp.chosen)

    if total == 0:
        # Shouldn't really happen
        raise EvaluationMetricError("no results: cannot compute accuracy")

    return 100.0 * float(hits) / total


class EvaluationMetricError(Exception):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read output from multiple choice cloze evaluation and report evaluation statistics")
    parser.add_argument("prediction_dir", help="Directory to read predictions from")
    parser.add_argument("--human", help="Display metrics in a human-readable form", action="store_true")
    parser.add_argument("-o", "--output", help="Output a summary of the results to the given filename")
    opts = parser.parse_args()

    # Read in all the prediction files
    print >>sys.stderr, "Reading files..."
    # Check whether there are multiple runs in this directory
    if os.path.exists(os.path.join(opts.prediction_dir, "run00")):
        # Evaluate all the runs
        prediction_dirs = [os.path.join(opts.prediction_dir, dirname)
                           for dirname in os.listdir(opts.prediction_dir) if dirname.startswith("run")]
    else:
        # Just a single run in the top directory
        prediction_dirs = [opts.prediction_dir]
    prediction_dirs.sort()

    # Create the output file and output meta-data
    outfile = None
    if opts.output:
        outfile = open(opts.output, 'w')

    run_results = []

    try:
        if outfile:
            # Output header of results file
            outfile.write("Evaluation of predictions from %s\n" % opts.prediction_dir)
            outfile.write("Runs:\n  %s\n" % "\n  ".join(prediction_dirs))
            outfile.write("\n\nRun,accuracy\n")

        for prediction_dir in prediction_dirs:
            if outfile:
                outfile.write("%s" % prediction_dir)
            print >>sys.stderr, "\nEvaluating directory %s" % prediction_dir
            # Read in the predictions in this directory
            prediction_sets = read_predictions_dir(prediction_dir)

            print >>sys.stderr, "Read %d prediction lists" % len(prediction_sets)

            try:
                # Some the evaluation metric on this prediction set
                value = evaluate(prediction_sets)
            except EvaluationMetricError as e:
                print >>sys.stderr, "could not compute accuracy: %s" % e
                value = 0.0
            run_results.append(value)

            if opts.human:
                if value > 10000:
                    number = "%.2g" % value
                else:
                    number = "%.2f" % value
            else:
                number = "%.2f" % value

            print("Accuracy: %s%%" % number)

            if outfile:
                outfile.write(",%.2f\n" % value)

        if len(run_results) > 1:
            # Output average accuracy
            results = np.array(run_results)
            mean = float(np.mean(results))
            std_dev = float(np.std(results))
            print("\n\nMean = %.4g, std = %.4g" % (mean, std_dev))

            if outfile:
                outfile.write("mean,%.2f\n" % mean)
                outfile.write("std,%.2f\n" % std_dev)
    finally:
        if outfile:
            outfile.close()
            print("Output written to %s" % opts.output)