import argparse
import os
import sys

import numpy
from scipy.stats.stats import chisquare

from whim.entity_narrative import read_predictions_dir


#def chi_square(sys0, sys1):
    ## Compute expected counts under independence
    #exps = numpy.outer(contingency.sum(axis=1), contingency.sum(axis=0)).astype(numpy.float64) / contingency.sum()
    ## Compute the chi^2 statistic
    #chis = ((contingency - exps) ** 2. / exps).sum()
    #return 1. - chi2.cdf(chis, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute significance between two accuracies")
    parser.add_argument("prediction_base_dir", help="Base directory to read predictions from. Results are "
                                                    "assumed to be in model-type/model-name subdirs")
    parser.add_argument("model_type0", help="Model type of predictions")
    parser.add_argument("model_name0", help="Model name of predictions")
    parser.add_argument("model_type1", help="Model type of predictions")
    parser.add_argument("model_name1", help="Model name of predictions")
    parser.add_argument("model_type_name_pairs", nargs="*", help="The rest of the args are interpreted as further "
                                                                 "model-type/model-name pairs")
    opts = parser.parse_args()

    # Piece together model types and names
    model_types = [opts.model_type0, opts.model_type1] + opts.model_type_name_pairs[::2]
    model_names = [opts.model_name0, opts.model_name1] + opts.model_type_name_pairs[1::2]
    prediction_dirs = [os.path.join(opts.prediction_base_dir, mtype, mname) for mtype, mname in
                       zip(model_types, model_names)]

    # Read in all the prediction files
    print >>sys.stderr, "Reading files..."
    prediction_sets = []
    results = []
    for system_num, prediction_dir in enumerate(prediction_dirs):
        predictions = read_predictions_dir(prediction_dir)
        print("Read %d predictions for system %d" % (len(predictions), system_num))

        results.append(numpy.bincount([1 if prediction.question.target_choice == prediction.choice else 0
                                       for prediction in predictions], minlength=2).astype(numpy.float64))
    print

    # Compute significance for all pairwise comparisons
    significant_comparisons = []
    for systemA_num, resultsA in enumerate(results[:-1]):
        systemA = "%s/%s" % (model_types[systemA_num], model_names[systemA_num])
        for systemB_num, resultsB in enumerate(results[systemA_num+1:], start=systemA_num+1):
            systemB = "%s/%s" % (model_types[systemB_num], model_names[systemB_num])
            print("Comparing %s to system %s" % (systemA, systemB))
            # Output accuracy scores
            accuracyA = 100. * resultsA[1] / resultsA.sum()
            accuracyB = 100. * resultsB[1] / resultsB.sum()
            print("  %s: %.2f%%" % (systemA, accuracyA))
            print("  %s: %.2f%%" % (systemB, accuracyB))
            # Compute significance with chi-squared test
            chi2_stat, p = chisquare(resultsA, resultsB)
            print("  p=%g %s" % (p, "***" if p < 0.01 else "**" if p < 0.05 else ""))

            if p < 0.05:
                significant_comparisons.append({
                    "A": systemA,
                    "B": systemB,
                    "p": p,
                    "A higher": (accuracyA > accuracyB),
                })

    if significant_comparisons:
        print("\nSignificant results:")
        for comp_info in significant_comparisons:
            higher_system = comp_info["A"] if comp_info["A higher"] else comp_info["B"]
            lower_system = comp_info["B"] if comp_info["A higher"] else comp_info["A"]
            print("  %s > %s,  p=%g" % (higher_system, lower_system, comp_info["p"]))