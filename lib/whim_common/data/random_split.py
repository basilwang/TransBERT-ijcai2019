import argparse
import random

parser = argparse.ArgumentParser(description="Read in a file and split its lines three ways, randomly: 80% goes "
                                             "to the training file, 10% to dev and 10% to test")
parser.add_argument("in_file", help="Input file")
parser.add_argument("out_prefix", help="Prefix to use for output files. '_training.txt', '_test.txt' and '_dev.txt' "
                                       "are appended to this to get the out filenames")

opts = parser.parse_args()

with open(opts.in_file, 'r') as input_file:
    output_prefix = opts.out_prefix
    with open("%s_training.txt" % output_prefix, 'w') as training_output:
        with open("%s_test.txt" % output_prefix, 'w') as test_output:
            with open("%s_dev.txt" % output_prefix, 'w') as dev_output:
                for line in input_file:
                    # Make a random choice of where to send it
                    r = random.random()
                    if r < 0.1:
                        # This makes the test set
                        test_output.write(line)
                    elif r < 0.2:
                        # This one's for the dev set
                        dev_output.write(line)
                    else:
                        # Just the training set
                        training_output.write(line)
