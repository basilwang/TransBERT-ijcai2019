import argparse
import os
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Like using grep -f, but works better with large pattern files. "
                                                 "Matches lines that begin with one of the given "
                                                 "patterns, followed by a space. Patterns will only match at the "
                                                 "beginning of lines and cannot contain spaces. Lines will be "
                                                 "sent to different output files depending of what pattern file "
                                                 "they matched")
    parser.add_argument("output_dir", help="Directory to split output to")
    parser.add_argument("-u", "--unmatched", dest="unmatched",
                        help="Name to use for the file to which all unmatched output is sent",
                        default="unmatched.txt")
    parser.add_argument("pattern_files", nargs="+", help="File(s) to read patterns from")
    opts = parser.parse_args()

    output_dir = os.path.abspath(opts.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First read in all the patterns
    patterns = []
    for pattern_filename in opts.pattern_files:
        with open(pattern_filename, 'r') as pattern_file:
            # Open a file to send matches to
            output_file = open(os.path.join(output_dir, os.path.basename(pattern_filename)), 'w')
            patterns.append((output_file, pattern_file.read().splitlines()))
    # Open a file for all unmatched output
    unmatched_file = open(os.path.join(output_dir, opts.unmatched), 'w')

    try:
        for line in sys.stdin:
            first_word, __, __ = line.partition(" ")
            # Try all the pattern sets
            for output_file, file_patterns in patterns:
                if first_word in file_patterns:
                    # Line matched: output to the corresponding file
                    output_file.write(line)
                    break
            else:
                # Line matched no pattern set
                unmatched_file.write(line)
    finally:
        for f, __ in patterns:
            f.close()
        unmatched_file.close()