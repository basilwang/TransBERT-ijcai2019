import argparse
import os
import random
import tarfile
from whim_common.utils.progress import get_progress_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training, dev and test sets for the Altavilla corpus. "
                                                 "Note that this should only be run once and the output then used "
                                                 "for everything: don't rerun it every time you need to split the "
                                                 "data! Splits 80:10:10")
    parser.add_argument("altavilla_dir", help="Directory containing the Altavilla corpus")
    parser.add_argument("output_dir", help="Directory to output the chapter lists to")
    opts = parser.parse_args()

    chapters_dir = os.path.join(opts.altavilla_dir, "chapters")
    output_dir = os.path.abspath(opts.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all the chapters from each tarball
    chapters = []
    print "Reading chapter files"
    chapter_tarballs = sum([[os.path.join(chapters_dir, dirpath, filename) for filename in filenames]
                            for dirpath, dirnames, filenames in os.walk(chapters_dir)], [])
    pbar = get_progress_bar(len(chapter_tarballs))

    for i, chapter_filename in enumerate(chapter_tarballs):
        with tarfile.open(chapter_filename, 'r:gz') as tarball:
            # Get all the *.txt filenames
            chapters.extend([member.name for member in tarball.getmembers()
                             if member.isfile() and member.name.endswith(".txt")])

        pbar.update(i)
    pbar.finish()

    print "Read {} chapters".format(len(chapters))
    # Dev/test sets should be 10% of the data each
    subset_size = len(chapters) / 10

    # Choose a random sample to be held out as the dev/test sets
    # First shuffle the list, then take slices
    random.shuffle(chapters)

    test_set = chapters[:subset_size]
    dev_set = chapters[subset_size:2*subset_size]
    training_set = chapters[2*subset_size:]

    print "Writing file lists to {}".format(output_dir)
    with open(os.path.join(output_dir, "test_chapters.txt"), 'w') as test_file:
        test_file.write("\n".join(test_set))
    with open(os.path.join(output_dir, "dev_chapters.txt"), 'w') as dev_file:
        dev_file.write("\n".join(dev_set))
    with open(os.path.join(output_dir, "training_chapters.txt"), 'w') as training_file:
        training_file.write("\n".join(training_set))
