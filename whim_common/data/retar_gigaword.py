"""
I foolishly tarred my Gigaword documents grouped by year, which results in too many documents per tar to be
easily manageable. Read through them and re-group them by month.

"""
import argparse
import os
import tarfile

from whim_common.data.compression import TarredCorpus
from whim_common.utils.progress import get_progress_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-group Gigaword documents that have been tarred by year so "
                                                 "that they're grouped by month")
    parser.add_argument("input_dir", help="Directory containing the tarred corpus")
    parser.add_argument("output_dir", help="Directory to output the re-tarred corpus to")
    opts = parser.parse_args()

    output_dir = opts.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    corpus = TarredCorpus(opts.input_dir, index=False)

    for tarball_path, tarball_filename in zip(corpus.tar_filenames, corpus.tarballs):
        print "Processing %s" % tarball_filename
        if not tarball_filename.startswith("nyt_"):
            print "Couldn't handle filename: %s. Skipping" % tarball_filename
            continue

        # Prepare an output tarball for each month
        year = tarball_filename[4:8]
        print "  Year: %s" % year
        month_tarballs = dict([
            (
                "%.2d" % month,
                tarfile.open(os.path.join(output_dir, "nyt_%s%.2d.tar" % (year, month)), 'w')
            ) for month in range(1, 13)
        ])
        # Open up the input tarball so we can iterate over its members
        tarball = tarfile.open(tarball_path, 'r')

        num_files = len(tarball.getnames())

        try:
            pbar = get_progress_bar(num_files, "Splitting tarball")
            for i, tarinfo in enumerate(tarball):
                pbar.update(i)
                # Extract the file to the tmp dir
                f = tarball.extractfile(tarinfo)
                filename = tarinfo.name
                if not filename.startswith("NYT_ENG_"):
                    print "Non-gigaword filename: %s. Skipping" % filename
                else:
                    # The two chars after the year are the month number
                    file_month = filename[12:14]
                    # We should have a tarball for the month
                    if file_month not in month_tarballs:
                        print "No month tarball for month '%s'" % file_month
                    else:
                        month_tarballs[file_month].addfile(tarinfo, f)
            pbar.finish()
        finally:
            print "Closing files"
            for month_tarball in month_tarballs.values():
                month_tarball.close()
            tarball.close()