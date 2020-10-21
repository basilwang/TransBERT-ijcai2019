import argparse
import os
import tarfile

from whim_common.data.compression import TarredCorpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split up a tarred corpus into training, test and dev sets, according "
                                                 "to pre-prepared lists of files that should be in each set")
    parser.add_argument("input_dir", help="Directory containing the tarred corpus")
    parser.add_argument("dev_list", help="Text file with a filename on each line specifying the dev set")
    parser.add_argument("test_list", help="Text file with a filename on each line specifying the test set")
    parser.add_argument("output_dir", help="Directory to output the new tarred corpora to")
    opts = parser.parse_args()

    # Prepare all the output dirs
    output_dir = opts.output_dir
    training_dir = os.path.join(output_dir, "training")
    dev_dir = os.path.join(output_dir, "dev")
    test_dir = os.path.join(output_dir, "test")
    for dir in [training_dir, dev_dir, test_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    print "Reading file lists"
    with open(opts.dev_list, 'r') as dev_file:
        dev_set = dev_file.read().splitlines()
    with open(opts.test_list, 'r') as test_file:
        test_set = test_file.read().splitlines()

    print "Preparing input corpus"
    corpus = TarredCorpus(opts.input_dir, index=False)
    num_tarballs = len(corpus.tarballs)
    open_tars = []
    try:
        for tar_num, tarball_name in enumerate(corpus.tarballs):
            print "%s (%d/%d)" % (tarball_name, tar_num+1, num_tarballs)
            tarball = tarfile.open(os.path.join(corpus.base_dir, tarball_name))

            # Create a corresponding output archive for each set
            training_tar = tarfile.open(os.path.join(training_dir, tarball_name), 'w')
            open_tars.append(training_tar)
            dev_tar = tarfile.open(os.path.join(dev_dir, tarball_name), 'w')
            open_tars.append(dev_tar)
            test_tar = tarfile.open(os.path.join(test_dir, tarball_name), 'w')
            open_tars.append(test_tar)

            dev_count = training_count = test_count = 0

            # Process each file in the tarball in order
            for tarinfo in tarball:
                # Extract the file from the archive
                f = tarball.extractfile(tarinfo)
                filename = tarinfo.name
                # Output the file to the appropriate set
                if filename in dev_set:
                    dev_tar.addfile(tarinfo, f)
                    dev_count += 1
                elif filename in test_set:
                    test_tar.addfile(tarinfo, f)
                    test_count += 1
                else:
                    training_tar.addfile(tarinfo, f)
                    training_count += 1

            total_files = training_count + test_count + dev_count
            if total_files == 0:
                print "No files in the archive"
            else:
                print "  Training: %.2f%%" % (float(training_count) / total_files * 100.)
                print "  Test:     %.2f%%" % (float(test_count) / total_files * 100.)
                print "  Dev:      %.2f%%" % (float(dev_count) / total_files * 100.)

            while open_tars:
                open_tars.pop().close()
    finally:
        print "Closing files"
        for tar in open_tars:
            tar.close()