import argparse
import os
import tarfile

from brownie.datastructures.mappings import MultiDict
from whim_common.data.compression.tar import SynchronizedTarredCorpora, CorpusSynchronizationError
from whim_common.utils.progress import get_progress_bar


class CallItADay(Exception):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a number of synchronized corpora and check that they are "
                                                 "properly synchronized")
    parser.add_argument("dirs", nargs="+", help="Paths to synced corpora")
    opts = parser.parse_args()
    
    stc = SynchronizedTarredCorpora(opts.dirs, index=False)

    total_errors = 0
    
    # Do the same thing that a STC does when you iterate, but just read filenames
    # We know that each corpus has the same tarballs
    seen = False
    for tarball_filename in stc.tarballs:
        print "\n### Checking %s ###" % tarball_filename
        errors = 0
        # Don't extract the tar files: just iterate over them
        corpus_tars = [
            tarfile.open(os.path.join(corpus.base_dir, tarball_filename), 'r') for corpus in stc.corpora
        ]

        # This line is what takes most of the time
        tar_filenames = [tar.getnames() for tar in corpus_tars]
        # Iterate over the untarred files: we assume all the files in the first corpus are also available
        # in the others
        # Use a lookahead so we can detect when there's a single file extra or missing
        tar_iters = [iter(names) for names in tar_filenames]
        try:
            # Buffer the first two items
            filenames = [tar.next() for tar in tar_iters]
            next_filenames = [tar.next() for tar in tar_iters]

            # Keep going till we reach the end
            while True:
                filename = filenames[0]

                # If this is a directory, we can skip it: its files will show up as separate tarinfos
                if not all(other_name == filename for other_name in filenames):
                    # If there's just one odd one, we might be able to detect what went wrong
                    names = MultiDict((fn, num) for (num, fn) in enumerate(filenames))
                    if len(names) == 2 and min(len(lst) for lst in names.listvalues()) == 1:
                        # There's one name that only features once and the rest are the same
                        odd_one_out = (nums[0] for (name, nums) in names.iterlists() if len(nums) == 1).next()
                        main_name = (name for (name, nums) in names.iterlists() if len(nums) != 1).next()
                        odd_name = filenames[odd_one_out]
                        # Try looking to the next file in the one recognised as odd
                        if next_filenames is None:
                            print "ERROR: One archive, %s, out of sync on last file. Filenames: %s" % (
                                stc.corpora[odd_one_out].base_dir,
                                ", ".join(filenames)
                            )
                        elif next_filenames[odd_one_out] == main_name:
                            # If this is the same as the current file on the others, we've had an insertion
                            print "ERROR: Detected additional file %s in %s, while other archives have %s. " \
                                  "Skipping file" % (odd_name, stc.corpora[odd_one_out].base_dir, main_name)
                            # Skip this additional file so we can continue checking the rest
                            filenames[odd_one_out] = next_filenames[odd_one_out]
                            next_filenames[odd_one_out] = tar_iters[odd_one_out].next()
                            errors += 1
                        elif all(fn == odd_name for fn in
                                 next_filenames[:odd_one_out] + next_filenames[odd_one_out+1:]):
                            # If the others all have the odd name coming up next, the odd one's missed a file
                            print "ERROR: Detected missing file %s in %s, which should be before %s. " \
                                  "Skipping file in others" % (main_name, stc.corpora[odd_one_out].base_dir, odd_name)
                            for i in range(len(filenames)):
                                if i != odd_one_out:
                                    # Skip the missing file in all the other archives so we can continue checking
                                    filenames[i] = next_filenames[i]
                                    next_filenames[i] = tar_iters[i].next()
                            errors += 1
                        else:
                            raise CorpusSynchronizationError(
                                "One archive, %s, out of sync. Not a single addition or deletion, so can't "
                                "get back in sync. Filenames: %s" % (
                                    stc.corpora[odd_one_out].base_dir,
                                    ", ".join(filenames)
                                )
                            )
                    else:
                        raise CorpusSynchronizationError(
                            "Filenames in tarballs (%s in %s) do not correspond: %s. "
                            "Couldn't work out what's going on, so can't continue" %
                            (tarball_filename,
                             ", ".join(corpus.base_dir for corpus in stc.corpora),
                             ", ".join(filenames)))

                if errors > 10:
                    print "Encountered 10 errors in the same archive: calling it a day"
                    raise CallItADay()

                # Move onto the next item
                filenames = next_filenames
                next_filenames = [tar.next() for tar in tar_iters]
        except StopIteration:
            # Reached end of iterations
            if errors:
                print "Reached end of archives, but hit %d errors" % errors
            else:
                print "Completed without errors"
        except CorpusSynchronizationError, e:
            print "ERROR: Unrecoverable synchronization error"
            print e
            if errors:
                print "(%d errors previously in archive)" % errors
        except CallItADay:
            pass
        total_errors += errors

    if total_errors:
        print "\nComplete. Total of %d errors found in corpus" % total_errors
    else:
        print "\nComplete. No errors found"
