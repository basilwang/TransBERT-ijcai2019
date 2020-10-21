import os
import random
import shutil
import tarfile
from tempfile import mkdtemp
from whim_common.utils.tarindexer import IndexedTarfile


class TarredCorpus(object):
    def __init__(self, base_dir, index=True, cache_counts=True):
        self.cache_counts = cache_counts
        self.base_dir = base_dir

        self.tar_filenames = [f for f in
                              [os.path.join(root, filename) for root, dirs, files in os.walk(base_dir)
                               for filename in files]
                              if f.endswith(".tar.gz") or f.endswith(".tar")]
        self.tar_filenames.sort()

        self.tarballs = [os.path.basename(f) for f in self.tar_filenames]

        if index and any(f.endswith(".gz") for f in self.tarballs):
            raise ValueError("cannot index gzipped tarballs in %s" % self.base_dir)
        elif index:
            # Prepare an index for each tarball
            self.tarball_indexes = dict([
                (name, IndexedTarfile.index_tar_file(tar_filename, progress=True))
                for (name, tar_filename) in zip(self.tarballs, self.tar_filenames)
            ])
        else:
            self.tarball_indexes = None
        self.__length = None

    def extract_file(self, archive_name, filename):
        if self.tarball_indexes is not None:
            # Look up the path in the tarball's index
            return self.tarball_indexes[archive_name].lookup(filename)
        else:
            with tarfile.open(os.path.join(self.base_dir, archive_name)) as archive:
                return archive.extractfile(filename).read()

    def __iter__(self):
        for __, __, tmp_filename in self.archive_iter():
            yield tmp_filename

    def fill_cache(self, archive_name, filenames):
        self.tarball_indexes[archive_name].fill_cache(filenames)

    def archive_iter(self, subsample=None, start=0):
        # Prepare a temporary directory to extract everything to
        tmp_dir = mkdtemp()
        file_num = -1
        try:
            for tar_name, tarball_filename in zip(self.tarballs, self.tar_filenames):
                # Extract the tarball to the temp dir
                with tarfile.open(tarball_filename, 'r') as tarball:
                    for tarinfo in tarball:
                        file_num += 1
                        # Allow the first portion of the corpus to be skipped
                        if file_num < start:
                            continue
                        # If subsampling, decide whether to extract this file
                        if subsample is not None and random.random() > subsample:
                            # Reject this file
                            continue
                        tarball.extract(tarinfo, tmp_dir)
                        filename = tarinfo.name
                        yield tar_name, filename, os.path.join(tmp_dir, filename)
                        # Remove the file once we're done with it (when we request another)
                        os.remove(os.path.join(tmp_dir, filename))
        finally:
            # Remove the temp dir
            shutil.rmtree(tmp_dir)

    def list_archive_iter(self):
        for tar_name, tarball_filename in zip(self.tarballs, self.tar_filenames):
            tarball = tarfile.open(os.path.join(self.base_dir, tarball_filename), 'r')
            filenames = tarball.getnames()
            for filename in filenames:
                yield tar_name, filename

    def __len__(self):
        # Cache length, as this can take a while
        if self.__length is None:
            if os.path.exists(os.path.join(self.base_dir, "counts")):
                with open(os.path.join(self.base_dir, "counts"), "r") as counts_file:
                    tar_counts = dict((name, int(count)) for (name, __, count) in [
                        line.partition(": ") for line in counts_file.read().splitlines()])
            else:
                tar_counts = {}
                # Count the number of members in each tarball
                for tar_name, tar_filename in zip(self.tarballs, self.tar_filenames):
                    with tarfile.open(tar_filename, "r") as tarball:
                        tar_length = len(tarball.getnames())
                        tar_counts[tar_name] = tar_length
                if self.cache_counts:
                    # Store these counts for next time
                    with open(os.path.join(self.base_dir, "counts"), "w") as counts_file:
                        counts_file.write("\n".join("%s: %d" % (name, count) for (name, count) in tar_counts.items()))
            self.__length = sum(tar_counts.values())
        return self.__length


class SynchronizedTarredCorpora(object):
    def __init__(self, base_dirs, index=True):
        self.corpora = [TarredCorpus(base_dir, index=index) for base_dir in base_dirs]
        self.tarballs = self.corpora[0].tarballs
        # Check that the corpora have the same tarballs in them
        if not all(c.tarballs == self.tarballs for c in self.corpora):
            raise CorpusSynchronizationError("not all corpora have the same tarballs in them, cannot synchronize: %s" %
                                             ", ".join(base_dirs))

    def __iter__(self):
        # Prepare a temporary directory to extract everything to
        tmp_dir = mkdtemp()
        try:
            # Make a subdir for each corpus
            corpus_dirs = [os.path.join(tmp_dir, "corpus%d" % corpus_num) for corpus_num in range(len(self.corpora))]
            for corpus_dir in corpus_dirs:
                os.makedirs(corpus_dir)

            # We know that each corpus has the same tarballs
            for tarball_filename in self.tarballs:
                # Don't extract the tar files: just iterate over them
                corpus_tars = [
                    tarfile.open(os.path.join(corpus.base_dir, tarball_filename), 'r') for corpus in self.corpora
                ]

                # Iterate over the untarred files: we assume all the files in the first corpus are also available
                # in the others
                for tarinfos in zip(*corpus_tars):
                    filename = tarinfos[0].name
                    if tarinfos[0].isdir():
                        # If this is a directory, we don't extract it: its files will show up as separate tarinfos
                        continue

                    if not all(tarinfo.name == filename for tarinfo in tarinfos):
                        raise IOError("filenames in tarballs (%s in %s) do not correspond: %s" %
                                      (tarball_filename,
                                       ", ".join(corpus.base_dir for corpus in self.corpora),
                                       ", ".join(tarinfo.name for tarinfo in tarinfos)))

                    corpus_filenames = []
                    for tarball, tarinfo, corpus_dir in zip(corpus_tars, tarinfos, corpus_dirs):
                        tarball.extract(tarinfo, corpus_dir)
                        corpus_filenames.append(os.path.join(corpus_dir, tarinfo.name))

                    # Just a single file
                    yield filename, corpus_filenames

                    # Remove the file once we're done with it (when we request another)
                    for corpus_filename in corpus_filenames:
                        os.remove(corpus_filename)
        finally:
            # Remove the temp dir
            shutil.rmtree(tmp_dir)

    def __len__(self):
        return len(self.corpora[0])


class CorpusSynchronizationError(Exception):
    pass


def detect_tarred_corpus(dirname):
    """
    Check the given directory for tar files. Return True if any are found, indicating that we should treat
    this as a tarred corpus. There may be non-tarred files in there (indices, etc), but if there are any
    tar files it's probably a tarred corpus.

    """
    for f in os.listdir(dirname):
        if f.endswith(".tar.gz") or f.endswith(".tar"):
            return True
    return False