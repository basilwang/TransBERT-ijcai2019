#!/usr/bin/python3
#
# tarindexer
# index tar files for fast access
#
# Copyright (c) 2013 Tom Wallroth
# Modified by Mark Granroth-Wilding
#
# Sources on github:
# http://github.com/devsnd/tarindexer/
#
# licensed under GNU GPL version 3 (or later)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#
from __future__ import absolute_import
import tarfile
import os
import codecs

from whim_common.utils.logging import get_console_logger
from whim_common.utils.progress import get_progress_bar


log = get_console_logger("Tar indexer")


class IndexedTarfile(object):
    def __init__(self, tar_filename, index_filename):
        self.index_filename = index_filename
        self.tar_filename = tar_filename

        self._cache = {}

    def clear_cache(self):
        self._cache = {}

    def fill_cache(self, paths):
        """
        To speed up loading lots of files from the same archive, fill the cache with a whole list of
        paths at once, which will then be quick to read.
        """
        self._cache.update(
            bulk_lookup_in_index(self.tar_filename, self.index_filename, paths)
        )

    @staticmethod
    def index_tar_file(tar_filename, progress=False):
        # Prepare an index (or use one that's already there)
        index_filename = index_tar(tar_filename, progress=progress)
        return IndexedTarfile(tar_filename, index_filename)

    def lookup(self, path):
        if path in self._cache:
            # No need to do a costly lookup: we've already done it
            return self._cache[path]
        else:
            return lookup_in_index(self.tar_filename, self.index_filename, path)


def index_tar(tar_filename, progress=False):
    # Pick a filename for the index file
    basename = tar_filename
    if ".gz" in tar_filename:
        basename = basename.rpartition(".gz")[0]
    basename = basename.rpartition(".tar")[0]
    index_filename = "%s.tar_index" % basename

    if not os.path.exists(index_filename):
        # Build the index
        log.info("Building tar index %s" % index_filename)
        index_tar_to_file(tar_filename, index_filename, progress=progress)
    return index_filename


def index_tar_to_file(tar_filename, index_filename, progress=False):
    filesize = os.path.getsize(tar_filename)

    if progress:
        pbar = get_progress_bar(filesize, title="Indexing tarfile")
    else:
        pbar = None

    with tarfile.open(tar_filename, 'r|') as db:
        with open(index_filename, 'w') as outfile:
            counter = 0
            for tarinfo in db:
                currentseek = tarinfo.offset_data
                rec = "%s %d %d\n" % (tarinfo.name, tarinfo.offset_data, tarinfo.size)
                outfile.write(rec)

                counter += 1
                if counter % 1000 == 0:
                    # free ram...
                    db.members = []
                if pbar and counter % 100 == 0:
                    pbar.update(currentseek)

    if pbar:
        pbar.finish()


def lookup_in_index(tar_filename, index_filename, path):
    with open(tar_filename, 'rb') as tar:
        with open(index_filename, 'r') as index_file:
            for line in index_file:
                m = line[:-1].rsplit(" ", 2)
                if path == m[0]:
                    tar.seek(int(m[1]))
                    return codecs.decode(tar.read(int(m[2])), 'ASCII')


def bulk_lookup_in_index(tar_filename, index_filename, paths):
    loaded = {}
    with open(tar_filename, 'rb') as tar:
        with open(index_filename, 'r') as index_file:
            for line in index_file:
                m = line[:-1].rsplit(" ", 2)
                if m[0] in paths:
                    tar.seek(int(m[1]))
                    loaded[m[0]] = codecs.decode(tar.read(int(m[2])), 'ASCII')

    # Check we loaded all the paths
    for path in paths:
        if path not in loaded:
            raise IOError("path %s was not found in archive's index %s" % (path, tar_filename))
    return loaded