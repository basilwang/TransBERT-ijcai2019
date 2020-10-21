from gzip import GzipFile


class FreebaseDumpReader(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with GzipFile(self.filename, "r") as f:
            for line in f:
                # Should be utf-8 strings
                line = line.decode("utf-8")
                # Remove newline and "." at end
                line = line.rstrip("\n .")
                line = line.rstrip("\t")
                # Split on tabs: there should be three items
                yield line.split("\t")

# TODO Wrap up the entities/predicates so that we can distinguish strings and FB entities
# Entities look like this:
#  <http://rdf.freebase.com/ns/g.11b60yp_7y>
# and should be stripped down to:
#  /g/11b60yp_7y
