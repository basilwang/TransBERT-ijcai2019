import numpy, numpy.matlib
from operator import itemgetter
from whim_common.utils.progress import get_progress_bar


class PartialOrder(object):
    def __init__(self, elements):
        self._matrix = numpy.zeros((len(elements), len(elements)), dtype=numpy.int8)

        self._element_ids = dict([(element, id) for (id, element) in enumerate(elements)])
        self._id_elements = dict(enumerate(elements))

        self.elements = elements

    def add(self, first, second, check_transitive=False):
        f = self._element_ids[first]
        s = self._element_ids[second]
        if self._matrix[f, s] == -1:
            raise OrderConflict
        elif self._matrix[f, s] == 1:
            # Already set: skip
            return

        # Add all transitive orderings
        # This includes the actual ordering we're adding
        before_f = numpy.concatenate((numpy.where(self._matrix[f, :] == -1)[0], numpy.array([f])))

        # Everything that was after second is now after everything that was before first (and the first itself)
        s_positives = self._matrix[s, :].copy()
        s_positives[s_positives < 0] = 0
        # Include the second element itself
        s_positives[s] = 1
        self._add_all(before_f, s_positives, check=check_transitive)

    def _add_all(self, first_ids, row, check=False):
        row_trans = row[numpy.newaxis].T
        if check and (
                numpy.any(numpy.abs(self._matrix[first_ids, :] - row) > 1)
                or numpy.any(numpy.abs(self._matrix[:, first_ids] + row_trans) > 1)):
            # This should be impossible
            # Print out what the conflict was
            conflict1 = numpy.where(numpy.abs(self._matrix[first_ids, :] - row) > 1)
            conflict2 = numpy.where(numpy.abs(self._matrix[:, first_ids] + row_trans) > 1)
            conflicts = ["%s, %s" % (self._id_elements[first_ids[id0]], self._id_elements[id1]) for (id0, id1) in zip(*conflict1)]
            conflicts.extend(["%s, %s" % (self._id_elements[id0], self._id_elements[first_ids[id1]]) for (id0, id1) in zip(*conflict2)])
            # There's at least one ordering that conflicts with what's already there
            raise UnexpectedOrderConflict("order conflict in adding transitive orderings: %s" % ", ".join(conflicts))
        self._matrix[first_ids, :] += row
        self._matrix[:, first_ids] -= row_trans
        # We might have set some 1s it 2s or -1s to -2s
        numpy.clip(self._matrix, -1, 1, self._matrix)

    def print_order(self):
        xs, ys = numpy.where(self._matrix > 0)
        for x, y in zip(xs, ys):
            print "%s -> %s" % (self._id_elements[x], self._id_elements[y])

    def in_order(self, first, second):
        f = self._element_ids[first]
        s = self._element_ids[second]
        order = self._matrix[f, s]
        if order == 1:
            return True
        elif order == -1:
            return False
        else:
            return None

    def after(self, element):
        return [self._id_elements[id] for id in numpy.where(self._matrix[self._element_ids[element], :] == 1)]

    def before(self, element):
        return [self._id_elements[id] for id in numpy.where(self._matrix[self._element_ids[element], :] == -1)]

    @property
    def is_total(self):
        # Diagonal should be 0, so check that all other cells are filled
        return numpy.abs(self._matrix).sum() == (len(self.elements) - 1) ** 2

    def total_order(self, approximate=False):
        if not approximate and not self.is_total:
            return None
        # We can just order elements by the number of other elements that come after them
        # TODO Not tested this
        return [self._id_elements[el] for (el, count_after) in \
                reversed(sorted(enumerate(self._matrix.sum(axis=1)), key=itemgetter(1)))]

    @staticmethod
    def from_counts(elements, counts, progress=False):
        order = PartialOrder(elements)

        if progress:
            pbar = get_progress_bar(len(counts))
        else:
            pbar = None

        for i, ((first, second), count) in enumerate(reversed(sorted(counts.items(), key=itemgetter(1)))):
            if first in elements and second in elements and first != second:
                try:
                    order.add(first, second)
                except OrderConflict:
                    # No problem: skip this one
                    pass

            if pbar:
                pbar.update(i)

        if pbar:
            pbar.finish()

        return order

    def write_matrix(self, filename):
        numpy.save(filename, self._matrix)


class OrderConflict(Exception):
    pass


class UnexpectedOrderConflict(Exception):
    pass