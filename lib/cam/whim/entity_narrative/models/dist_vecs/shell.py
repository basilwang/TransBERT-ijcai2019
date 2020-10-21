import numpy

from cam.whim.entity_narrative.shell.commands import ModelShell


class DistVecsShell(ModelShell):
    def do_neighbours(self, line, **kwargs):
        entities, events, line = ModelShell.parse_event_context(line)
        if line.strip():
            print "Ignoring remainder of input: %s" % line

        # Get the vector projection of the events
        projection = self.model.project_chains([(entities[0], events)])[0]
        # Get similarity of this to each of the vectors
        # Normalize
        projection /= numpy.sqrt((projection ** 2.).sum())
        # Vectors may not be normalized
        vectors = self.model.vectors
        # Masks aren't stopping division warnings. Poo, can't be arsed to sort it now
        vectors_norms = numpy.ma.masked_equal(numpy.ma.masked_invalid(
            numpy.sqrt((vectors ** 2.).sum(axis=1))[:, numpy.newaxis]), 0.)
        vectors /= vectors_norms
        # Now the cosine is just the dot product
        scores = numpy.dot(vectors, projection)
        scores = numpy.ma.masked_invalid(scores)

        neighbours = list(reversed(scores.argsort(fill_value=0.)))
        for neighbour_id in neighbours[:10]:
            print self.model.dictionary[neighbour_id], scores[neighbour_id]