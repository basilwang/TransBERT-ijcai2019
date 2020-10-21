"""
Dependency data structures. Note that these are separate from the data structures in the
C&C dependency module, which are really grammatical relations. That module includes a method to
convert from those to these simpler structures.

These dependency graphs correspond to the sort of thing produced by a dependency parser.

"""


class Dependency(object):
    """
    A normal dependency relation, with just a head and dependent.

    """
    def __init__(self, dep_type, head, dependent, type_specializer=None):
        # String dependency type
        self.type_specializer = type_specializer
        self.dep_type = dep_type
        # Head and dependent, just indices
        self.head = head
        self.dependent = dependent

    def __str__(self):
        return "%s(%d, %d)" % (self.specific_type, self.head, self.dependent)

    def __repr__(self):
        return str(self)

    @property
    def specific_type(self):
        if self.type_specializer is not None:
            return "%s_%s" % (self.dep_type, self.type_specializer)
        else:
            return self.dep_type


class DependencyGraph(object):
    def __init__(self, dependencies=[], word_map=None):
        # Dict word index -> word string
        self.words = word_map
        # List of dependencies (see above)
        self.dependencies = dependencies
        self._heads = None

    def head(self, dependent):
        if self._heads is None:
            # Build the head map for easier lookup
            self._heads = {}
            for dep in self.dependencies:
                self._heads[dep.dependent] = (dep.head, dep.dep_type)
        # Return the head index and the dependency type
        if dependent in self._heads:
            return self._heads[dependent]
        elif dependent == 0:
            raise ValueError("root node has no head")
        else:
            raise ValueError("no dependency in graph with word %s as a dependent" % dependent)

    @staticmethod
    def from_file(filename):
        """
        Read in a dependency file and return a list of dependency graphs.

        """
        with open(filename, 'r') as infile:
            data = infile.read()
        lines = [l.strip() for l in data.splitlines()]
        # A blank dep graph at the end gets lost if we don't add an extra line break
        lines.append("")

        graph_strings = []
        current_graph = []
        blank_lines = 0
        for line in lines:
            if not line:
                # Blank line
                blank_lines += 1
                if current_graph:
                    # Blank line marks end of current graph
                    graph_strings.append("\n".join(current_graph))
                    current_graph = []
                if blank_lines % 2 == 0:
                    # Second blank line in a row: empty dep graph
                    graph_strings.append("")
            else:
                blank_lines = 0
                current_graph.append(line)

        if current_graph:
            # Finish up the last graph
            graph_strings.append("\n".join(current_graph))

        # Build a graph from each graph string
        return [DependencyGraph.from_string(g) for g in graph_strings]

    @staticmethod
    def from_string(string):
        """
        Build a single dependency graph from a string.

        """
        # Normal dependencies look like:
        #  type(word0_id0, word1_id1)
        deps = []
        words = {}
        for line in string.splitlines():
            # Ignore any blank lines
            if line:
                dep_type, __, line = line.partition("(")
                word_id0, __, word_id1 = line.strip(")").partition(", ")
                word0, __, id0 = word_id0.partition("_")
                word1, __, id1 = word_id1.partition("_")
                id0, id1 = int(id0), int(id1)
                # Update the word map
                words[id0] = word0
                words[id1] = word1
                deps.append(Dependency(dep_type, id0, id1))
        return DependencyGraph(deps, word_map=words)

    @staticmethod
    def from_conll(string):
        """
        Read a graph from the CoNLL format, used by Malt.

        """
        words = {}
        dependencies = []

        for line in string.splitlines():
            if line:
                # Split the line into its tab-separated fields
                fields = line.split("\t")
                # First col is the word number
                word_num = int(fields[0])
                # Then word and lemma
                words[word_num] = fields[1]
                # Seventh column contains the word's head
                head_word = int(fields[6])
                dep_type = fields[7]
                dependencies.append(Dependency(dep_type, head_word, word_num))
        return DependencyGraph(dependencies, words)

    @staticmethod
    def from_conll_file(filename):
        """
        Read in a list of graphs from a file in CoNLL format.

        """
        with open(filename, 'r') as infile:
            graph_lines = []
            graphs = []

            for line in infile:
                line = line.strip("\n")
                if not line:
                    # Blank line signals end of current graph
                    graphs.append(DependencyGraph.from_conll("\n".join(graph_lines)))
                    graph_lines = []
                else:
                    graph_lines.append(line)

            if graph_lines:
                graphs.append(DependencyGraph.from_conll("\n".join(graph_lines)))
        return graphs
