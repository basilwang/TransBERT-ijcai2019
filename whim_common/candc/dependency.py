from whim_common.data.dependency import Dependency as SimpleDependency, DependencyGraph as SimpleDependencyGraph


class Dependency(object):
    """
    A dependency relation in a dependency graph, or more accurately, usually, a grammatical
    relation (Briscoe and Caroll), as output by C&C.

    """
    def __init__(self, dep_type, args):
        # String dependency type
        self.dep_type = dep_type
        # List of word indices
        self.args = args

    def __str__(self):
        return "(%s, %s)" % (self.dep_type,
                             ", ".join("%s%s" % (word, "_%d" % index if index is not None else "")
                                       for (word, index) in self.args))

    def __repr__(self):
        return str(self)

    def to_simple_dependency(self):
        # Pull out the head and dependent
        # Their position depends on the dependency type
        # Some things don't correspond to B&C's original GR spec: C&C adds det(hd, dep); aux without a type
        type_specializer = None
        if self.dep_type in ["dependent", "mod", "cmod", "xmod", "ncmod", "detmod", "arg_mod", "iobj", "xcomp",
                             "ccomp"]:
            # dep(X, head, dependent, [Y])
            # arg_mod also has another arg, which we ignore
            head = self.args[1][1]
            dependent = self.args[2][1]
            # If the first arg is not empty, include it as part of the dependency type
            if self.args[0] is not None:
                type_specializer = self.args[0][0]
        elif self.dep_type in ["arg", "subj_or_dobj", "subj", "csubj", "xsubj", "ncsubj", "comp", "obj", "dobj",
                               "obj2", "clausal", "det", "aux"]:
            # dep(head, dependent, [X])
            head = self.args[0][1]
            dependent = self.args[1][1]
        elif self.dep_type == "conj":
            # Dependencies get distributed over conjunction, but the GRs also include one relation for the conjunction
            # We can leave this out of the simple graph
            return None
        else:
            raise ValueError("unknown dependency type '%s': cannot convert to a simple dependency without knowing "
                             "where the head is" % self.dep_type)
        return SimpleDependency(self.dep_type, head, dependent, type_specializer=type_specializer)


class DependencyGraph(object):
    """
    Really a graph of grammatical relations, which are somewhat richer than dependencies.
    For backwards compatibility, these are referred to as dependencies, in contrast to
    simple dependencies (see below), which are normal dependencies.

    """
    def __init__(self, dependencies=[], word_map=None):
        # Dict word index -> word string
        self._words = word_map
        # List of dependencies (see above)
        self.dependencies = dependencies

    @property
    def words(self):
        if self._words is None:
            # Build the word map
            self._words = {}
            for dep in self.dependencies:
                for word, index in dep.args:
                    if word is not None and index is not None:
                        self._words[index] = word
        return self._words

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
        #  (type word0_id0 word1_id1 [word2_id2])
        deps = []
        for line in string.splitlines():
            # Ignore any blank lines
            if line:
                tokens = line.strip("()").split()
                dep_type = tokens[0]
                args = [split_lex(t) for t in tokens[1:]]
                deps.append(Dependency(dep_type, args))

        return DependencyGraph(deps)

    def get_by_arg0(self, word):
        return [dep for dep in self.dependencies if len(dep.args) > 0 and dep.args[0][1] == word]

    def get_by_arg1(self, word):
        return [dep for dep in self.dependencies if len(dep.args) > 1 and dep.args[1][1] == word]

    def to_simple_dependencies(self):
        """
        Produce a standard dependency graph (SimpleDependencyGraph) from the
        Briscoe and Caroll relations.

        """
        deps = [dep.to_simple_dependency() for dep in self.dependencies]
        deps = [dep for dep in deps if dep is not None]
        return SimpleDependencyGraph(deps, word_map=self.words)


def split_lex(s):
    if s == "_":
        return None, None
    word, __, index = s.rpartition("_")

    if not word:
        # This is a non-word marker, e.g. poss or obj
        word = index
        index = None
    else:
        index = int(index)
    return word, index


def build_outgoing_edges(dep_graph):
    """
    Builds something more like an actual graph data structure from the arc-only representation of
    the graph used by default.

    """
    word_edges = {}
    for dependency in dep_graph.dependencies:
        # The source of the edge is the first argument
        source_word = dependency.args[0][1]
        for arg_num, (__, dest_word) in enumerate(dependency.args[1:]):
            # Add an edge from the source word to each other arg
            word_edges.setdefault(source_word, []).append(((dependency.dep_type, arg_num), dest_word))
    return word_edges


def build_incoming_edges(dep_graph):
    """
    The upward version of build_outgoing_edges().

    """
    word_edges = {}
    for dependency in dep_graph.dependencies:
        # The source of the edge is the first argument
        source_word = dependency.args[0][1]
        for arg_num, (__, dest_word) in enumerate(dependency.args[1:]):
            # Add an edge from the source word to each other arg, indexed upwards
            word_edges.setdefault(dest_word, []).append(((dependency.dep_type, arg_num), source_word))
    return word_edges
