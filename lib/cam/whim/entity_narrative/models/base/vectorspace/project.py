import argparse
import cPickle as pickle
from cam.whim.entity_narrative.chains.document import RichEventDocumentCorpus
from cam.whim.entity_narrative.models.base.model import NarrativeChainModel
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel
from whim_common.utils.progress import get_progress_bar


class VectorCorpus(object):
    """
    Data structure for storing vector projections of a whole corpus.
    Iterates over event chains or events in the corpus, computing their projections and storing
    them in the vector corpus, along with information to retrieve the original data points.

    """
    def __init__(self, data_points, corpus, vector_size, model_type, model_name):
        self.model_name = model_name
        self.model_type = model_type
        self.vector_size = vector_size
        self.corpus = corpus
        self.data_points = data_points

        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = NarrativeChainModel.load_by_type(self.model_type, self.model_name)
        return self._model

    @staticmethod
    def build_from_docs(corpus, model_type, model_name, progress=False, buffer_size=10000, project_events=False):
        data_points = []
        for vector, source, chain in VectorCorpus.project_from_docs(corpus,
                                                                    model_type,
                                                                    model_name,
                                                                    progress=progress,
                                                                    buffer_size=buffer_size,
                                                                    project_events=project_events):

            data_points.append((vector, source))

        if data_points:
            vector_size = data_points[0][0].shape[0]
        else:
            vector_size = 0
        return VectorCorpus(data_points, corpus, vector_size, model_type, model_name)

    @staticmethod
    def project_from_docs(corpus, model_type, model_name, progress=False, buffer_size=10000, project_events=False,
                          filter_chains=None):
        """
        Project events or chains directly from a document corpus using a vector projection model.
        Yields vectors paired with their events/chains.

        :param corpus:
        :param model_type: narrative chain model type
        :param model_name:
        :param progress: show progress while projecting
        :param buffer_size: batch size to project at once
        :param project_events: project individual events instead of whole chains
        :return:
        """
        model = NarrativeChainModel.load_by_type(model_type, model_name)
        # Allow models that are full vector space models, or ones that implement the projection function we need
        if not (isinstance(model, VectorSpaceNarrativeChainModel) or
                (project_events and hasattr(model, "project_events") or
                (not project_events and hasattr(model, "project_chains")))):
            raise ValueError("can only build a vector corpus using a vector space model or one that provides "
                             "projection functions, not model type %s" % model_type)

        if progress:
            total_docs = len(corpus)
            pbar = get_progress_bar(total_docs, title="Projecting corpus")
        else:
            pbar = None

        # Instead of projecting each chain one by one, or doc by doc, buffer lots and do them in a batch
        # This is hugely faster!
        chain_buffer = []
        source_buffer = []
        for i, (archive, filename, doc) in enumerate(corpus.archive_iter()):
            if pbar:
                pbar.update(i)

            chains = doc.get_chains()
            if filter_chains is not None:
                chains = filter_chains(chains)

            if len(chains):
                if project_events:
                    # Add individual events to the buffers
                    chain_buffer.extend([(entity, event) for (entity, events) in chains for event in events])
                    source_buffer.extend([(archive, filename, chain_num, event_num)
                                          for chain_num in range(len(chains))
                                          for event_num in range(len(chains[chain_num][1]))])
                else:
                    # Project whole chains
                    chain_buffer.extend(chains)
                    source_buffer.extend([(archive, filename, chain_num) for chain_num in range(len(chains))])

            if len(chain_buffer) > buffer_size:
                # Project chains/events into vector space using model
                if project_events:
                    chain_vectors = model.project_events(chain_buffer)
                else:
                    chain_vectors = model.project_chains(chain_buffer)
                for j, (source, chain) in enumerate(zip(source_buffer, chain_buffer)):
                    yield (chain_vectors[j], source, chain)
                chain_buffer = []
                source_buffer = []

        if chain_buffer:
            # Clear up remaining buffer
            if project_events:
                chain_vectors = model.project_events(chain_buffer)
            else:
                chain_vectors = model.project_chains(chain_buffer)
            for j, (source, chain) in enumerate(zip(source_buffer, chain_buffer)):
                yield (chain_vectors[j], source, chain)

        if pbar:
            pbar.finish()

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump({
                "data_points": self.data_points,
                "corpus_dir": self.corpus.directory,
                "corpus_tarred": self.corpus.tarred,
                "vector_size": self.vector_size,
                "model_type": self.model_type,
                "model_name": self.model_name,
            }, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = pickle.load(f)
        corpus = RichEventDocumentCorpus(data["corpus_dir"], tarred=data["corpus_tarred"])
        return VectorCorpus(data["data_points"], corpus, data["vector_size"], data["model_type"], data["model_name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce vector projections of all the chains/events in a corpus")
    parser.add_argument("corpus_dir", help="Path to rich event chain corpus to project into vector space")
    parser.add_argument("model_type", help="Model type to load for projection")
    parser.add_argument("model_name", help="Name of model to load for projection")
    parser.add_argument("output_file", help="File to output the vector corpus to")
    parser.add_argument("--tarred", action="store_true", help="The corpus is tarred")
    parser.add_argument("--events", action="store_true", help="Project each individual event, not whole chains")
    opts = parser.parse_args()

    project_events = opts.events

    print "Loading corpus"
    corpus = RichEventDocumentCorpus(opts.corpus_dir, tarred=opts.tarred)
    # Doing this caches the corpus length, which we're going to need anyway
    num_docs = len(corpus)
    if project_events:
        print "Projecting events from %d documents" % num_docs
    else:
        print "Projecting chains from %d documents" % num_docs
    vcorpus = VectorCorpus.build_from_docs(corpus, opts.model_type, opts.model_name, progress=True,
                                           project_events=project_events)

    print "Outputting corpus to %s" % opts.output_file
    vcorpus.save(opts.output_file)