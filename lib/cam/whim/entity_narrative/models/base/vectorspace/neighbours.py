import cPickle as pickle
from operator import itemgetter
from redis.exceptions import ConnectionError
from redis import Redis
import os

from cam.whim.entity_narrative.models.base.model import NarrativeChainModel
from cam.whim.entity_narrative.models.base.vectorspace.project import VectorCorpus
from whim_common.utils.logging import get_console_logger
from nearpy import Engine
from nearpy.filters.vectorfilter import VectorFilter
from nearpy.hashes import RandomBinaryProjections
from nearpy.storage import RedisStorage


class NearestNeighbourFinder(object):
    """
    Using an already-projected corpus in the form of a VectorCorpus, allow easy queries to find nearest
    neighbour event chains in the corpus, given a new event chain.

    """
    def __init__(self, model_type, model_name, hash, corpus_path, with_events=False):
        self.hash = hash
        self.corpus_path = corpus_path
        self.model_type = model_type
        self.model_name = model_name
        self.with_events = with_events

        self.model = None
        self.search_engine = None

    def init_engine(self, redis_port=6379):
        # Need to load the model to get information about it
        self.model = NarrativeChainModel.load_by_type(self.model_type, self.model_name)
        vector_size = self.model.vector_size

        # Point the Redis server to the model's database
        db_filename = "vectors.rdb"
        model_dir = self.model.get_model_directory(self.model_name)

        # Prepare an engine for reading vectors from
        try:
            redis = Redis(host='localhost', port=redis_port, db=0)
        except ConnectionError, e:
            raise RuntimeError("could not connect to redis server on port %s. Is it running? (%s)" % (redis_port, e))
        # Set the storage location to be in the model's directory/file
        redis.config_set("dbfilename", "vectors.rdb")
        redis.config_set("dir", model_dir)

        redis_storage = RedisStorage(redis)
        self.search_engine = Engine(vector_size, lshashes=[self.hash], storage=redis_storage,
                                    fetch_vector_filters=[UniqueVectorFilter()], vector_filters=[])

    @staticmethod
    def load(model_type, model_name, redis_port=6379):
        model = NarrativeChainModel.load_by_type(model_type, model_name)
        model_dir = model.get_model_directory(model_name)

        if not os.path.exists(os.path.join(model_dir, "neighbour_finder")):
            raise IOError("no nearest-neighbour finder has been prepared for the model %s/%s" %
                          (model_type, model_name))

        with open(os.path.join(model_dir, "neighbour_finder"), "r") as finder_file:
            finder = pickle.load(finder_file)

        finder.init_engine(redis_port=redis_port)
        return finder

    @staticmethod
    def build_from_document_corpus(corpus, model_type, model_name,
                                   progress=False, project_events=False, include_events=False, hash_size=50,
                                   log=None, redis_port=6379, filter_chains=None):
        if log is None:
            log = get_console_logger("neighbour indexing")

        log.info("Loading model %s/%s" % (model_type, model_name))
        model = NarrativeChainModel.load_by_type(model_type, model_name)
        vector_size = model.vector_size

        db_filename = "vectors.rdb"
        # Make sure the model directory exists, so we can get the Redis server pointing there
        model_dir = model.get_model_directory(model_name)
        # If the Redis stored db already exists, remove it, so that we don't end up adding to old data
        if os.path.exists(os.path.join(model_dir, db_filename)):
            os.remove(os.path.join(model_dir, db_filename))
        log.info("Storing vectors in %s" % os.path.join(model_dir, db_filename))

        log.info("Preparing neighbour search hash")
        # Create binary hash
        binary_hash = RandomBinaryProjections("%s:%s_binary_hash" % (model_type, model_name), hash_size)

        log.info("Connecting to Redis server on port %d" % redis_port)
        # Prepare an engine for storing the vectors in
        try:
            redis = Redis(host='localhost', port=redis_port, db=0)
        except ConnectionError, e:
            raise RuntimeError("could not connect to redis server on port %s. Is it running? (%s)" % (redis_port, e))
        # Set the storage location to be in the model's directory
        redis.config_set("dbfilename", "vectors.rdb")
        redis.config_set("dir", model_dir)
        # Use this as the storage engine for the nearest-neighbour index
        redis_storage = RedisStorage(redis)
        search_engine = Engine(vector_size, lshashes=[binary_hash], storage=redis_storage)

        for vector, source, chain in VectorCorpus.project_from_docs(corpus,
                                                                    model_type,
                                                                    model_name,
                                                                    progress=progress,
                                                                    buffer_size=10000,
                                                                    project_events=project_events,
                                                                    filter_chains=filter_chains):
            data = (source, chain) if include_events else source
            search_engine.store_vector(vector, data)

        finder = NearestNeighbourFinder(model_type, model_name, binary_hash, corpus.directory,
                                        with_events=include_events)
        log.info("Storing finder in %s" % os.path.join(model_dir, "neighbour_finder"))
        with open(os.path.join(model_dir, "neighbour_finder"), "w") as finder_file:
            pickle.dump(finder, finder_file)
        return finder

    def find_nearest_neighbours(self, vector):
        # Find nearest neighbours in the vector space
        results = self.search_engine.neighbours(vector)
        # Sort the results by distance (lower is better)
        results.sort(key=itemgetter(2))

        if self.with_events:
            # The event instances have been stored with the data
            for (result_vector, (source, chain), score) in results:
                yield chain, source, result_vector, score
        else:
            for (result_vector, source, score) in results:
                yield None, source, result_vector, score

    def find_nearest_neighbour(self, vector):
        results = list(self.find_nearest_neighbours(vector))
        if results:
            return min(results, key=itemgetter(3))
        else:
            return None

    def get_hash_bucket(self, vector):
        """
        Return all the vectors (and associated data) from the same hash bucket as the input vector.
        """
        bucket = []
        for lshash in self.search_engine.lshashes:
            for bucket_key in lshash.hash_vector(vector, querying=True):
                bucket_content = self.search_engine.storage.get_bucket(lshash.hash_name, bucket_key)
                bucket.extend(bucket_content)
        return bucket


class UniqueVectorFilter(VectorFilter):
    """
    Like UniqueFilter, but filters on the vector instead of the data, so that only a single instances of
    each vector is return.

    """
    def filter_vectors(self, input_list):
        unique_dict = {}
        for v in input_list:
            unique_dict[v[0].tostring()] = v
        return list(unique_dict.values())
