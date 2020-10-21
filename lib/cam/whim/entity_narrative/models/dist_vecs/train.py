from collections import Counter
from sklearn.decomposition.truncated_svd import TruncatedSVD

from scipy.sparse import csr_matrix, lil_matrix
import numpy

from cam.whim.entity_narrative.models.base.train import NarrativeChainModelTrainer
from whim_common.utils.progress import get_progress_bar
from gensim.corpora.dictionary import Dictionary


class DistributionalVectorsTrainer(NarrativeChainModelTrainer):
    def prepare_arguments(self, parser):
        """ Add arguments to an argparse parser for training. """
        parser.add_argument("--pair-threshold", type=int,
                            help="Apply a threshold number of counts to pairs")
        parser.add_argument("--event-threshold", type=int,
                            help="Apply a threshold number of counts to events")
        parser.add_argument("--only-verb", action="store_true",
                            help="Only use the verbs as context, not the verb+dependency, which is the default")
        parser.add_argument("--adj", action="store_true", help="Include predicative adjectives")
        parser.add_argument("--pmi", action="store_true", help="Apply PMI to the vectors")
        parser.add_argument("--ppmi", action="store_true", help="Apply PPMI to the vectors")
        parser.add_argument("--svd", type=int, help="Apply SVD to the vector space (after PMI, if using that). "
                                                    "Reduce to the given number of dimensions")

    def train(self, model_name, corpus, log, opts, chain_features=None):
        from cam.whim.entity_narrative.models.dist_vecs.model import DistributionalVectorsNarrativeChainModel
        log.info("Training context vectors model")

        training_metadata = {
            "data": corpus.directory,
            "pmi": opts.pmi or opts.ppmi,
            "ppmi": opts.ppmi,
        }

        log.info("Extracting event counts")
        pbar = get_progress_bar(len(corpus), title="Event feature extraction")
        # Loop over all the chains again to collect events
        event_counts = Counter()
        for doc_num, document in enumerate(corpus):
            chains = document.get_chains()
            if len(chains):
                event_chains = list(
                    DistributionalVectorsNarrativeChainModel.extract_chain_feature_lists(chains, only_verb=opts.only_verb,
                                                                                  adjectives=opts.adj)
                )
                # Count all the events
                for chain in event_chains:
                    event_counts.update(chain)

            pbar.update(doc_num)
        pbar.finish()

        if opts.event_threshold is not None and opts.event_threshold > 0:
            log.info("Applying event threshold")
            # Apply a threshold event count
            to_remove = [event for (event, count) in event_counts.items() if count < opts.event_threshold]
            pbar = get_progress_bar(len(to_remove), title="Filtering counts")
            for i, event in enumerate(to_remove):
                del event_counts[event]
                pbar.update(i)
            pbar.finish()

        log.info("Extracting pair counts")
        pbar = get_progress_bar(len(corpus), title="Pair feature extraction")
        # Loop over all the chains again to collect pairs of events
        pair_counts = Counter()
        for doc_num, document in enumerate(corpus):
            chains = document.get_chains()
            if len(chains):
                event_chains = list(
                    DistributionalVectorsNarrativeChainModel.extract_chain_feature_lists(chains, only_verb=opts.only_verb,
                                                                                  adjectives=opts.adj)
                )
                # Count all the events
                for chain in event_chains:
                    # Count all pairs
                    pairs = []
                    for i in range(len(chain)-1):
                        for j in range(i+1, len(chain)):
                            if chain[i] in event_counts and chain[j] in event_counts:
                                pairs.append(tuple(sorted([chain[i], chain[j]])))
                    pair_counts.update(pairs)

            pbar.update(doc_num)
        pbar.finish()

        if opts.pair_threshold is not None and opts.pair_threshold > 0:
            log.info("Applying pair threshold")
            # Apply a threshold pair count
            to_remove = [pair for (pair, count) in pair_counts.items() if count < opts.pair_threshold]
            if to_remove:
                pbar = get_progress_bar(len(to_remove), title="Filtering pair counts")
                for i, pair in enumerate(to_remove):
                    del pair_counts[pair]
                    pbar.update(i)
                pbar.finish()
            else:
                log.info("No counts removed")

        # Create a dictionary of the remaining vocabulary
        log.info("Building dictionary")
        dictionary = Dictionary([[event] for event in event_counts.keys()])
        # Put all the co-occurrence counts into a big matrix
        log.info("Building counts matrix: vocab size %d" % len(dictionary))
        vectors = numpy.zeros((len(dictionary), len(dictionary)), dtype=numpy.float64)
        # Fill the matrix with raw counts
        for (event0, event1), count in pair_counts.items():
            if event0 in dictionary.token2id and event1 in dictionary.token2id:
                e0, e1 = dictionary.token2id[event0], dictionary.token2id[event1]
                vectors[e0, e1] = count
                # Add the count both ways (it's only stored once above)
                vectors[e1, e0] = count

        # Now there are many things we could do to these counts
        if opts.pmi or opts.ppmi:
            log.info("Applying %sPMI" % "P" if opts.ppmi else "")
            # Apply PMI to the matrix
            # Compute the total counts for each event (note row and col totals are the same)
            log_totals = numpy.ma.log(vectors.sum(axis=0))
            vectors = numpy.ma.log(vectors * vectors.sum()) - log_totals
            vectors = (vectors.T - log_totals).T
            vectors = vectors.filled(0.)

            if opts.ppmi:
                # Threshold the PMIs at zero
                vectors[vectors < 0.] = 0.

        # Convert to sparse for SVD and storage
        vectors = csr_matrix(vectors)

        if opts.svd:
            log.info("Fitting SVD with %d dimensions" % opts.svd)
            training_metadata["svd from"] = vectors.shape[1]
            training_metadata["svd"] = opts.svd
            vector_svd = TruncatedSVD(opts.svd)
            vectors = vector_svd.fit_transform(vectors)

        log.info("Saving model: %s" % model_name)
        model = DistributionalVectorsNarrativeChainModel(dictionary, vectors, only_verb=opts.only_verb,
                                                  training_metadata=training_metadata, adjectives=opts.adj)
        model.save(model_name)
        return model
