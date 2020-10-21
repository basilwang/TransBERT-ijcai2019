from collections import Counter

from cam.whim.entity_narrative.models.base.train import NarrativeChainModelTrainer
from whim_common.utils.progress import get_progress_bar


class CandjTrainer(NarrativeChainModelTrainer):
    def prepare_arguments(self, parser):
        """ Add arguments to an argparse parser for training. """
        parser.add_argument("--pair-threshold", type=int,
                            help="Apply a threshold number of counts to pairs")
        parser.add_argument("--event-threshold", type=int,
                            help="Apply a threshold number of counts to events")

    def train(self, model_name, corpus, log, opts, chain_features=None):
        from cam.whim.entity_narrative.models.candj.model import CandjNarrativeChainModel
        log.info("Training C&J model")

        log.info("Extracting event counts")
        pbar = get_progress_bar(len(corpus), title="Event feature extraction")
        # Loop over all the chains again to collect events and pairs of events
        event_counts = Counter()
        pair_counts = Counter()
        for doc_num, document in enumerate(corpus):
            chains = document.get_chains()
            if len(chains):
                event_chains = list(CandjNarrativeChainModel.extract_chain_feature_dicts(chains))
                # Count all the events
                for chain in event_chains:
                    event_counts.update(chain)
                    # Also count all pairs
                    pairs = []
                    for i in range(len(chain)-1):
                        for j in range(i+1, len(chain)):
                            pairs.append(tuple(sorted([chain[i], chain[j]])))
                    pair_counts.update(pairs)

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
            # Also remove any pairs involving these events
            pairs_to_remove = [(event0, event1) for ((event0, event1), count) in pair_counts.items() if
                               event0 in to_remove or event1 in to_remove]
            pbar = get_progress_bar(len(pairs_to_remove), title="Filtering pair counts")
            for i, pair in enumerate(pairs_to_remove):
                del pair_counts[pair]
                pbar.update(i)
            pbar.finish()
        if opts.pair_threshold is not None and opts.pair_threshold > 0:
            log.info("Apply pair threshold")
            # Apply a threshold pair count
            to_remove = [pair for (pair, count) in pair_counts.items() if count < opts.pair_threshold]
            pbar = get_progress_bar(len(to_remove), title="Filtering pair counts")
            for i, pair in enumerate(to_remove):
                del pair_counts[pair]
                pbar.update(i)
            pbar.finish()

        log.info("Saving model: %s" % model_name)
        model = CandjNarrativeChainModel(event_counts, pair_counts)
        model.save(model_name)
        return model