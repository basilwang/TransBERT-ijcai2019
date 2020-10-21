"""
A slimline version of the event-comp model that can be stored and loaded faster than the main model.

This may be system-specific, so is generally stored in a different location from the main model.
It may also not be robust to changes in the code, so may need to be regenerated from the main model if
something's changed.

"""
import os
import cPickle as pickle
import shutil
import numpy
from cam.whim.entity_narrative.models.base.coherence import CoherenceScorer
from cam.whim.entity_narrative.models.event_comp.model import EventCompositionNarrativeChainModel
from cam.whim.entity_narrative.models.event_comp.sample import NextEventProjectionSampler
from whim_common.utils.local import LOCAL_DIR
from whim_common.utils.progress import get_progress_bar


class SlimlineEventCompositionNarrativeChainModel(EventCompositionNarrativeChainModel, CoherenceScorer):
    def __init__(self, pair_coherence_fn, event_projection_fn, vector_size,
                 pred_vocab, pred_id2word, arg_vocab, arg_id2word,
                 arg_comp_layers, event_comp_layers,
                 predicative_adjectives=False, transitivity=False):
        self.event_comp_layers = event_comp_layers
        self.arg_comp_layers = arg_comp_layers
        self._vector_size = vector_size

        self.transitivity = transitivity
        self.predicative_adjectives = predicative_adjectives
        self.arg_vocab = arg_vocab
        self.arg_id2word = arg_id2word
        self.pred_vocab = pred_vocab
        self.pred_id2word = pred_id2word

        self.pair_coherence_fn = pair_coherence_fn
        self.event_projection_fn = event_projection_fn

    @property
    def vector_size(self):
        return self._vector_size

    def chain_coherence(self, entity, events):
        """
        Rate the coherence of a chain by computing the average pairwise coherence between all the
        events in the chain.

        """
        if len(events) < 2:
            return None
        pair_input_indices = list(self.get_pair_input_vectors([(entity, (events[i], events[j]))
                                                               for i in range(len(events)-1)
                                                               for j in range(i+1, len(events))], none_unknowns=True))

        known_pair_input_indices = [val for val in pair_input_indices if val is not None]
        # Remove pairs where both inputs are identical: they shouldn't contribute to overall coherence
        known_pair_input_indices = [inputs for inputs in known_pair_input_indices if inputs[:4] != inputs[4:]]

        if len(known_pair_input_indices) == 0:
            return None

        # Now compose the vectors using the learned prediction function, giving a score for each pair
        input_arrays = [numpy.array(pos_indices, dtype=numpy.int32) for pos_indices in zip(*known_pair_input_indices)]
        known_coherences = self.pair_coherence_fn(*input_arrays)

        return numpy.mean(known_coherences)

    def chain_coherences(self, chains, batch_size=1000):
        """
        Batched version for faster scoring of large datasets.

        """
        pair_batch = []
        batch_data = []
        num_pairs = []
        for (entity, events), data in chains:
            # Add all the pairs for this chain
            chain_pairs = [(entity, (events[i], events[j]))
                           for i in range(len(events)-1)
                           for j in range(i+1, len(events))]
            # Note how many pairs we're evaluating from this chain
            num_pairs.append(len(chain_pairs))
            # Add all the pairs to the batch
            pair_batch.extend(chain_pairs)
            batch_data.append(data)

            # Check whether we've filled a batch
            # Note that the batches might go slightly over the batch size
            if len(pair_batch) >= batch_size:
                coherences = self.get_pair_coherences(pair_batch, remove_identical=True)
                taken = 0
                for chain_num_pairs, chain_data in zip(num_pairs, batch_data):
                    # Get all the coherences for this chain
                    # Exclude any nans, from e.g. unknown predicates
                    chain_coherences = coherences[taken:taken+chain_num_pairs]
                    chain_coherences = chain_coherences[~numpy.isnan(chain_coherences)]
                    if len(chain_coherences) == 0:
                        # No valid coherences for this chain: can't report a figure
                        yield None, chain_data
                    else:
                        yield numpy.mean(chain_coherences), chain_data
                    taken += chain_num_pairs

                pair_batch = []
                num_pairs = []
                batch_data = []

        # Finish up the final batch
        if len(pair_batch):
            coherences = self.get_pair_coherences(pair_batch, remove_identical=True)
            taken = 0
            for chain_num_pairs, chain_data in zip(num_pairs, batch_data):
                chain_coherences = coherences[taken:taken+chain_num_pairs]
                chain_coherences = chain_coherences[~numpy.isnan(chain_coherences)]
                if len(chain_coherences) == 0:
                    yield None, chain_data
                else:
                    yield numpy.mean(chain_coherences), chain_data
                taken += chain_num_pairs

    def _score_choices(self, entities, contexts, choice_lists, progress=False):
        pbar = None
        if progress:
            pbar = get_progress_bar(len(entities), title="Scoring choices")

        scores = numpy.zeros((len(entities), max(len(choices) for choices in choice_lists)), dtype=numpy.float32)

        for test_num, (entity, context_events, choices) in enumerate(zip(entities, contexts, choice_lists)):
            context_size = len(context_events)
            # Project each pair of context event and candidate to get a coherence score
            pairs = [
                (entity, [context_event, choice])
                for choice in choices for context_event in context_events
                ]
            coherences = list(self.get_pair_coherences(pairs, unknown_value=0.))
            # Mean the coherences across the context events for each candidate
            choice_scores = [sum(coherences[choice_num*context_size:(choice_num+1)*context_size], 0.) / context_size
                             for choice_num in range(len(choices))]
            scores[test_num, :] = choice_scores

            if pbar:
                pbar.update(test_num)

        if pbar:
            pbar.finish()

        return scores

    def score_choices_by_id(self, entity, context, (pred_choices, arg0_choices, arg1_choices, arg2_choices)):
        """
        Like _score_choices, but for when you already have the model ids of the candidates. Also only
        compares to a single context.

        """
        pred_choices = numpy.array(pred_choices, dtype=numpy.int32)
        arg0_choices = numpy.array(arg0_choices, dtype=numpy.int32)
        arg1_choices = numpy.array(arg1_choices, dtype=numpy.int32)
        arg2_choices = numpy.array(arg2_choices, dtype=numpy.int32)

        # Get the context IDs
        context_input_indices = [numpy.array(l, dtype=numpy.int32) for l in
                                 zip(*list(self.get_chain_input_vectors([(entity, context)]))[0])]
        context_size = context_input_indices[0].shape[0]
        # Repeat the context events to match up with the candidates
        context_input_indices = [numpy.tile(a, pred_choices.shape[0]) for a in context_input_indices]

        # Repeat the input IDs to match up with each context event
        candidate_ids = [numpy.repeat(pred_choices, context_size),
                         numpy.repeat(arg0_choices, context_size),
                         numpy.repeat(arg1_choices, context_size),
                         numpy.repeat(arg2_choices, context_size)]

        # Project each pair of context event and candidate to get a coherence score
        input_ids = context_input_indices + candidate_ids
        coherences = self.pair_coherence_fn(*input_ids)
        # Reshaping puts all the coherences from the same candidate on a row
        coherences = coherences.reshape((pred_choices.shape[0], context_size)).mean(axis=1)

        return coherences

    def get_pair_coherences(self, pairs, unknown_value=float("nan"), remove_identical=False):
        """

        :param pairs:
        :param unknown_value: controls how unknown predicates are handled. By default, pairs including an
         unknown predicate are given NaN coherence so they're distinguishable from incoherence pairs.
         You might want to set this to a float (like 0.). The special string value "skip" causes these values
         to be left out. Note that this means the number of coherences might not be equal to the input pairs.
        :return: 1D numpy array
        """
        pair_input_indices = list(self.get_pair_input_vectors(pairs, none_unknowns=True))
        # Keep a note of where unknown inputs were
        known_indices = [i for (i, val) in enumerate(pair_input_indices) if val is not None]
        if remove_identical:
            # Treat as unknown any pairs whose inputs are identical
            known_indices = [i for i in known_indices if pair_input_indices[i][:4] != pair_input_indices[i][4:]]

        coherences = numpy.zeros(len(pairs), dtype=numpy.float64)
        # Set everything to NaNs so unknown inputs are signalled as such
        if unknown_value != "skip":
            coherences[:] = unknown_value

        if len(known_indices) == 0:
            if unknown_value == "skip":
                return numpy.array([], dtype=numpy.float64)
            else:
                return coherences

        known_pair_input_indices = [pair_input_indices[i] for i in known_indices]
        # Now compose the vectors using the learned prediction function, giving a score for each pair
        input_arrays = [numpy.array(pos_indices, dtype=numpy.int32) for pos_indices in zip(*known_pair_input_indices)]
        known_coherences = self.pair_coherence_fn(*input_arrays)
        if unknown_value == "skip":
            # Only return the known values
            return known_coherences
        else:
            coherences[known_indices] = known_coherences
            return coherences

    def project_events(self, events, progress=False):
        """
        Provide an interface function to get the projection of individual events to the deepest
        layer before composition (i.e. the deepest layer of the base arg composition network).

        Input events should be (entity, event) pairs.

        """
        # Prepare input vectors for each of the events
        event_indices = [chain_tuples[0] if len(chain_tuples) else None for chain_tuples in
                         self.get_chain_input_vectors([(entity, [event]) for (entity, event) in events])]
        known_indices = [i for (i, inputs) in enumerate(event_indices) if inputs is not None]
        event_indices_inputs = [tuples for tuples in event_indices if tuples is not None]
        # Prepare a big array to put the non-zero projections in
        result = numpy.zeros((len(events), self.vector_size), dtype=numpy.float64)
        # Now project the vectors using the learned projection function
        projection = self.event_projection_fn(
            *[numpy.array(pos_indices, dtype=numpy.int32) for pos_indices in zip(*event_indices_inputs)]
        )
        result[known_indices] = projection
        return result

    @property
    def _description(self):
        return """\
Neural composition event composition coherence model
Event argument composition: %d layers (%d->%s)
Event pair composition:     %d layers ([%d|%d]->%s->1)""" % (
            len(self.arg_comp_layers),
            len(self.pred_vocab) + 3*len(self.arg_vocab),
            "->".join(str(s) for s in self.arg_comp_layers),
            len(self.event_comp_layers), self.vector_size, self.vector_size,
            "->".join(str(s) for s in self.event_comp_layers)
        )

    @staticmethod
    def from_main_model(model):
        """
        Create a slimline version of an existing model
        """
        return SlimlineEventCompositionNarrativeChainModel(
            model.pair_projection_model.coherence_fn,
            model.pair_projection_model.event_network.project,
            model.pair_projection_model.event_network.projection_size,
            model.pred_vocab, model.pred_id2word, model.arg_vocab, model.arg_id2word,
            model.pair_projection_model.event_network.layer_sizes, model.pair_projection_model.layer_sizes,
            model.predicative_adjectives, model.transitivity,
        )

    def save(self, model_name):
        """
        Save the local version of the model that can be loaded quickly.

        """
        directory = os.path.join(LOCAL_DIR, "model_cache", self.MODEL_TYPE_NAME, model_name)
        # Make sure the directory exists and is empty
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        with open(os.path.join(directory, "model"), "w") as f:
            pickle.dump(self, f, -1)

    @staticmethod
    def load(model_name):
        directory = os.path.join(LOCAL_DIR, "model_cache", EventCompositionNarrativeChainModel.MODEL_TYPE_NAME,
                                 model_name)

        if not os.path.exists(directory):
            raise IOError("cached slimline version of model event-comp/%s does not exist" % model_name)

        with open(os.path.join(directory, "model"), "r") as f:
            return pickle.load(f)

    @staticmethod
    def load_or_create(model_name):
        """
        Load a cached slimline model if one exists. Otherwise, load the main model and cache it as a slimline
        model.

        """
        created = False
        try:
            model = SlimlineEventCompositionNarrativeChainModel.load(model_name)
        except IOError:
            # Model has not been cached
            # Do it now
            created = True
            main_model = EventCompositionNarrativeChainModel.load(model_name)
            model = SlimlineEventCompositionNarrativeChainModel.from_main_model(main_model)
            # Save the slimline model for next time
            model.save(model_name)

        return model, created

    @staticmethod
    def load_projection_sampler(model_name):
        filename = os.path.join(LOCAL_DIR, "model_cache", EventCompositionNarrativeChainModel.MODEL_TYPE_NAME,
                                model_name, "projection_sampler")

        if not os.path.exists(filename):
            raise IOError("cached projection sampler for model event-comp/%s does not exist" % model_name)

        with open(filename, "r") as f:
            return pickle.load(f)

    @staticmethod
    def save_projection_sampler(model_name, sampler):
        filename = os.path.join(LOCAL_DIR, "model_cache", EventCompositionNarrativeChainModel.MODEL_TYPE_NAME,
                                model_name, "projection_sampler")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, "w") as f:
            pickle.dump(sampler, f, -1)

    @staticmethod
    def create_projection_sampler(model_name, neighbour_finder=None, learning_rate=0.1, num_samples=1):
        # Need to load full model to create the sampler
        main_model = EventCompositionNarrativeChainModel.load(model_name)
        # Also get the slimline model
        model = SlimlineEventCompositionNarrativeChainModel.from_main_model(main_model)
        sampler = NextEventProjectionSampler.from_model(main_model, neighbour_finder=neighbour_finder,
                                                        learning_rate=learning_rate, num_samples=num_samples,
                                                        slimline_model=model)
        # Save for next time
        SlimlineEventCompositionNarrativeChainModel.save_projection_sampler(model_name, sampler)
        return sampler

    @staticmethod
    def load_or_create_projection_sampler(model_name, neighbour_finder=None, learning_rate=0.1, num_samples=1):
        created = False

        try:
            sampler = SlimlineEventCompositionNarrativeChainModel.load_projection_sampler(model_name)
        except IOError:
            # No stored sampler
            created = True
            sampler = SlimlineEventCompositionNarrativeChainModel.create_projection_sampler(model_name,
                                                                                            neighbour_finder,
                                                                                            learning_rate,
                                                                                            num_samples)

        return sampler, created

    def _load_from_directory(cls, directory, **kwargs):
        # Make sure this never gets called
        raise NotImplementedError("you cannot load a slimline version of the model using the normal model save routine")

    def _save_to_directory(self, directory, model_name, human_name=None):
        # Make sure this never gets called
        raise NotImplementedError("you cannot save a slimline version of the model using the normal model save routine")
