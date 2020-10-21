import os
import cPickle as pickle

import numpy
import theano
import theano.tensor as T

from cam.whim.entity_narrative.models.arg_comp.model import ArgumentCompositionNarrativeChainModel, \
    build_id2word
from cam.whim.entity_narrative.models.event_comp.shell import EventCompositionShell
from cam.whim.entity_narrative.models.base.coherence import CoherenceScorer
from cam.whim.entity_narrative.models.base.model import NarrativeChainModel
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel
from whim_common.utils.progress import get_progress_bar
from whim_common.utils.theano.nn.autoencoder import DenoisingAutoencoder
from .train import EventCompositionTrainer


class EventCompositionNarrativeChainModel(NarrativeChainModel, CoherenceScorer):
    """
    Neural network that performs a composition of the predicate and arguments of events and then
    composes a pair of events to assign them a score reflecting how likely they are to appear in
    the same chain.

    To train, first (pre-)train an arg-comp model. This is then used to initialize the arg-comp part
    of this network and the event composition (coherence scoring) function is learned as a function
    of the event projections.

    The event projections themselves are also updated during training. This does not update the
    arg-comp model that was loaded as initialization -- it's effectively copied into this model.

    """
    MODEL_TYPE_NAME = "event-comp"
    TRAINER_CLASS = EventCompositionTrainer
    SHELL_TYPE = EventCompositionShell
    MODEL_OPTIONS = dict(VectorSpaceNarrativeChainModel.MODEL_OPTIONS, **{})

    def __init__(self, pair_projection_model, pred_vocab, arg_vocab, predicative_adjectives=False, transitivity=False,
                 **kwargs):
        super(EventCompositionNarrativeChainModel, self).__init__(**kwargs)
        self.pair_projection_model = pair_projection_model
        self.transitivity = transitivity
        self.predicative_adjectives = predicative_adjectives
        self.arg_vocab = arg_vocab
        self.arg_id2word = build_id2word(arg_vocab)
        self.pred_vocab = pred_vocab
        self.pred_id2word = build_id2word(pred_vocab)

    @property
    def vector_size(self):
        return self.pair_projection_model.event_network.projection_size

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
        known_coherences = self.pair_projection_model.coherence_fn(*input_arrays)

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
        coherences = self.pair_projection_model.coherence_fn(*input_ids)
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
        known_coherences = self.pair_projection_model.coherence_fn(*input_arrays)
        if unknown_value == "skip":
            # Only return the known values
            return known_coherences
        else:
            coherences[known_indices] = known_coherences
            return coherences

    def get_pair_input_vectors(self, pairs, none_unknowns=False):
        for pair_num, pair_features in \
                enumerate(ArgumentCompositionNarrativeChainModel.extract_chain_word_lists(
                    pairs, predicative_adjectives=self.predicative_adjectives, transitivity=self.transitivity)):
            if pair_features[0][0] in self.pred_vocab and pair_features[1][0] in self.pred_vocab:
                # pair_features should look like:
                #  [(pred_a, arg0_a, arg1_a, arg2_a), (pred_b, arg0_b, arg1_b, arg2_b)]
                yield (
                    self.pred_vocab[pair_features[0][0]].index,
                    self.arg_vocab[pair_features[0][1]].index if pair_features[0][1] in self.arg_vocab else -1,
                    self.arg_vocab[pair_features[0][2]].index if pair_features[0][2] in self.arg_vocab else -1,
                    self.arg_vocab[pair_features[0][3]].index if pair_features[0][3] in self.arg_vocab else -1,
                    self.pred_vocab[pair_features[1][0]].index,
                    self.arg_vocab[pair_features[1][1]].index if pair_features[1][1] in self.arg_vocab else -1,
                    self.arg_vocab[pair_features[1][2]].index if pair_features[1][2] in self.arg_vocab else -1,
                    self.arg_vocab[pair_features[1][3]].index if pair_features[1][3] in self.arg_vocab else -1,
                )
            elif none_unknowns:
                # Yield a None where the predicate is unknown, instead of just skipping
                yield None

    def get_chain_input_vectors(self, chains):
        for chain_num, chain_features in \
                enumerate(ArgumentCompositionNarrativeChainModel.extract_chain_word_lists(
                    chains, predicative_adjectives=self.predicative_adjectives, transitivity=self.transitivity)):
            yield [
                (
                    self.pred_vocab[pred].index,
                    self.arg_vocab[arg0].index if arg0 in self.arg_vocab else -1,
                    self.arg_vocab[arg1].index if arg1 in self.arg_vocab else -1,
                    self.arg_vocab[arg2].index if arg2 in self.arg_vocab else -1
                ) for (pred, arg0, arg1, arg2) in chain_features if pred in self.pred_vocab
            ]

    def get_event_input_indices(self, entity, event):
        pred, arg0, arg1, arg2 = list(ArgumentCompositionNarrativeChainModel.extract_chain_word_lists(
            [(entity, [event])], predicative_adjectives=self.predicative_adjectives, transitivity=self.transitivity)
        )[0][0]

        if pred not in self.pred_vocab:
            return None
        else:
            return (
                self.pred_vocab[pred].index,
                self.arg_vocab[arg0].index if arg0 in self.arg_vocab else -1,
                self.arg_vocab[arg1].index if arg1 in self.arg_vocab else -1,
                self.arg_vocab[arg2].index if arg2 in self.arg_vocab else -1
            )

    def get_model_predicate_repr(self, entity, event):
        return ArgumentCompositionNarrativeChainModel.get_predicate_repr(
            entity, event,
            predicative_adjectives=self.predicative_adjectives,
            transitivity=self.transitivity,
        )

    def filter_equivalents(self, base_event, chain):
        """
        Return the chain filtered to remove events that, from the point of view of the model,
        are equivalent.

        :param base_event: (entity, event) pair
        :param event_list: (entity, event_list) pair
        """
        # Get the model's feature list representation of the base event and the chain events
        reprs = list(ArgumentCompositionNarrativeChainModel.extract_chain_word_lists(
            [
                (base_event[0], [base_event[1]]),
                chain
            ], predicative_adjectives=self.predicative_adjectives, transitivity=self.transitivity
        ))
        base_event_repr = reprs[0][0]
        chain_reprs = reprs[1]
        # Return a filtered version of the input chain
        return chain[0], [chain[1][idx] for idx, repr in enumerate(chain_reprs) if repr != base_event_repr]

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
        result = numpy.zeros((len(events), self.pair_projection_model.event_network.projection_size),
                             dtype=numpy.float64)
        # Now project the vectors using the learned projection function
        projection = self.pair_projection_model.event_network.project(
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
            len(self.pair_projection_model.event_network.layers),
            len(self.pred_vocab) + 3*len(self.arg_vocab),
            "->".join(str(s) for s in self.pair_projection_model.event_network.layer_sizes),
            len(self.pair_projection_model.layers),
            self.pair_projection_model.event_network.projection_size,
            self.pair_projection_model.event_network.projection_size,
            "->".join(str(s) for s in self.pair_projection_model.layer_sizes)
        )

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        # Module and model type were renamed
        # Alias old names so that old models can be loaded without pickle running into problems below
        import sys

        sys.modules["cam.whim.entity_narrative.models.autoencoder_pair_comp"] = \
            sys.modules["cam.whim.entity_narrative.models.event_comp"]
        sys.modules["cam.whim.entity_narrative.models.autoencoder_comp"] = \
            sys.modules["cam.whim.entity_narrative.models.arg_comp"]

        with open(os.path.join(directory, "pred_vocab"), "r") as f:
            pred_vocab = pickle.load(f)
        with open(os.path.join(directory, "arg_vocab"), "r") as f:
            arg_vocab = pickle.load(f)
        with open(os.path.join(directory, "params"), "r") as f:
            params = pickle.load(f)
        # Also need the pair projection model, which include the event projection model
        with open(os.path.join(directory, "pair_projection_model"), "r") as f:
            pair_projection_model = pickle.load(f)

        return EventCompositionNarrativeChainModel(
            pair_projection_model,
            pred_vocab, arg_vocab,
            predicative_adjectives=params["predicative_adjectives"],
            transitivity=params["transitivity"],
            **kwargs
        )

    def _save_to_directory(self, directory, model_name, human_name=None):
        with open(os.path.join(directory, "pred_vocab"), "w") as f:
            pickle.dump(self.pred_vocab, f, -1)
        with open(os.path.join(directory, "arg_vocab"), "w") as f:
            pickle.dump(self.arg_vocab, f, -1)
        with open(os.path.join(directory, "params"), "w") as f:
            pickle.dump({
                "predicative_adjectives": self.predicative_adjectives,
                "transitivity": self.transitivity,
            }, f, -1)
        with open(os.path.join(directory, "pair_projection_model"), "w") as f:
            pickle.dump(self.pair_projection_model, f, -1)

    def get_nearest_neighbour_finder(self, model_name, port):
        if self._nearest_neighbour_finder is None:
            from cam.whim.entity_narrative.models.base.vectorspace.neighbours import NearestNeighbourFinder
            # Cache once loaded
            self._nearest_neighbour_finder = NearestNeighbourFinder.load(self.MODEL_TYPE_NAME, model_name,
                                                                         redis_port=port)
        return self._nearest_neighbour_finder


class PairCompositionNetwork(object):
    """
    NN composition of pairs of events to predict a score for whether they're in the same chain.

    """
    def __init__(self, event_network, layer_sizes=[100]):
        self.event_network = event_network
        self.input_a, self.input_b = self.event_network.get_projection_pair()
        self.input_vector = T.concatenate((self.input_a, self.input_b), axis=1)
        self.layer_sizes = layer_sizes

        # Initialize each layer as an autoencoder, allowing us to initialize it by pretraining
        self.layers = []
        self.layer_outputs = []
        input_size = self.event_network.layer_sizes[-1] * 2
        layer_input = self.input_vector
        for layer_size in layer_sizes:
            self.layers.append(
                DenoisingAutoencoder(layer_input, input_size, layer_size, non_linearity="tanh")
            )
            layer_input = self.layers[-1].hidden_layer
            self.layer_outputs.append(layer_input)
            input_size = layer_size
        self.final_projection = layer_input

        # Add a final layer, which will only ever be trained with a supervised objective
        # This is simply a logistic regression layer to predict a coherence score for the input pair
        self.prediction_weights = theano.shared(
            # Just initialize to zeros, so we start off predicting 0.5 for every input
            numpy.asarray(
                numpy.random.uniform(
                    low=2. * -numpy.sqrt(6. / (layer_sizes[-1] + 1)),
                    high=2. * numpy.sqrt(6. / (layer_sizes[-1] + 1)),
                    size=(layer_sizes[-1], 1),
                ),
                dtype=theano.config.floatX
            ),
            name="prediction_w",
            borrow=True
        )
        self.prediction_bias = theano.shared(
            value=numpy.zeros(1, dtype=theano.config.floatX),
            name="prediction_b",
            borrow=True
        )
        self.prediction = T.nnet.sigmoid(
            T.dot(self.final_projection, self.prediction_weights) + self.prediction_bias
        )

        self.pair_inputs = [
            self.event_network.predicate_input_a, self.event_network.arg0_input_a,
            self.event_network.arg1_input_a, self.event_network.arg2_input_a,
            self.event_network.predicate_input_b, self.event_network.arg0_input_b,
            self.event_network.arg1_input_b, self.event_network.arg2_input_b
        ]
        self.triple_inputs = [
            self.event_network.predicate_input_a, self.event_network.arg0_input_a,
            self.event_network.arg1_input_a, self.event_network.arg2_input_a,
            self.event_network.predicate_input_b, self.event_network.arg0_input_b,
            self.event_network.arg1_input_b, self.event_network.arg2_input_b,
            self.event_network.predicate_input_c, self.event_network.arg0_input_c,
            self.event_network.arg1_input_c, self.event_network.arg2_input_c
        ]

        self._coherence_fn = None

    @property
    def coherence_fn(self):
        if self._coherence_fn is None:
            self._coherence_fn = theano.function(
                inputs=self.pair_inputs,
                outputs=self.prediction,
                name="pair_coherence",
            )
        return self._coherence_fn

    def get_coherence_pair(self):
        # Clone prediction function so we can perform two predictions in the same step
        coherence_a = self.prediction

        # Replace b inputs with c inputs
        input_replacements = dict(zip(self.triple_inputs[4:8], self.triple_inputs[8:12]))
        coherence_b = theano.clone(self.prediction, replace=input_replacements)

        return coherence_a, coherence_b

    def get_layer_input_function(self, layer_num):
        if layer_num >= len(self.layers):
            raise ValueError("cannot get input function for layer %d in a %d-layer network" %
                             (layer_num, len(self.layers)))
        elif layer_num == 0:
            # The input to the first layer is just the concatenated input vectors
            output_eq = self.input_vector
        else:
            # Otherwise it's the output from the previous layer
            output_eq = self.layer_outputs[layer_num-1]

        return theano.function(
            inputs=self.pair_inputs,
            outputs=output_eq,
            name="layer-%d-input" % layer_num,
        )

    def copy_coherence_function(self, input_a=None, input_b=None):
        """
        Build a new coherence function, copying all weights and such from this network, replacing
        components given as kwargs. Note that this uses the same shared variables and any other
        non-replaced components as the network's original expression graph: bear in mind if you use
        it to update weights or combine with other graphs.

        """
        input_a = input_a or self.input_a
        input_b = input_b or self.input_b

        # Build a new coherence function, combining these two projections
        input_vector = T.concatenate([input_a, input_b], axis=input_a.ndim-1)

        # Initialize each layer as an autoencoder. We'll then set its weights and never use it as an autoencoder
        layers = []
        layer_outputs = []
        input_size = self.event_network.layer_sizes[-1] * 2
        layer_input = input_vector
        for layer_size in self.layer_sizes:
            layers.append(
                DenoisingAutoencoder(layer_input, input_size, layer_size, non_linearity="tanh")
            )
            layer_input = layers[-1].hidden_layer
            layer_outputs.append(layer_input)
            input_size = layer_size
        final_projection = layer_input

        # Set the weights of all the layers to the ones trained in the base network
        for layer, layer_weights in zip(layers, self.get_weights()):
            layer.set_weights(layer_weights)

        # Add a final layer
        # This is simply a logistic regression layer to predict a coherence score for the input pair
        activation = T.dot(final_projection, self.prediction_weights) + self.prediction_bias
        # Remove the last dimension, which should now just be of size 1
        activation = activation.reshape(activation.shape[:-1], ndim=activation.ndim-1)
        prediction = T.nnet.sigmoid(activation)

        return prediction, input_vector, layers, layer_outputs, activation

    def get_weights(self):
        return [ae.get_weights() for ae in self.layers] + \
               [self.prediction_weights.get_value(),
                self.prediction_bias.get_value()]

    def set_weights(self, weights):
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)
        self.prediction_weights.set_value(weights[len(self.layers)])
        self.prediction_bias.set_value(weights[len(self.layers) + 1])

    def __getstate__(self):
        return {
            "weights": self.get_weights(),
            "layer_sizes": self.layer_sizes,
            "event_network": self.event_network,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(state["event_network"], layer_sizes=state["layer_sizes"])
        self.set_weights(state["weights"])


# Alias for backwards compatibility (unpickling, in particular)
AutoencoderPairCompositionNarrativeChainModel = EventCompositionNarrativeChainModel
