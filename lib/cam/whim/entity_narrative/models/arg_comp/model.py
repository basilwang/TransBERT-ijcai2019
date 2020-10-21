import os
import pickle

import numpy
import theano
import theano.tensor as T

from whim_common.data.coref import Entity
from cam.whim.entity_narrative.chains.document import predicate_relation, predicate_relation_with_transitivity
from cam.whim.entity_narrative.models.arg_comp.shell import ArgumentCompositionShell
from cam.whim.entity_narrative.models.base.vectorspace.model import VectorSpaceNarrativeChainModel
from whim_common.utils.theano.nn.autoencoder import DenoisingAutoencoder
from .train import ArgumentCompositionTrainer


class ArgumentCompositionNarrativeChainModel(VectorSpaceNarrativeChainModel):
    """
    Neural network composition of event predicates and arguments into a single vector.

    This constitutes the argument composition component of the NN coherence model, referred to as Event-comp
    in the paper. That model is trained by first training one of these in some form to prepare the arg-comp
    network and then building an event composition network on top of it (and updating the learned weights in
    the arg comp network as well).

    If you use this model on its own, it just uses vector similarity of the event projections. This doesn't
    work brilliantly -- event-comp is better.

    """
    MODEL_TYPE_NAME = "arg-comp"
    TRAINER_CLASS = ArgumentCompositionTrainer
    DEFAULT_METRIC = "dot"
    SHELL_TYPE = ArgumentCompositionShell
    MODEL_OPTIONS = dict(VectorSpaceNarrativeChainModel.MODEL_OPTIONS, **{
        "layer": {
            "type": int,
            "help": "Don't project all the way through to the deepest layer, but use projections from an intermediate "
                    "layer. By default, uses deepest layer. (NB: can't be changed after model is loaded)"
        }
    })

    def __init__(self, projection_model, pred_vocab, arg_vocab, predicative_adjectives=False, transitivity=False,
                 arg2_vocab=None, preposition_vocab=None, **kwargs):
        super(ArgumentCompositionNarrativeChainModel, self).__init__(**kwargs)
        self.transitivity = transitivity
        self.predicative_adjectives = predicative_adjectives
        self.arg_vocab = arg_vocab
        self.arg_id2word = build_id2word(arg_vocab)
        self.pred_vocab = pred_vocab
        self.pred_id2word = build_id2word(pred_vocab)
        self.projection_model = projection_model

        if arg2_vocab is not None:
            # This is only set if we're using a separate vocab for arg2, so that we can include prepositions
            self.arg2_vocab = arg2_vocab
            self.arg2_id2word = build_id2word(arg2_vocab)
            self._prepositions = preposition_vocab
        else:
            # Use the same vocab for arg2 as the other args
            self.arg2_vocab = self.arg_vocab
            self.arg2_id2word = self.arg_id2word
        self.include_prepositions = preposition_vocab is not None

        if self.model_options["layer"] is not None:
            # We've been asked to use a specific layer's projection, not the full projection
            self._project = self.projection_model.get_layer_projection(self.model_options["layer"])
        else:
            self._project = self.projection_model.project

    @property
    def vector_size(self):
        if self.model_options["layer"] is not None:
            if self.model_options["layer"] == -1:
                return self.projection_model.input_size
            else:
                return self.projection_model.layer_sizes[self.model_options["layer"]]
        else:
            return self.projection_model.projection_size

    def project_chains(self, chains, progress=False):
        vectors = numpy.zeros((len(chains), self.vector_size), dtype=numpy.float64)
        for chain_num, chain_word_vecs in enumerate(self.get_chain_event_vectors(chains)):
            # Just sum up the vectors to get the chain repr
            vectors[chain_num] = numpy.sum(chain_word_vecs, axis=0)
        return vectors

    def get_chain_event_vectors(self, chains):
        for chain_input_indices in self.get_chain_input_vectors(chains):
            if len(chain_input_indices):
                # Now compose the vectors using the learned projection function
                yield self._project(*[numpy.array(pos_indices, dtype=numpy.int32)
                                      for pos_indices in zip(*chain_input_indices)])

    def get_chain_input_vectors(self, chains):
        for chain_num, chain_features in \
                enumerate(ArgumentCompositionNarrativeChainModel.extract_chain_word_lists(
                    chains, predicative_adjectives=self.predicative_adjectives, transitivity=self.transitivity,
                    include_prepositions=self.include_prepositions,
                )):
            yield [
                (
                    self.pred_vocab[pred].index,
                    self.arg_vocab[arg0].index if arg0 in self.arg_vocab else -1,
                    self.arg_vocab[arg1].index if arg1 in self.arg_vocab else -1,
                    self.arg2_vocab[arg2].index if arg2 in self.arg2_vocab else -1
                ) for (pred, arg0, arg1, arg2) in chain_features if pred in self.pred_vocab
            ]

    @staticmethod
    def extract_chain_word_lists(chains, predicative_adjectives=False, transitivity=False, include_prepositions=False):
        """
        Chains should be given as a list of (entity, event list) pairs.
        Produces a list of string representations of the events of each chain.

        """
        for entity, chain in chains:
            yield [
                [ArgumentCompositionNarrativeChainModel.get_predicate_repr(
                    entity, ev,
                    predicative_adjectives=predicative_adjectives,
                    transitivity=transitivity)] +
                ArgumentCompositionNarrativeChainModel.extract_chain_arg_list(ev, protagonist=entity,
                                                                              include_prepositions=include_prepositions)
                for ev in chain
            ]

    @staticmethod
    def extract_chain_arg_list(event, protagonist=None, include_prepositions=False):
        features = []
        # A word for each argument, if it's filled
        for arg_num, arg in enumerate([event.subject, event.object,
                                       event.iobject[1] if event.iobject is not None else None]):
            if arg is None:
                # Arg is not filled: use None to indicate this
                features.append(None)
            elif type(arg) is Entity:
                if arg is protagonist:
                    # The chain's protagonist: don't include a word for them
                    features.append(None)
                else:
                    # Try getting a headword for the entity
                    arg_word = arg.get_head_word()
                    if arg_word is None:
                        # Treat as if there was no entity: we have no other way to get a representation
                        features.append(None)
                    else:
                        features.append(arg_word)
            else:
                # Otherwise it must be a string -- the headword of an NP
                features.append(str(arg).lower())
        # If iobject was filled and we're including prepositions, put them together here
        if include_prepositions and features[-1] is not None:
            # Include preposition in form "prep::headword"
            features[-1] = "%s::%s" % (event.iobject[0], features[-1])
        return features

    @staticmethod
    def get_predicate_repr(entity, event, predicative_adjectives=False, transitivity=False):
        if transitivity:
            fn = predicate_relation_with_transitivity
        else:
            fn = predicate_relation
        return fn(entity, event, handle_predicative=predicative_adjectives)

    def get_model_predicate_repr(self, entity, event):
        return ArgumentCompositionNarrativeChainModel.get_predicate_repr(
            entity, event,
            predicative_adjectives=self.predicative_adjectives,
            transitivity=self.transitivity,
        )

    @property
    def _description(self):
        return """\
Neural composition event model
Event projection:     %d""" % (
            self.vector_size,
        )

    @classmethod
    def _load_from_directory(cls, directory, **kwargs):
        with open(os.path.join(directory, "pred_vocab"), "r") as f:
            pred_vocab = pickle.load(f)
        with open(os.path.join(directory, "arg_vocab"), "r") as f:
            arg_vocab = pickle.load(f)
        with open(os.path.join(directory, "projection_model"), "r") as f:
            projection_model = pickle.load(f)
        with open(os.path.join(directory, "params"), "r") as f:
            params = pickle.load(f)

        if os.path.exists(os.path.join(directory, "arg2_vocab")):
            with open(os.path.join(directory, "arg2_vocab"), "r") as f:
                arg2_vocab = pickle.load(f)
        else:
            arg2_vocab = None

        if os.path.exists(os.path.join(directory, "prepositions")):
            with open(os.path.join(directory, "prepositions"), "r") as f:
                prepositions = f.read().splitlines()
        else:
            prepositions = None

        return ArgumentCompositionNarrativeChainModel(projection_model, pred_vocab, arg_vocab,
                                                      predicative_adjectives=params["predicative_adjectives"],
                                                      transitivity=params["transitivity"],
                                                      arg2_vocab=arg2_vocab,
                                                      preposition_vocab=prepositions,
                                                      **kwargs)

    def _save_to_directory(self, directory, model_name, human_name=None):
        with open(os.path.join(directory, "pred_vocab"), "w") as f:
            pickle.dump(self.pred_vocab, f, -1)
        with open(os.path.join(directory, "arg_vocab"), "w") as f:
            pickle.dump(self.arg_vocab, f, -1)
        with open(os.path.join(directory, "projection_model"), "w") as f:
            pickle.dump(self.projection_model, f, -1)
        with open(os.path.join(directory, "params"), "w") as f:
            pickle.dump({
                "predicative_adjectives": self.predicative_adjectives,
                "transitivity": self.transitivity,
            }, f, -1)
        if self.arg2_vocab is not self.arg_vocab:
            with open(os.path.join(directory, "arg2_vocab"), "w") as f:
                pickle.dump(self.arg2_vocab, f, -1)
        if self._prepositions is not None:
            with open(os.path.join(directory, "prepositions"), "w") as f:
                f.write("\n".join(self._prepositions))

    def predict_args(self, entity, event, limit=100):
        """
        Given an event (with no arguments other than the entity), predict most likely values for the
        argument slots.

        This only made sense when the network was strictly made up of autoencoders. If you use it
        after fine-tuning of the full network, it will give odd results, as the reconstruction comes
        partly from the remnants of the autoencoders.

        """
        pred = self.get_model_predicate_repr(entity, event)
        if pred not in self.pred_vocab:
            # Unknown predicate, can't predict any args
            return [], [], []
        pred_index = self.pred_vocab[pred].index
        # Project into the hidden representation and out again
        __, arg0_predictions, arg1_predictions, arg2_predictions = \
            self.projection_model.reconstruct([pred_index], [-1], [-1], [-1])
        # Pull out the highest ranked arg words for each slot
        top_arg0s = [(self.arg_id2word[i], arg0_predictions[0, i])
                     for i in reversed(arg0_predictions[0].argsort())][:limit]
        top_arg1s = [(self.arg_id2word[i], arg1_predictions[0, i])
                     for i in reversed(arg1_predictions[0].argsort())][:limit]
        top_arg2s = [(self.arg_id2word[i], arg2_predictions[0, i])
                     for i in reversed(arg2_predictions[0].argsort())][:limit]
        return top_arg0s, top_arg1s, top_arg2s


class EventVectorNetwork(object):
    def __init__(self, predicate_vectors, argument_vectors, layer_sizes=[100],
                 predicate_input=None, arg0_input=None, arg1_input=None, arg2_input=None,
                 inputs_a=None, inputs_b=None, inputs_c=None):
        self.layer_sizes = layer_sizes
        self.projection_size = self.layer_sizes[-1]
        self.pred_vocab_size = predicate_vectors.shape[0]
        self.pred_vector_size = predicate_vectors.shape[1]

        # Very first inputs are integers to select the input vectors
        if predicate_input is not None:
            self.predicate_input = predicate_input
        else:
            self.predicate_input = T.vector("pred", dtype="int32")
        if arg0_input is not None:
            self.arg0_input = arg0_input
        else:
            self.arg0_input = T.vector("arg0", dtype="int32")
        if arg1_input is not None:
            self.arg1_input = arg1_input
        else:
            self.arg1_input = T.vector("arg1", dtype="int32")
        if arg2_input is not None:
            self.arg2_input = arg2_input
        else:
            self.arg2_input = T.vector("arg2", dtype="int32")

        # Wrap the input vector matrices in a Theano variable
        if type(argument_vectors) is tuple:
            arg0_vecs = argument_vectors[0]
            arg1_vecs = argument_vectors[1]
            arg2_vecs = argument_vectors[2]
        else:
            # Just one set of argument vectors: use for all arg positions
            arg0_vecs = arg1_vecs = arg2_vecs = argument_vectors

        # Wrap the input vector matrices in a Theano variable
        self.predicate_vectors = theano.shared(numpy.asarray(predicate_vectors, dtype=theano.config.floatX),
                                              name="pred_vectors")
        self.argument0_vectors = theano.shared(numpy.asarray(arg0_vecs, dtype=theano.config.floatX),
                                              name="arg0_vectors", borrow=False)
        self.argument1_vectors = theano.shared(numpy.asarray(arg1_vecs, dtype=theano.config.floatX),
                                              name="arg1_vectors", borrow=False)
        self.argument2_vectors = theano.shared(numpy.asarray(arg2_vecs, dtype=theano.config.floatX),
                                              name="arg2_vectors", borrow=False)

        # Theoretically, this could be different for the different arguments, though the way we use it now it never is
        self.arg_vocab_size = arg0_vecs.shape[0]
        self.arg_vector_size = arg0_vecs.shape[1]
        # Except in the case where we have a special arg2 vocab to allow for prepositions
        self.arg2_vocab_size = arg2_vecs.shape[0]

        # In order to stop the projections being thrown off by empty arguments, we need to learn an empty argument
        # vector. This is initialized to zero
        self.empty_arg0_vector = theano.shared(
            numpy.asarray(
                numpy.zeros(self.arg_vector_size)[None, :],
                dtype=theano.config.floatX
            ),
            name="empty_arg0",
            borrow=True,
            broadcastable=(True, False),
        )
        self.empty_arg1_vector = theano.shared(
            numpy.asarray(
                numpy.zeros(self.arg_vector_size)[None, :],
                dtype=theano.config.floatX
            ),
            name="empty_arg1",
            borrow=True,
            broadcastable=(True, False),
        )
        self.empty_arg2_vector = theano.shared(
            numpy.asarray(
                numpy.zeros(self.arg_vector_size)[None, :],
                dtype=theano.config.floatX
            ),
            name="empty_arg2",
            borrow=True,
            broadcastable=(True, False),
        )

        # NB I don't seem to use these anywhere!
        self.composition_weights = theano.shared(
            numpy.zeros((layer_sizes[-1]*2, 1), dtype=theano.config.floatX),
            name="composition_w",
            borrow=True,
        )
        self.composition_bias = theano.shared(
            numpy.zeros(1, dtype=theano.config.floatX),
            name="composition_b",
            borrow=True,
        )

        self.input_size = self.pred_vector_size + 3*self.arg_vector_size
        # Build the theano expression for this network
        self.input_vector, self.layers, self.layer_outputs, self.projection_layer = \
            EventVectorNetwork.build_projection_layer(
                self.predicate_input, self.arg0_input, self.arg1_input, self.arg2_input,
                self.predicate_vectors, self.argument0_vectors, self.argument1_vectors, self.argument2_vectors,
                self.empty_arg0_vector, self.empty_arg1_vector, self.empty_arg2_vector,
                self.input_size, self.layer_sizes
            )

        self.norm_projection_layer = self.projection_layer / \
                                     T.sqrt((self.projection_layer ** 2.).sum(axis=1)).reshape(
                                         (self.projection_layer.shape[0], 1))

        ### Composition of two projections
        if inputs_a is not None:
            self.predicate_input_a, self.arg0_input_a, self.arg1_input_a, self.arg2_input_a = inputs_a
        else:
            self.predicate_input_a = T.vector("pred_a", dtype="int32")
            self.arg0_input_a = T.vector("arg0_a", dtype="int32")
            self.arg1_input_a = T.vector("arg1_a", dtype="int32")
            self.arg2_input_a = T.vector("arg2_a", dtype="int32")

        if inputs_b is not None:
            self.predicate_input_b, self.arg0_input_b, self.arg1_input_b, self.arg2_input_b = inputs_b
        else:
            self.predicate_input_b = T.vector("pred_b", dtype="int32")
            self.arg0_input_b = T.vector("arg0_b", dtype="int32")
            self.arg1_input_b = T.vector("arg1_b", dtype="int32")
            self.arg2_input_b = T.vector("arg2_b", dtype="int32")

        # Or three
        if inputs_c is not None:
            self.predicate_input_c, self.arg0_input_c, self.arg1_input_c, self.arg2_input_c = inputs_c
        else:
            self.predicate_input_c = T.vector("pred_c", dtype="int32")
            self.arg0_input_c = T.vector("arg0_c", dtype="int32")
            self.arg1_input_c = T.vector("arg1_c", dtype="int32")
            self.arg2_input_c = T.vector("arg2_c", dtype="int32")

        # Compile the Theano functions
        # This projects all the way from the input to the output, once each layer's been trained
        self.project = theano.function(
            inputs=[self.predicate_input, self.arg0_input, self.arg1_input, self.arg2_input],
            outputs=self.projection_layer,
            name="project"
        )
        # Reconstruct input from the deepest projection
        reconstructed = self.projection_layer
        for layer in reversed(self.layers):
            reconstructed = layer.get_reconstructed_input(reconstructed)
        # Reverse the projection onto the input indices, so each vector gives the similarity to each input word
        pred_weights = T.dot(reconstructed[:, :self.pred_vector_size], self.predicate_vectors.T)
        arg0_weights = T.dot(reconstructed[:, self.pred_vector_size:(self.pred_vector_size+self.arg_vector_size)],
                             self.argument0_vectors.T)
        arg1_weights = T.dot(reconstructed[:, (self.pred_vector_size+self.arg_vector_size):(self.pred_vector_size+2*self.arg_vector_size)],
                             self.argument1_vectors.T)
        arg2_weights = T.dot(reconstructed[:, (self.pred_vector_size+2*self.arg_vector_size):],
                             self.argument2_vectors.T)
        self.reconstruct = theano.function(
            inputs=[self.predicate_input, self.arg0_input, self.arg1_input, self.arg2_input],
            outputs=[pred_weights, arg0_weights, arg1_weights, arg2_weights],
            name="reconstruct",
        )

    def copy_projection_function(self, predicate_input=None, arg0_input=None, arg1_input=None, arg2_input=None):
        """
        Build a new projection function, copying all weights and such from this network, replacing
        components given as kwargs. Note that this uses the same shared variables and any other
        non-replaced components as the network's original expression graph: bear in mind if you use
        it to update weights or combine with other graphs.

        """
        predicate_input = predicate_input or self.predicate_input
        arg0_input = arg0_input or self.arg0_input
        arg1_input = arg1_input or self.arg1_input
        arg2_input = arg2_input or self.arg2_input

        # Build a new projection function
        input_vector, layers, layer_outputs, projection_layer = EventVectorNetwork.build_projection_layer(
            predicate_input, arg0_input, arg1_input, arg2_input,
            self.predicate_vectors, self.argument0_vectors, self.argument1_vectors, self.argument2_vectors,
            self.empty_arg0_vector, self.empty_arg1_vector, self.empty_arg2_vector,
            self.input_size, self.layer_sizes
        )

        # Set all the layers' weights to the ones in the base network
        for layer, layer_weights in zip(layers, self.get_weights()):
            # get_weights gives some more things too, but the first weights correspond to layers
            layer.set_weights(layer_weights)

        return projection_layer, input_vector, layers, layer_outputs

    @staticmethod
    def build_projection_layer(predicate_input, arg0_input, arg1_input, arg2_input,
                               predicate_vectors, argument0_vectors, argument1_vectors, argument2_vectors,
                               empty_arg0_vector, empty_arg1_vector, empty_arg2_vector,
                               input_size, layer_sizes):
        # Rearrange these so we can test for -1 indices
        # In the standard case, this does dimshuffle((0, "x")), which changes a 1D vector into a column vector
        shuffled_dims = tuple(list(range(arg0_input.ndim)) + ["x"])
        arg0_input_col = arg0_input.dimshuffle(shuffled_dims)
        arg1_input_col = arg1_input.dimshuffle(shuffled_dims)
        arg2_input_col = arg2_input.dimshuffle(shuffled_dims)

        # As standard, the empty vectors already have one empty, broadcastable dimension at the start
        # Add more if the inputs are of higher rank
        #empty_arg0_matrix = empty_arg0_vector.dimshuffle(["x"]*(arg0_input.ndim-1) + [0, 1])
        #empty_arg1_matrix = empty_arg1_vector.dimshuffle(["x"]*(arg0_input.ndim-1) + [0, 1])
        #empty_arg2_matrix = empty_arg2_vector.dimshuffle(["x"]*(arg0_input.ndim-1) + [0, 1])

        # Make the input to the first autoencoder by selecting the appropriate vectors from the given matrices
        input_vector = T.concatenate([
            predicate_vectors[predicate_input],
            T.switch(T.neq(arg0_input_col, -1), argument0_vectors[arg0_input], empty_arg0_vector),
            T.switch(T.neq(arg1_input_col, -1), argument1_vectors[arg1_input], empty_arg1_vector),
            T.switch(T.neq(arg2_input_col, -1), argument2_vectors[arg2_input], empty_arg2_vector),
        ], axis=predicate_input.ndim)

        # Build and initialize each layer of the autoencoder
        previous_output = input_vector
        layers = []
        layer_outputs = []
        for layer_size in layer_sizes:
            layers.append(
                DenoisingAutoencoder(
                    input=previous_output,
                    n_hidden=layer_size,
                    n_visible=input_size,
                    non_linearity="tanh",
                )
            )
            input_size = layer_size
            previous_output = layers[-1].hidden_layer
            layer_outputs.append(previous_output)
        projection_layer = previous_output

        return input_vector, layers, layer_outputs, projection_layer

    def shift_input_vectors(self, shift_vector):
        shift_vector = numpy.asarray(shift_vector, dtype=theano.config.floatX)
        for vec_var, range_start, range_end in [
            (self.predicate_vectors, 0, self.pred_vector_size),
            (self.argument0_vectors, self.pred_vector_size, self.pred_vector_size+self.arg_vector_size),
            (self.argument1_vectors, self.pred_vector_size+self.arg_vector_size, self.pred_vector_size+2*self.arg_vector_size),
            (self.argument2_vectors, self.pred_vector_size+2*self.arg_vector_size, self.pred_vector_size+3*self.arg_vector_size),
        ]:
            vecs = vec_var.get_value()
            vecs += shift_vector[:vecs.shape[1]]
            vec_var.set_value(vecs)

    def scale_input_vectors(self, scale_vector):
        scale_vector = numpy.asarray(scale_vector, dtype=theano.config.floatX)
        for vec_var, range_start, range_end in [
            (self.predicate_vectors, 0, self.pred_vector_size),
            (self.argument0_vectors, self.pred_vector_size, self.pred_vector_size+self.arg_vector_size),
            (self.argument1_vectors, self.pred_vector_size+self.arg_vector_size, self.pred_vector_size+2*self.arg_vector_size),
            (self.argument2_vectors, self.pred_vector_size+2*self.arg_vector_size, self.pred_vector_size+3*self.arg_vector_size),
        ]:
            vecs = vec_var.get_value()
            vecs *= scale_vector[:vecs.shape[1]]
            vec_var.set_value(vecs)

    def get_projection_pair(self, normalize=False):
        if normalize:
            projection = self.norm_projection_layer
        else:
            projection = self.projection_layer

        # Use the deepest projection layer as our target representation
        # Now clone this so that we can perform two projections in the same step
        projection_a = theano.clone(projection, replace={
            self.predicate_input: self.predicate_input_a,
            self.arg0_input: self.arg0_input_a,
            self.arg1_input: self.arg1_input_a,
            self.arg2_input: self.arg2_input_a,
        })
        projection_b = theano.clone(projection, replace={
            self.predicate_input: self.predicate_input_b,
            self.arg0_input: self.arg0_input_b,
            self.arg1_input: self.arg1_input_b,
            self.arg2_input: self.arg2_input_b,
        })
        return projection_a, projection_b

    def get_projection_triple(self, normalize=False):
        if normalize:
            projection = self.norm_projection_layer
        else:
            projection = self.projection_layer

        # Use the deepest projection layer as our target representation
        # Now clone this so that we can perform three projections in the same step
        projection_a = theano.clone(projection, replace={
            self.predicate_input: self.predicate_input_a,
            self.arg0_input: self.arg0_input_a,
            self.arg1_input: self.arg1_input_a,
            self.arg2_input: self.arg2_input_a,
        })
        projection_b = theano.clone(projection, replace={
            self.predicate_input: self.predicate_input_b,
            self.arg0_input: self.arg0_input_b,
            self.arg1_input: self.arg1_input_b,
            self.arg2_input: self.arg2_input_b,
        })
        projection_c = theano.clone(projection, replace={
            self.predicate_input: self.predicate_input_c,
            self.arg0_input: self.arg0_input_c,
            self.arg1_input: self.arg1_input_c,
            self.arg2_input: self.arg2_input_c,
        })
        return projection_a, projection_b, projection_c

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
            inputs=[self.predicate_input, self.arg0_input, self.arg1_input, self.arg2_input],
            outputs=output_eq,
            name="layer-%d-input" % layer_num,
        )

    def get_layer_projection(self, layer_num):
        if layer_num >= len(self.layers):
            raise ValueError("cannot get input function for layer %d in a %d-layer network" %
                             (layer_num, len(self.layers)))
        elif layer_num == -1:
            # Special value to get the input to the first layer: i.e. just the concatenated input vectors
            output_eq = self.input_vector
        else:
            output_eq = self.layer_outputs[layer_num]

        return theano.function(
            inputs=[self.predicate_input, self.arg0_input, self.arg1_input, self.arg2_input],
            outputs=output_eq,
            name="layer-%d-projection" % layer_num,
        )

    def get_weights(self):
        return [ae.get_weights() for ae in self.layers] + \
               [self.empty_arg0_vector.get_value(),
                self.empty_arg1_vector.get_value(),
                self.empty_arg2_vector.get_value(),
                self.composition_weights.get_value(),
                self.composition_bias.get_value()]

    def set_weights(self, weights):
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)
        self.empty_arg0_vector.set_value(weights[len(self.layers)])
        self.empty_arg1_vector.set_value(weights[len(self.layers)+1])
        self.empty_arg2_vector.set_value(weights[len(self.layers)+2])
        if len(weights) > len(self.layers) + 3:
            self.composition_weights.set_value(weights[len(self.layers)+3])
            self.composition_bias.set_value(weights[len(self.layers)+4])

    def __getstate__(self):
        return {
            "weights": self.get_weights(),
            "predicate_vectors": self.predicate_vectors.get_value(),
            "argument_vectors": (
                self.argument0_vectors.get_value(),
                self.argument1_vectors.get_value(),
                self.argument2_vectors.get_value()
            ),
            "layer_sizes": self.layer_sizes,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(state["predicate_vectors"], state["argument_vectors"], layer_sizes=state["layer_sizes"])
        self.set_weights(state["weights"])


def build_id2word(vocab):
    return [word for (id, word) in sorted([(v.index, word) for (word, v) in vocab.items()])]