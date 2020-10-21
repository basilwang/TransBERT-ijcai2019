import copy

import theano

from whim_common.ml.autoencoder.layer import DenoisingAutoencoder


class StackedDenoisingAutoencoder(object):
    """
    A stack of denoising autoencoders. Handy wrapper to initialize them all at once.

    """
    def __init__(self, input_size, layer_sizes, input=None, non_linearity="sigmoid"):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.non_linearity = non_linearity
        self.layers = []

        # Create an autoencoder for each layer
        layer_input = input
        layer_input_size = input_size
        for layer_size in self.layer_sizes:
            # Initialize the layer
            layer = DenoisingAutoencoder(
                input=layer_input,
                n_visible=layer_input_size,
                n_hidden=layer_size,
                non_linearity=non_linearity,
            )
            self.layers.append(layer)
            # Use this layer's hidden units as input to the next
            layer_input = layer.hidden_layer
            layer_input_size = layer_size

        # If an input variable wasn't given, the first layer will have created its own default: use that
        if input is None:
            input = self.layers[0].x
        self.x = input
        self.deepest_hidden_layer = self.layers[-1].hidden_layer

        # Don't compile functions on init, but cached when needed
        self._projection_fn = None

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_weights(self, layer_weights):
        for layer, weights in zip(self.layers, layer_weights):
            layer.set_weights(weights)

    def to_feedforward(self):
        """
        Throw away the reconstruction weights and produce a feedforward network.

        """
        from ..feedforward import FeedforwardNetwork
        network = FeedforwardNetwork(self.input_size, copy.copy(self.layer_sizes),
                                     input=self.x, non_linearity=self.non_linearity)
        # Get the forward weights from each layer
        for ff_layer, ae_layer in zip(network.layers, self.layers):
            ff_layer.W.set_value(ae_layer.W.get_value())
            ff_layer.b.set_value(ae_layer.b.get_value())
        return network

    @property
    def project(self):
        if self._projection_fn is None:
            # Compile the projection function
            self._projection_fn = theano.function(
                inputs=[self.x],
                outputs=self.deepest_hidden_layer,
            )
        return self._projection_fn

    def __getstate__(self):
        return {
            "input_size": self.input_size,
            "layer_sizes": self.layer_sizes,
            # Store the weights for each layer, not the layers themselves
            "weights": self.get_weights(),
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(input_size=state["input_size"], layer_sizes=state["layer_sizes"],
                      non_linearity=state["non_linearity"])
        self.set_weights(state["weights"])
