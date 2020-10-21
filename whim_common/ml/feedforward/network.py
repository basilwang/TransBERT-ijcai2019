import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.random import randint, uniform


class FeedforwardNetwork(object):
    def __init__(self, input_size, layer_sizes, input=None, non_linearity="sigmoid"):
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.non_linearity = non_linearity

        self.layers = []
        layer_input = input
        layer_input_size = input_size
        for layer_size in layer_sizes:
            # Build a layer
            layer = FeedforwardLayer(input=layer_input, n_visible=layer_input_size, n_hidden=layer_size,
                                     non_linearity=non_linearity)
            self.layers.append(layer)
            # Use the output of this layer as the input to the next
            layer_input = layer.hidden_layer
            layer_input_size = layer_size
        self.x = self.layers[0].x if input is None else input

    @property
    def hidden_layer(self):
        return self.layers[-1].hidden_layer

    def projection(self, xs):
        if self._hidden_fn is None:
            self._hidden_fn = theano.function(
                inputs=[self.x],
                outputs=self.hidden_layer,
            )
        return self._hidden_fn(xs)

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_weights(self, weights):
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)

    def add_layer(self, layer_size, non_linearity=None):
        """
        Add an extra layer on top of the network, taking input from the previous deepest layer.
        """
        if non_linearity is None:
            # Default to the network's main non-linearity
            non_linearity = self.non_linearity

        self.layer_sizes.append(layer_size)
        self.layers.append(
            FeedforwardLayer(
                input=self.hidden_layer,
                n_visible=self.layers[-1].n_hidden,
                n_hidden=layer_size,
                non_linearity=non_linearity
            )
        )

    def get_cost(self, y, loss="xent"):
        prediction = self.hidden_layer
        if loss == "xent":
            # Cross-entropy loss function
            if self.non_linearity == "tanh":
                # If we're using tanh activation, we need to shift & scale the output into the (0, 1) range
                prediction = 0.5 * (prediction + 1.)
            L = - T.sum((y * T.log(prediction) + (1 - y) * T.log(1 - prediction)), axis=1)
        elif loss == "l2":
            # Squared error loss function
            L = T.sum(0.5 * ((y - prediction)**2), axis=1)
        else:
            raise ValueError("unknown loss function '%s'. Expected one of: 'xent', 'l2'" % loss)
        # L is now a vector, where each element is the cross-entropy/L2 cost of the prediction of the
        # corresponding example of the minibatch
        return T.mean(L)

    def get_l2_regularization(self, extra_params=[]):
        return T.mean(T.concatenate([T.flatten(layer.W) for layer in self.layers] + extra_params) ** 2.)

    @property
    def params(self):
        return sum((layer.params for layer in self.layers), [])

    def __getstate__(self):
        return {
            "input_size": self.input_size,
            "layer_sizes": self.layer_sizes,
            "non_linearity": self.non_linearity,
            "weights": self.get_weights(),
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(state["input_size"], state["layer_sizes"], non_linearity=state["non_linearity"])
        self.set_weights(state["weights"])


class FeedforwardLayer(object):
    def __init__(self, input=None, n_visible=784, n_hidden=500, non_linearity="sigmoid"):
        """
        Initialize by specifying the number of visible units (the
        dimension d of the input), the number of hidden units (the dimension
        d' of the latent or hidden space). The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer.

        NB: based on cut-down version of autoencoder implementation. Doesn't currently
        implement the cost functions for supervised training.

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        """
        self.n_visible = n_visible
        self.num_features = n_visible  # Alias
        self.n_hidden = n_hidden
        self.non_linearity = non_linearity

        if non_linearity == "sigmoid":
            self.activation_fn = T.nnet.sigmoid
            self.inverse_activation_fn = lambda x: T.log(x / (1-x))
        elif non_linearity == "tanh":
            self.activation_fn = T.tanh
            self.inverse_activation_fn = T.arctanh
        else:
            raise ValueError("unkown non-linearity '%s'. Must be 'sigmoid' or 'tanh'" % non_linearity)

        # create a Theano random generator that gives symbolic random values
        theano_rng = RandomStreams(randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        # W is initialized with `initial_W` which is uniformely sampled
        # from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        initial_W = numpy.asarray(
            uniform(
                low=-numpy.sqrt(6. / (n_hidden + n_visible)),
                high=numpy.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden),
            ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=initial_W, name='W', borrow=True)

        bhid = theano.shared(
            value=numpy.zeros(
                n_hidden,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        self.x_as_int = T.cast(self.x, "int8")

        self.params = [self.W]

        self.hidden_layer = self.get_hidden_values(self.x)

        self._hidden_fn = None
        self._b_copy = None
        self._b_prime_copy = None

    def projection(self, xs):
        if self._hidden_fn is None:
            self._hidden_fn = theano.function(
                inputs=[self.x],
                outputs=self.hidden_layer,
            )
        return self._hidden_fn(xs)

    def get_weights(self):
        """
        Return a copy of all the weight arrays in a tuple.

        """
        return (self.W.get_value().copy(),
                self.b.get_value().copy())

    def set_weights(self, weights):
        """
        Set all weights from a tuple, like that returned by get_weights().

        """
        self.W.set_value(weights[0])
        self.b.set_value(weights[1])

    def get_hidden_layer_activation(self, input):
        return T.dot(input, self.W) + self.b

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return self.activation_fn(T.dot(input, self.W) + self.b)

    ####################################
    #### Pickling for model storage ####

    def __getstate__(self):
        return {
            "W": self.W.get_value(),
            "b": self.b.get_value(),
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "non_linearity": self.non_linearity,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(n_visible=state["n_visible"], n_hidden=state["n_hidden"], non_linearity=state["non_linearity"])
        self.set_weights((state["W"], state["b"]))
