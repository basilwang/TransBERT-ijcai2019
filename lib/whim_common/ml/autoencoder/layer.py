"""
Autoencoder implementation based closely on that in the Theano tutorial on
denoising autoencoders.

"""
import numpy
from numpy.random.mtrand import randint, uniform
from theano import tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams


class DenoisingAutoencoder(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoder tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    """
    def __init__(self, input=None, n_visible=784, n_hidden=500, no_bias=False, non_linearity="sigmoid"):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type continuous_corruption: bool
        :param continuous_corruption: instead of just zeroing out randomly-chosen elements, scale all the
            1-valued inputs randomly to floats chosen uniformly in the range (1-corruption_level) to 1.

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

        bvis = theano.shared(
            value=numpy.zeros(
                n_visible,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

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
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        self.x_as_int = T.cast(self.x, "int8")

        if no_bias:
            # Don't include a bias in the network
            # The bias still features in the equations, but isn't include among the params, so doesn't get updated
            # It therefore retains its initial values: 0
            self.params = [self.W]
        else:
            self.params = [self.W, self.b, self.b_prime]

        self.hidden_layer = self.get_hidden_values(self.x)

        self._b_copy = None
        self._b_prime_copy = None
        self._hidden_fn = None
        self._probs_fn = None

    def to_feedforward(self):
        """
        Create a feedforward layer that is the same as using this autoencoder as a feedforward
        layer. That is, it has the same forwards weights and throws away the backward
        (reconstruction) weights.
        """
        from ..feedforward import FeedforwardLayer
        # Initialize a layer
        layer = FeedforwardLayer(
            input=self.x, n_visible=self.n_visible, n_hidden=self.n_hidden, non_linearity=self.non_linearity
        )
        # Set its weights to be the same as the forward part of ours
        layer.set_weights((self.W, self.b))
        return layer

    @property
    def bias_disabled(self):
        # When the bias is disabled, we keep a copy of the bias weights
        return self._b_copy is not None

    def toggle_bias(self):
        if self.bias_disabled:
            # Re-enable the bias
            self.b.set_value(self._b_copy)
            self.b_prime.set_value(self._b_prime_copy)
            self._b_copy = None
            self._b_prime_copy = None
            return True
        else:
            # Keep a copy of the bias weights and zero out the ones being used
            self._b_copy = self.b.get_value().copy()
            self._b_prime_copy = self.b_prime.get_value().copy()
            self.b.set_value(numpy.zeros(self.n_hidden, dtype=theano.config.floatX))
            self.b_prime.set_value(numpy.zeros(self.n_visible, dtype=theano.config.floatX))
            return False

    def projection(self, xs):
        if self._hidden_fn is None:
            # Compile when first needed
            self._hidden_fn = theano.function(
                inputs=[self.x],
                outputs=self.hidden_layer,
            )
        return self._hidden_fn(xs)

    @property
    def prediction_dist(self):
        if self._probs_fn is None:
            self._probs_fn = theano.function(
                inputs=[self.x],
                outputs=self.get_prediction_dist(),
            )
        return self._probs_fn

    def get_weights(self):
        """
        Return a copy of all the weight arrays in a tuple.

        """
        return (self.W.get_value().copy(),
                self.b.get_value().copy(),
                self.b_prime.get_value().copy())

    def set_weights(self, weights):
        """
        Set all weights from a tuple, like that returned by get_weights().

        """
        self.W.set_value(weights[0])
        self.b.set_value(weights[1])
        self.b_prime.set_value(weights[2])

    def get_corrupted_input(self, input, corruption_level=0., continuous_corruption=False):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        if corruption_level > 0.:
            if continuous_corruption:
                return self.theano_rng.uniform(size=input.shape,
                                               low=1.-corruption_level, high=1.,
                                               dtype=theano.config.floatX) * input
            else:
                return self.theano_rng.binomial(size=input.shape, n=1,
                                                p=1 - corruption_level,
                                                dtype=theano.config.floatX) * input
        else:
            return input

    def get_hidden_layer_activation(self, input):
        return T.dot(input, self.W) + self.b

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return self.activation_fn(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return self.activation_fn(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_reconstruction(self, corrupt=True, corruption_level=0., continuous_corruption=False):
        if corrupt and corruption_level > 0.:
            tilde_x = self.get_corrupted_input(self.x, corruption_level=corruption_level,
                                               continuous_corruption=continuous_corruption)
        else:
            tilde_x = self.x
        y = self.get_hidden_values(tilde_x)
        return self.get_reconstructed_input(y)

    def get_predictions(self, threshold):
        return T.gt(self.get_reconstruction(corrupt=False), threshold)

    def get_f_scores(self):
        prediction = self.get_predictions(0.5)  # 0.5 is an arbitrary threshold

        # Different computation for R, P and F with the autoencoder
        true_pos = T.sum(prediction & self.x_as_int, axis=0)
        pos = T.sum(self.x_as_int, axis=0)
        predicted_pos = T.sum(prediction, axis=0)

        # If pos==0 (no actual positives) recall is undefined
        # Simple way out of div zero: wherever pos==0, setting pos=1 is fine (since recall==1)
        recalls = T.switch(T.eq(pos, 0), float('nan'), true_pos) / T.switch(T.eq(pos, 0), 1., pos)
        # Simple way out of div zero: wherever predicted_pos==0 we're setting num directly, so 1 denom is fine
        precisions = T.switch(
            T.eq(predicted_pos, 0) & T.eq(pos, 0),
            float('nan'),  # Don't penalize precision if there are no positives
            true_pos / T.switch(T.eq(predicted_pos, 0), 1., predicted_pos)
        )
        f_scores = T.switch(
            T.isnan(precisions) | T.isnan(recalls),
            float('nan'),
            2. * precisions * recalls /
            T.switch(
                precisions + recalls > 0,
                precisions + recalls,
                1.
            ),
        )
        return f_scores, precisions, recalls

    def get_prediction_dist(self, exlcude_input=False):
        # Normalize the output layer
        reconstruction = self.get_reconstruction(corrupt=False)

        if exlcude_input:
            # Zero out things that were in the input, so we just get a distribution over new predictions
            reconstruction *= T.eq(self.x, 0.)

        probs = (reconstruction.T / T.sum(reconstruction, axis=1)).T
        return probs

    def get_cost(self, class_cost_weights=None, corruption_level=0., continuous_corruption=False,
                 loss="xent"):
        z = self.get_reconstruction(corruption_level=corruption_level, continuous_corruption=continuous_corruption)

        if class_cost_weights is None:
            class_cost_weights = 1.0

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if loss == "xent":
            # Cross-entropy loss function
            if self.non_linearity == "tanh":
                # If we're using tanh activation, we need to shift & scale the output into the (0, 1) range
                z = 0.5 * (z + 1.)
            L = - T.sum((self.x * T.log(z) + (1 - self.x) * T.log(1 - z)) * class_cost_weights, axis=1)
        elif loss == "l2":
            # Squared error loss function
            L = T.sum(0.5 * ((self.x - z)**2), axis=1)
        else:
            raise ValueError("unknown loss function '%s'. Expected one of: 'xent', 'l2'" % loss)
        # L is now a vector, where each element is the cross-entropy cost of the reconstruction of the
        # corresponding example of the minibatch
        return T.mean(L)

    def get_l2_regularization(self):
        return T.mean(self.W ** 2.)

    def get_cost_updates(self, learning_rate, regularization, class_cost_weights=None,
                         corruption_level=0., continuous_corruption=False, loss="xent"):
        """ This function computes the cost and the updates for one training
        step of the dA

        """
        cost = self.get_cost(class_cost_weights=class_cost_weights,
                             corruption_level=corruption_level,
                             continuous_corruption=continuous_corruption,
                             loss=loss)
        reg = self.get_l2_regularization()
        # Include the regularization term, weighted by lambda
        cost += regularization * reg

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = [T.grad(cost, param) for param in self.params]
        # generate the list of updates
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]

        return cost, updates

    ####################################
    #### Pickling for model storage ####

    def __getstate__(self):
        return {
            "W": self.W.get_value(),
            "b": self.b.get_value(),
            "b_prime": self.b_prime.get_value(),
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "non_linearity": self.non_linearity,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(n_visible=state["n_visible"], n_hidden=state["n_hidden"], non_linearity=state["non_linearity"])
        self.set_weights((state["W"], state["b"], state["b_prime"]))
