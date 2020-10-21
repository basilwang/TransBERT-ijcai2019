import math
import warnings

import numpy
import numpy.random
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

from whim_common.utils.plotting import plot_costs


class LogisticRegression(object):
    def __init__(self, num_features, num_classes, input=None):
        self.num_features = num_features
        self.num_classes = num_classes

        # Set up Theano network for the model
        # Features: (m, num_features)
        if input:
            x = input
        else:
            x = T.matrix("x")
        # Classes (int indices): (m)
        y = T.ivector("y")
        # Weights and bias, initialized to 0
        self.w = theano.shared(
            value=numpy.zeros((num_features, num_classes)),
            name='w',
            borrow=True
        )
        self.b = theano.shared(numpy.zeros(num_classes), name="b")

        # Construct Theano expression graph
        self.activation = T.dot(x, self.w) + self.b
        # Softmax activations to get a probability distribution over the classes
        self.class_probs = nnet.softmax(self.activation)
        # Another version without the bias included
        self.class_probs_without_bias = nnet.softmax(T.dot(x, self.w))
        # The predicted class is that with highest activation (no need to do the softmax for this)
        self.prediction = T.argmax(self.activation, axis=1)
        # Cross-entropy loss function
        self.xent = -T.mean(T.log(self.class_probs)[T.arange(y.shape[0]), y])
        # Cross-entropy summed, rather than averaged, so we can split it up and sum the results
        self.xent_sum = -T.sum(T.log(self.class_probs)[T.arange(y.shape[0]), y])
        # Params that are to be updated
        self.params = [self.w, self.b]
        # Error in terms of hard predictions (accuracy)
        self.error = T.mean(T.cast(T.neq(self.prediction, y), "float64"))

        # Compile
        self._predict_fn = theano.function(inputs=[x], outputs=self.prediction)
        self._prob_fn = theano.function(inputs=[x], outputs=self.class_probs)
        self._prob_fn_without_bias = theano.function(inputs=[x], outputs=self.class_probs_without_bias)
        self._error_fn = theano.function(
            inputs=[x, y],
            outputs=self.error
        )

        self.x = x
        self.y = y

    def train(self, xs, ys, iterations=10000, iteration_callback=None,
              validation_xs=None, validation_ys=None, learning_rate=0.1, regularization=0.01,
              plot_errors=None, plot_cost=None, minibatch=1, early_stopping_iterations=5, batch_getter=None,
              validation_batch_size=0):
        """
        Train on data stored in Theano tensors.

        E.g.
        xs = rng.randn(N, num_features)
        ys = rng.randint(size=N, low=0, high=2)

        iteration_callback is called after each iteration with args (iteration, error array).

        Returns True if we stop early because of early stopping criterion

        """
        if plot_errors:
            warnings.warn("Error plotting is no longer implemented for logistic regression")

        if batch_getter is None:
            # Default getter produces an iterator over fixed minibatch sizes and assumes the xs and ys are arrays
            def batch_getter(x_source, y_source, batch_size):
                if batch_size == 0:
                    # Interpret minibatch == 0 as batch
                    batch_size = x_source.shape[0]
                for batch in range(int(math.ceil(float(x_source.shape[0]) / batch_size))):
                    yield x_source[minibatch*batch:minibatch*(batch+1)], y_source[minibatch*batch:minibatch*(batch+1)]
            return batch_getter

        # Build the cost function
        # The cost to minimize, including L2 regularization
        cost = self.xent + regularization * (self.w ** 2).mean()
        _summed_cost_fn = theano.function(
            inputs=[self.x, self.y],
            outputs=self.xent_sum,
        )
        updates = [(param, param - learning_rate * T.grad(cost, param)) for param in self.params]
        # Build the training function
        _train_fn = theano.function(
            inputs=[self.x, self.y],
            outputs=cost,
            updates=updates,
        )

        validation_costs = []
        costs = []

        best_val_cost = numpy.inf
        best_weights = None
        last_best_iter = 0
        early_stop = False

        for i in range(iterations):
            new_best = False

            # Do an update for each minibatch
            num_batches = 0
            training_cost = 0.
            training_points = 0

            for batch_xs, batch_ys in batch_getter(xs, ys, minibatch):
                num_batches += 1
                # Randomize the order within the batch
                permutation = numpy.random.permutation(batch_xs.shape[0])
                batch_xs = batch_xs[permutation].copy()
                batch_ys = batch_ys[permutation].copy()

                # Update on the batch
                _train_fn(batch_xs, batch_ys)
                # Compute the training cost as we go
                training_cost += _summed_cost_fn(batch_xs, batch_ys)
                training_points += batch_xs.shape[0]

            training_cost /= training_points
            # Compute training set cost
            costs.append(training_cost)

            if validation_xs is not None:
                # Compute cost in val set (without regularization, of course)
                validation_cost = 0.
                validation_points = 0
                for val_batch_xs, val_batch_ys in batch_getter(validation_xs, validation_ys, validation_batch_size):
                    validation_cost += _summed_cost_fn(val_batch_xs, val_batch_ys)
                    validation_points += val_batch_xs.shape[0]
                validation_cost /= validation_points
                validation_costs.append(validation_cost)

                # Check whether we've got a new set of best weights, according to validation cost
                if i == 0 or validation_costs[-1] < best_val_cost:
                    best_val_cost = validation_costs[-1]
                    best_weights = self.get_weights()
                    last_best_iter = i
                    new_best = True

                # Test for the early stopping condition
                if i - last_best_iter >= early_stopping_iterations:
                    # We've gone for enough iterations without an improvement in validation cost
                    # Give up and use best weights so far
                    self.set_weights(best_weights)
                    early_stop = True

            if iteration_callback is not None:
                # The empty lists are where val and training errors used to be, left for backwards compat
                iteration_callback(i, costs, validation_costs, [], [], new_best)

            # Plot some graphs
            if plot_cost:
                plot_costs(plot_cost, (costs, "training cost"), (validation_costs, "val cost"))

            if early_stop:
                # We've decided to give up here on the basis of validation cost
                return True

    def predict(self, xs):
        return self._predict_fn(xs)

    def probs(self, xs, disable_bias=False):
        if disable_bias:
            return self._prob_fn_without_bias(xs)
        else:
            return self._prob_fn(xs)

    def get_weights(self):
        return self.w.get_value(), self.b.get_value()

    def set_weights(self, weights):
        self.w.set_value(weights[0])
        self.b.set_value(weights[1])

    ####################################
    #### Pickling for model storage ####
    def __getstate__(self):
        return {
            "w": self.w.get_value(),
            "b": self.b.get_value(),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(state["num_features"], state["num_classes"])
        self.w.set_value(state["w"])
        self.b.set_value(state["b"])