"""
Autoencoder implementation based closely on that in the Theano tutorial on
denoising autoencoders.

"""
import numpy
from numpy.random import randint, uniform

try:
    from matplotlib import pyplot as plt
    from whim_common.utils.plotting import plot_costs
except ImportError:
    # Plotting will not be available, as pyplot isn't installed
    plt = None
    plot_costs = None
import random

import theano
import theano.tensor as T
from theano import Param
from theano.tensor.shared_randomstreams import RandomStreams
from whim_common.utils.logging import get_console_logger
from whim_common.utils.probability.sample import balanced_array_sample


class DenoisingAutoencoderTrainer(object):
    def __init__(self, network):
        self.network = network
        self.learning_rate = T.scalar("learning_rate")
        self.regularization = T.scalar("regularization")

        # F-score computations
        f_score_fn = network.get_f_scores()
        self.f_scores_fn = theano.function(
            inputs=[network.x],
            outputs=f_score_fn,
        )

    def compute_f_scores(self, xs):
        f_scores, precisions, recalls = self.f_scores_fn(xs)

        f_scores = f_scores[numpy.where(~numpy.isnan(f_scores))]
        precisions = precisions[numpy.where(~numpy.isnan(precisions))]
        recalls = recalls[numpy.where(~numpy.isnan(recalls))]

        return numpy.mean(f_scores), numpy.mean(precisions), numpy.mean(recalls), f_scores.shape[0]

    def train(self, xs, iterations=10000, iteration_callback=None,
              batch_size=20, batch_callback=None, validation_set=None, stopping_iterations=10, log=None,
              cost_plot_filename=None, training_cost_prop_change_threshold=0.0005, learning_rate=0.1,
              regularization=None, class_weights=None, corruption_level=0., continuous_corruption=False,
              loss="xent"):
        """
        Train on data stored in Theano tensors. Uses minibatch training.

        xs are the vectors to train on. Targets needn't be given, since the input and output are the
        same in an autoencoder.

        iteration_callback is called after each iteration with args (iteration, error array).

        If a validation set (matrix) is given, it is used to compute an error after each iteration
        and to enforce a stopping criterion. The algorithm will terminate if it goes stopping_iterations
        iterations without an improvement in validation error.

        If compute_error_frequency > 1 (default=5), this number of iterations are performed between each time
        the error is computed on the training set.

        The algorithm will assume it has converged and stop early if the proportional change between successive
        training costs drops below training_cost_prop_change_threshold for five iterations in a row.

        Uses L2 regularization.

        Several params are included just to implement the same interface as single_hidden_layer.
        Might want to change this later to be a bit neater.

        """
        if log is None:
            log = get_console_logger("Autoencoder train")

        log.info("Training params: learning rate=%s, noise ratio=%.1f%% (%s), regularization=%.2f" %
                 (learning_rate, self.network.corruption_level * 100.0,
                  "continuous corruption" if self.network.continuous_corruption else "zeroing corruption",
                  regularization))
        log.info("Training with SGD, batch size=%d" % batch_size)

        if class_weights is None:
            # Don't apply any weighting
            class_weights_vector = None
        elif class_weights == "freq":
            # Apply inverse frequency weighting
            class_counts = numpy.maximum(xs.sum(axis=0), 1.0)
            class_weights_vector = 1. / class_counts
            class_weights_vector *= xs.shape[1] / class_weights_vector.sum()
            log.info("Using inverse frequency class weighting in cost function")
        elif class_weights == "log":
            class_counts = numpy.maximum(xs.sum(axis=0), 1.0)
            class_weights_vector = 1. / (numpy.log(class_counts) + 1.)
            class_weights_vector *= xs.shape[1] / class_weights_vector.sum()
            log.info("Using inverse log frequency class weighting in cost function")
        else:
            raise ValueError("invalid class weighting '%s'" % class_weights)

        ######## Compile functions
        # Prepare cost/update functions for training
        cost, updates = self.network.get_cost_updates(self.learning_rate, self.regularization,
                                                      class_cost_weights=class_weights_vector,
                                                      corruption_level=corruption_level,
                                                      continuous_corruption=continuous_corruption,
                                                      loss=loss)
        # Prepare training functions
        cost_fn = theano.function(
            inputs=[self.network.x, Param(self.regularization, default=0.0)],
            outputs=cost,
        )
        train_fn = theano.function(
            inputs=[
                self.network.x,
                Param(self.learning_rate, default=0.1),
                Param(self.regularization, default=0.0)
            ],
            outputs=cost,
            updates=updates,
        )
        # Prepare a function to test how close to the identity function the learned mapping is
        # A lower value indicates that it's generalizing more (though not necessarily better)
        identity_ratio = T.mean(T.sum(self.network.get_prediction_dist() * (self.network.x > 0), axis=1))
        identity_ratio_fn = theano.function(
            inputs=[self.network.x],
            outputs=identity_ratio
        )
        ###########

        # Throw away ys in validation set
        validation_set = validation_set[0]

        # Prepare a prediction validation set by holding one event out of every chain in the val set
        prediction_targets = numpy.array([random.choice(numpy.where(x_row > 0)[0]) for x_row in validation_set],
                                         dtype=numpy.int16)
        prediction_contexts = validation_set.copy()
        prediction_contexts[range(prediction_contexts.shape[0]), prediction_targets] = 0.
        prediction_balanced_sample = balanced_array_sample(prediction_targets, balance_ratio=4., min_inclusion=1)
        prediction_targets = prediction_targets[prediction_balanced_sample]
        prediction_contexts = prediction_contexts[prediction_balanced_sample]
        log.info("Prepared roughly balanced prediction set from validation set with %d examples" %
                 prediction_contexts.shape[0])

        # Work out how many batches to do
        if batch_size is None or batch_size == 0:
            num_batches = 1
        else:
            num_batches = xs.shape[0] / batch_size
            if xs.shape[0] % batch_size != 0:
                num_batches += 1

        # Keep a record of costs, so we can plot them
        val_costs = []
        training_costs = []

        # Compute costs using the initialized network
        training_cost = cost_fn(xs)
        training_costs.append(training_cost)
        if validation_set is not None:
            val_cost = cost_fn(validation_set)
            val_costs.append(val_cost)
        else:
            val_cost = None

        log.info("Computing initial validation scores")
        f_score, precision, recall, f_score_classes = self.compute_f_scores(validation_set)
        log.info("F-score: %.4f%% (mean over %d classes), P=%.4f%%, R=%.4f%%" %
                 (f_score * 100.0, f_score_classes, precision * 100.0, recall * 100.0))
        log_prob = self.network.prediction_log_prob(prediction_contexts, prediction_targets)
        log.info("Logprob = %.4g" % log_prob)
        gen_log_prob = self.network.generalization_log_prob(prediction_contexts, prediction_targets)
        log.info("Generalization logprob = %.4g" % gen_log_prob)
        identity_ratio = identity_ratio_fn(validation_set)
        log.info("Identity ratio = %.4g" % identity_ratio)

        # Keep a copy of the best weights so far
        best_weights = best_iter = best_val_cost = None
        if validation_set is not None:
            best_weights = self.network.get_weights()
            best_iter = -1
            best_val_cost = val_cost

        below_threshold_its = 0

        for i in range(iterations):
            # Shuffle the training data between iterations, as one should with SGD
            shuffle = numpy.random.permutation(xs.shape[0])
            xs[:] = xs[shuffle]

            err = 0.0
            if num_batches > 1:
                for batch in range(num_batches):
                    # Update the model with this batch's data
                    batch_err = train_fn(xs[batch*batch_size:(batch+1)*batch_size],
                                         learning_rate=learning_rate,
                                         regularization=regularization)
                    err += batch_err

                    if batch_callback is not None:
                        batch_callback(batch, num_batches, batch_err)
            else:
                # Batch training: no need to loop
                ### Always perform one batch iteration to start with to get us into a good part of the space
                train_fn(xs, learning_rate=learning_rate, regularization=regularization)

            # Go back and compute training cost
            training_cost = cost_fn(xs)
            training_costs.append(training_cost)

            if validation_set is not None:
                # Compute the cost function on the validation set
                val_cost = cost_fn(validation_set)
                val_costs.append(val_cost)
                if val_cost <= best_val_cost:
                    # We assume that, if the validation error remains the same, it's better to use the new set of
                    # weights (with, presumably, a better training error)
                    if val_cost == best_val_cost:
                        log.info("Same validation cost: %.4f, using new weights" % val_cost)
                    else:
                        log.info("New best validation cost: %.4f" % val_cost)
                    # Update our best estimate
                    best_weights = self.network.get_weights()
                    best_iter = i
                    best_val_cost = val_cost
                if val_cost >= best_val_cost and i - best_iter >= stopping_iterations:
                    # We've gone on long enough without improving validation error
                    # Time to call a halt and use the best validation error we got
                    log.info("Stopping after %d iterations of increasing validation cost" %
                             stopping_iterations)
                    break

            log.info("COMPLETED ITERATION %d: training cost=%.5f, val cost=%.5f" %
                     (i, training_cost, val_cost))

            if cost_plot_filename:
                # Plot the cost function as we train
                # Skip the first costs, as they're usually so much higher than others that the rest is indistinguishable
                columns = [(training_costs[1:], "Train cost")]
                if validation_set is not None:
                    columns.append((val_costs[1:], "Val cost"))
                ax = plot_costs(None, *columns)
                # Add a line at the most recent best val cost
                ax.axvline(float(best_iter), color="b")
                ax.text(float(best_iter+1)+0.1, best_val_cost*1.1, "Best val cost", color="b")
                plt.savefig(cost_plot_filename)

            f_score, precision, recall, f_score_classes = self.compute_f_scores(validation_set)
            log.info("Validation f-score: %.4f%% (mean over %d classes), P=%.4f%%, R=%.4f%%" %
                     (f_score * 100.0, f_score_classes, precision * 100.0, recall * 100.0))
            #log_prob = self.network.prediction_log_prob(prediction_contexts, prediction_targets)
            #log.info("Prediction logprob = %.4g" % log_prob)
            gen_log_prob = self.network.generalization_log_prob(prediction_contexts, prediction_targets)
            log.info("Generalization logprob = %.4g" % gen_log_prob)
            identity_ratio = identity_ratio_fn(validation_set)
            log.info("Validation identity ratio = %.4g" % identity_ratio)

            if iteration_callback is not None:
                # Not computing training error at the moment
                iteration_callback(i, training_cost, val_cost, 0.0, best_iter)

            # Check the proportional change between this iteration's training cost and the last
            if len(training_costs) > 2:
                training_cost_prop_change = abs((training_costs[-2] - training_costs[-1]) / training_costs[-2])
                if training_cost_prop_change < training_cost_prop_change_threshold:
                    # Very small change in training cost - maybe we've converged
                    below_threshold_its += 1
                    if below_threshold_its >= 5:
                        # We've had enough iterations with very small changes: we've converged
                        log.info("Proportional change in training cost (%g) below %g for five successive iterations: "
                                 "converged" % (training_cost_prop_change, training_cost_prop_change_threshold))
                        break
                    else:
                        log.info("Proportional change in training cost (%g) below %g for %d successive iterations: "
                                 "waiting until it's been low for five iterations" %
                                 (training_cost_prop_change, training_cost_prop_change_threshold, below_threshold_its))
                else:
                    # Reset the below threshold counter
                    below_threshold_its = 0

        if best_weights is not None:
            # Use the weights that gave us the best error on the validation set
            self.network.set_weights(best_weights)


class DenoisingAutoencoderIterableTrainer(object):
    """
    Train without loading all the data into memory at once.

    """
    def __init__(self, network):
        self.network = network
        self.learning_rate = T.scalar("learning_rate")
        self.regularization = T.scalar("regularization")

        # F-score computations
        f_score_fn = network.get_f_scores()
        self.f_scores_fn = theano.function(
            inputs=[network.x],
            outputs=f_score_fn,
        )

    def compute_f_scores(self, xs):
        f_scores, precisions, recalls = self.f_scores_fn(xs)

        f_scores = f_scores[numpy.where(~numpy.isnan(f_scores))]
        precisions = precisions[numpy.where(~numpy.isnan(precisions))]
        recalls = recalls[numpy.where(~numpy.isnan(recalls))]

        return numpy.mean(f_scores), numpy.mean(precisions), numpy.mean(recalls), f_scores.shape[0]

    def train(self, batch_iterator, iterations=10000, iteration_callback=None,
              validation_set=None, stopping_iterations=10, log=None,
              cost_plot_filename=None, training_cost_prop_change_threshold=0.0005, learning_rate=0.1,
              regularization=0., class_weights_vector=None, corruption_level=0., continuous_corruption=False,
              loss="xent"):
        """
        Train on data stored in Theano tensors. Uses minibatch training.

        batch_iterator should be a repeatable iterator producing batches.

        iteration_callback is called after each iteration with args (iteration, error array).

        If a validation set (matrix) is given, it is used to compute an error after each iteration
        and to enforce a stopping criterion. The algorithm will terminate if it goes stopping_iterations
        iterations without an improvement in validation error.

        If compute_error_frequency > 1 (default=5), this number of iterations are performed between each time
        the error is computed on the training set.

        The algorithm will assume it has converged and stop early if the proportional change between successive
        training costs drops below training_cost_prop_change_threshold for five iterations in a row.

        Uses L2 regularization.

        """
        if log is None:
            log = get_console_logger("Autoencoder train")

        log.info("Training params: learning rate=%s, noise ratio=%.1f%% (%s), regularization=%s" %
                 (learning_rate, corruption_level * 100.0,
                  "continuous corruption" if continuous_corruption else "zeroing corruption",
                  regularization))
        log.info("Training with SGD")

        ######## Compile functions
        # Prepare cost/update functions for training
        cost, updates = self.network.get_cost_updates(self.learning_rate, self.regularization,
                                                      class_cost_weights=class_weights_vector,
                                                      corruption_level=corruption_level,
                                                      continuous_corruption=continuous_corruption,
                                                      loss=loss)
        # Prepare training functions
        cost_fn = theano.function(
            inputs=[self.network.x, Param(self.regularization, default=0.0)],
            outputs=cost,
        )
        train_fn = theano.function(
            inputs=[
                self.network.x,
                Param(self.learning_rate, default=0.1),
                Param(self.regularization, default=0.0)
            ],
            outputs=cost,
            updates=updates,
        )
        # Prepare a function to test how close to the identity function the learned mapping is
        # A lower value indicates that it's generalizing more (though not necessarily better)
        identity_ratio = T.mean(T.sum(self.network.get_prediction_dist() * (self.network.x > 0), axis=1))
        identity_ratio_fn = theano.function(
            inputs=[self.network.x],
            outputs=identity_ratio
        )
        ###########

        # Keep a record of costs, so we can plot them
        val_costs = []
        training_costs = []

        # Keep a copy of the best weights so far
        val_cost = 0.
        best_weights = best_iter = best_val_cost = None
        if validation_set is not None:
            best_weights = self.network.get_weights()
            best_iter = -1
            best_val_cost = cost_fn(validation_set)

            log.info("Computing initial validation scores")
            f_score, precision, recall, f_score_classes = self.compute_f_scores(validation_set)
            log.info("F-score: %.4f%% (mean over %d classes), P=%.4f%%, R=%.4f%%" %
                     (f_score * 100.0, f_score_classes, precision * 100.0, recall * 100.0))
            identity_ratio = identity_ratio_fn(validation_set)
            log.info("Identity ratio = %.4g" % identity_ratio)

        below_threshold_its = 0

        for i in range(iterations):
            err = 0.0
            batch_num = 0
            for batch_num, batch in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should with SGD
                # Just shuffle within batches
                shuffle = numpy.random.permutation(batch.shape[0])
                batch[:] = batch[shuffle]

                # Update the model with this batch's data
                err += train_fn(batch,
                                learning_rate=learning_rate,
                                regularization=regularization)

            training_costs.append(err / batch_num)

            if validation_set is not None:
                # Compute the cost function on the validation set
                val_cost = cost_fn(validation_set) / validation_set.shape[0]
                val_costs.append(val_cost)
                if val_cost <= best_val_cost:
                    # We assume that, if the validation error remains the same, it's better to use the new set of
                    # weights (with, presumably, a better training error)
                    if val_cost == best_val_cost:
                        log.info("Same validation cost: %.4f, using new weights" % val_cost)
                    else:
                        log.info("New best validation cost: %.4f" % val_cost)
                    # Update our best estimate
                    best_weights = self.network.get_weights()
                    best_iter = i
                    best_val_cost = val_cost
                if val_cost >= best_val_cost and i - best_iter >= stopping_iterations:
                    # We've gone on long enough without improving validation error
                    # Time to call a halt and use the best validation error we got
                    log.info("Stopping after %d iterations of increasing validation cost" %
                             stopping_iterations)
                    break

            log.info("COMPLETED ITERATION %d: training cost=%.5g, val cost=%.5g" %
                     (i, training_costs[-1], val_cost))

            if cost_plot_filename:
                # Plot the cost function as we train
                # Skip the first costs, as they're usually so much higher than others that the rest is indistinguishable
                columns = [(training_costs[1:], "Train cost")]
                if validation_set is not None:
                    columns.append((val_costs[1:], "Val cost"))
                ax = plot_costs(None, *columns)
                # Add a line at the most recent best val cost
                ax.axvline(float(best_iter), color="b")
                ax.text(float(best_iter+1)+0.1, best_val_cost*1.1, "Best val cost", color="b")
                plt.savefig(cost_plot_filename)

            if validation_set is not None:
                f_score, precision, recall, f_score_classes = self.compute_f_scores(validation_set)
                log.info("Validation f-score: %.4f%% (mean over %d classes), P=%.4f%%, R=%.4f%%" %
                         (f_score * 100.0, f_score_classes, precision * 100.0, recall * 100.0))
                identity_ratio = identity_ratio_fn(validation_set)
                log.info("Validation identity ratio = %.4g" % identity_ratio)

            if iteration_callback is not None:
                # Not computing training error at the moment
                iteration_callback(i, training_costs[-1], val_cost, 0.0, best_iter)

            # Check the proportional change between this iteration's training cost and the last
            if len(training_costs) > 2:
                training_cost_prop_change = abs((training_costs[-2] - training_costs[-1]) / training_costs[-2])
                if training_cost_prop_change < training_cost_prop_change_threshold:
                    # Very small change in training cost - maybe we've converged
                    below_threshold_its += 1
                    if below_threshold_its >= 5:
                        # We've had enough iterations with very small changes: we've converged
                        log.info("Proportional change in training cost (%g) below %g for five successive iterations: "
                                 "converged" % (training_cost_prop_change, training_cost_prop_change_threshold))
                        break
                    else:
                        log.info("Proportional change in training cost (%g) below %g for %d successive iterations: "
                                 "waiting until it's been low for five iterations" %
                                 (training_cost_prop_change, training_cost_prop_change_threshold, below_threshold_its))
                else:
                    # Reset the below threshold counter
                    below_threshold_its = 0

        if best_weights is not None:
            # Use the weights that gave us the best error on the validation set
            self.network.set_weights(best_weights)


class DenoisingAutoencoder(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    """
    TRAINER = DenoisingAutoencoderTrainer

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

        self._probs_fn = theano.function(
            inputs=[self.x],
            outputs=self.get_prediction_dist(),
        )

        self.hidden_layer = self.get_hidden_values(self.x)
        self._hidden_fn = theano.function(
            inputs=[self.x],
            outputs=self.hidden_layer,
        )

        self._b_copy = None
        self._b_prime_copy = None

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
        return self._hidden_fn(xs)

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
        # basilwang 2020-10-07 cast bool to int32
        # true_pos = T.sum(prediction & self.x_as_int, axis=0)
        true_pos = T.sum(T.cast(prediction,'int32') & self.x_as_int, axis=0)
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

    def get_cost(self, regularization=0.0, class_cost_weights=None, corruption_level=0., continuous_corruption=False,
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
        cost = T.mean(L) + regularization * T.mean(self.W ** 2.)
        return cost

    def get_cost_updates(self, learning_rate, regularization, class_cost_weights=None,
                         corruption_level=0., continuous_corruption=False, loss="xent"):
        """ This function computes the cost and the updates for one training
        step of the dA

        """
        cost = self.get_cost(regularization, class_cost_weights=class_cost_weights,
                             corruption_level=corruption_level, continuous_corruption=continuous_corruption,
                             loss=loss)

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
