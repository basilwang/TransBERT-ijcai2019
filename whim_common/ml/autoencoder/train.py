import numpy
from theano import tensor as T, Param
import theano
from whim_common.utils.logging import get_console_logger

try:
    from whim_common.utils.plotting import plot_costs
except ImportError:
    # Plotting will not be available, as pyplot isn't installed
    plot_costs = None


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
                from matplotlib import pyplot as plt
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


class StackedDenoisingAutoencoderIterableTrainer(object):
    """ Train a whole stacked autoencoder """
    def __init__(self, network, input=None):
        self.network = network
        self.learning_rate = T.scalar("learning_rate")
        self.regularization = T.scalar("regularization")
        if input is None:
            input = network.x
        self.input = input

    def train(self, batch_iterator, iterations=10000, iteration_callback=None,
              validation_set=None, stopping_iterations=10, log=None,
              cost_plot_filename=None, training_cost_prop_change_threshold=0.0005, learning_rate=0.1,
              regularization=0., class_weights_vector=None, corruption_level=0., continuous_corruption=False,
              loss="xent"):
        """
        See autoencoder trainer: uses the same training for each layer in turn, then rolls out and
        trains the whole thing together.

        """
        if log is None:
            log = get_console_logger("Autoencoder train")

        # Because the layers are all already properly stacked, when we get the cost/updates for a layer,
        # it's already a function of the original input, but only updates the layer itself
        for layer_num, layer in enumerate(self.network.layers):
            log.info("TRAINING LAYER %d" % layer_num)
            ## Compile functions
            # Prepare cost/update functions for training
            cost, updates = layer.get_cost_updates(self.learning_rate, self.regularization,
                                                   class_cost_weights=class_weights_vector,
                                                   corruption_level=corruption_level,
                                                   continuous_corruption=continuous_corruption,
                                                   loss=loss)
            # Prepare training functions
            # Note that these use the initial input, not the layer input
            cost_fn = theano.function(
                inputs=[self.input, Param(self.regularization, default=0.0)],
                outputs=cost,
            )
            train_fn = theano.function(
                inputs=[
                    self.input,
                    Param(self.learning_rate, default=0.1),
                    Param(self.regularization, default=0.0)
                ],
                outputs=cost,
                updates=updates,
            )
            # Prepare a function to test how close to the identity function the learned mapping is
            # A lower value indicates that it's generalizing more (though not necessarily better)
            identity_ratio = T.mean(T.sum(layer.get_prediction_dist() * (layer.x > 0), axis=1))
            identity_ratio_fn = theano.function(
                inputs=[self.input],
                outputs=identity_ratio
            )

            # Keep a record of costs, so we can plot them
            val_costs = []
            training_costs = []

            # Keep a copy of the best weights so far
            val_cost = 0.
            best_weights = best_iter = best_val_cost = None
            if validation_set is not None:
                best_weights = layer.get_weights()
                best_iter = -1
                best_val_cost = cost_fn(validation_set)

                log.info("Computing initial validation scores")
                identity_ratio = identity_ratio_fn(validation_set)
                log.info("Identity ratio = %.4g" % identity_ratio)

            log.info("Computing initial training cost")
            batch_costs = [cost_fn(batch) for batch in batch_iterator]
            initial_cost = sum(batch_costs) / len(batch_costs)
            log.info("Cost = %g (%d batches)" % (initial_cost, len(batch_costs)))

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
                        best_weights = layer.get_weights()
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
                else:
                    log.info("COMPLETED ITERATION %d: training cost=%.5g" %
                             (i, training_costs[-1]))

                if cost_plot_filename:
                    # Plot the cost function as we train
                    # Skip the first costs, as they're usually so much higher that the rest is indistinguishable
                    columns = [(training_costs[1:], "Train cost")]
                    if validation_set is not None:
                        columns.append((val_costs[1:], "Val cost"))
                    ax = plot_costs(None, *columns)
                    # Add a line at the most recent best val cost
                    ax.axvline(float(best_iter), color="b")
                    ax.text(float(best_iter+1)+0.1, best_val_cost*1.1, "Best val cost", color="b")
                    from matplotlib import pyplot as plt
                    plt.savefig(cost_plot_filename)

                if validation_set is not None:
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
                layer.set_weights(best_weights)
