from itertools import izip
import warnings
import theano
import numpy
from theano import tensor as T
from theano import Param
from whim_common.utils.logging import get_console_logger

try:
    from whim_common.utils.plotting import plot_costs
except ImportError:
    # Plotting will not be available, as pyplot isn't installed
    plot_costs = None


class FeedforwardNetworkTrainer(object):
    def __init__(self, network, optimization="sgd", loss="xent", input_var=None, extra_update_params=[],
                 extra_reg_params=[]):
        """
        optimization selects the type of optimization algorithm used. The default, 'sgd', is
        standard stochastic gradient descent. Currently, the only alternative is 'adadelta',
        which implements AdaDelta updates.

        loss is "xent" or "l2".

        extra_update_params allows you to specify other parameters that should be updated during
        training. They must, of course, feature in expression that computes the cost function.

        Likewise, extra_reg_params allows you to include extra parameters in the L2 regularization
        term. They should each be a 1D vector.

        """
        self.network = network
        if input_var is not None:
            x = input_var
        else:
            # Take the network's input as input to the training functions
            x = network.x
        # Create a target variable, of the same rank and type as the hidden layer
        # Special case for where the last layer has just a single unit: don't want y to need to be (M,1), just a vector
        if self.network.layer_sizes[-1] == 1:
            y = T.tensor(network.hidden_layer.dtype, (False,), name="y")
            # For computing the cost, add an extra dimension so the result is (M,1), not (M,)
            label_for_cost = y.dimshuffle(0, "x")
        else:
            y = label_for_cost = T.tensor(network.hidden_layer.dtype, network.hidden_layer.broadcastable, name="y")

        # Training params
        self.learning_rate = T.scalar("learning_rate")
        self.reg_coef = T.scalar("reg_coef")
        # Needed for AdaDelta
        self.decay = T.scalar("decay")
        self.optimization = optimization

        # Build cost function
        reg = self.network.get_l2_regularization(extra_params=extra_update_params)
        cost = self.network.get_cost(label_for_cost, loss=loss)
        cost_with_reg = cost + self.reg_coef * reg

        parameters = network.params + extra_update_params
        # Compute the gradient of the cost wrt the parameters
        gradients = [T.grad(cost_with_reg, param) for param in parameters]

        if optimization == "adadelta":
            # AdaDelta updates, based on Shawn Tan's implementation:
            #   https://blog.wtf.sg/2014/08/28/implementing-adadelta/
            extra_params = [Param(self.learning_rate, default=1e-6),
                            Param(self.decay, default=0.95)]

            # Store intermediate updates
            gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape)) for p in parameters]
            deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape)) for p in parameters]

            # Calculates the new "average" delta for the next iteration
            gradients_sq_new = [self.decay*g_sq + (1-self.decay)*(g**2) for g_sq, g in izip(gradients_sq, gradients)]

            # Calculates the step in direction
            # The square root is an approximation to getting the RMS for the average value
            deltas = [(T.sqrt(d_sq+self.learning_rate)/T.sqrt(g_sq+self.learning_rate))*grad
                      for d_sq, g_sq, grad in izip(deltas_sq, gradients_sq_new, gradients)]

            # calculates the new "average" deltas for the next step.
            deltas_sq_new = [self.decay*d_sq + (1-self.decay)*(d**2) for d_sq, d in izip(deltas_sq, deltas)]

            # Prepare the updates list
            updates = (
                # Update the squared gradients
                zip(gradients_sq, gradients_sq_new) +
                # Update the squared deltas
                zip(deltas_sq,deltas_sq_new) +
                # Update the model's actual parameters
                [(param, param - delta) for (param, delta) in izip(parameters, deltas)]
            )
        else:
            # Standard SGD updates
            extra_params = [Param(self.learning_rate, default=0.1)]
            updates = [(param, param - self.learning_rate * grad) for (param, grad) in zip(parameters, gradients)]

        # Compile
        self._train_fn = theano.function(
            inputs=[
                x, y,
                Param(self.reg_coef, default=0.01),
            ] + extra_params,
            outputs=T.mean(cost),
            updates=updates,
        )
        self._cost_fn = theano.function(
            inputs=[
                x, y,
            ],
            outputs=cost,
        )

    def get_class_counts(self, ys):
        return numpy.bincount(ys, minlength=self.network.num_classes)

    def train(self, batch_iterator, iterations=10000, iteration_callback=None, learning_rate=None, regularization=None,
              batch_callback=None, validation_set=None, stopping_iterations=10, log=None,
              cost_plot_filename=None, training_cost_prop_change_threshold=None):
        """
        Train on data stored in Theano tensors. Uses minibatch training.

        The input is given as an iterator over batches that should produce (x, y) pairs.

        E.g.
        xs = rng.randn(N, num_features)
        ys = rng.randint(size=N, low=0, high=2)

        iteration_callback is called after each iteration with args (iteration, error array).

        If a validation set (xs, ys) is given, it is used to compute an error after each iteration
        and to enforce a stopping criterion. The algorithm will terminate if it goes stopping_iterations
        iterations without an improvement in validation error.

        Updates for each target class can be weighted by giving a vector class_weights. Alternatively,
        give the string 'freq' to weight them by inverse class frequency, or leave as None to apply
        no weighting.

        If compute_error_frequency > 1 (default=5), this number of iterations are performed between each time
        the error is computed on the training set.

        The algorithm will assume it has converged and stop early if the proportional change between successive
        training costs drops below training_cost_prop_change_threshold for five iterations in a row.
        If threshold is given as None, this stopping condition will not be used.

        """
        if log is None:
            log = get_console_logger("MLP train")

        if plot_costs is None and cost_plot_filename is not None:
            warnings.warn("disabling plotting, since matplotlib couldn't be loaded")
            cost_plot_filename = None
        elif cost_plot_filename is not None:
            log.info("Plotting costs to %s" % cost_plot_filename)

        kwargs = {}
        if learning_rate is not None:
            kwargs["learning_rate"] = learning_rate
        if regularization is not None:
            kwargs["reg_coef"] = regularization
        log.info("Training params: learning rate=%s, reg coef=%s, algorithm=%s" %
                 (learning_rate, regularization, self.optimization))

        # Keep a record of costs, so we can plot them
        val_costs = []
        training_costs = []

        # Compute costs using the initialized network
        initial_batch_costs = [self.compute_cost(xs, ys) for (xs, ys) in batch_iterator]
        training_cost = sum(initial_batch_costs) / len(initial_batch_costs)
        log.info("Initial training cost: %g" % training_cost)
        training_costs.append(training_cost)
        if validation_set is not None:
            val_cost = self.compute_cost(validation_set[0], validation_set[1])
            val_costs.append(val_cost)
        else:
            val_cost = None
        log.info("Training on %d batches" % len(initial_batch_costs))

        # Keep a copy of the best weights so far
        best_weights = best_iter = best_val_cost = None
        if validation_set is not None:
            best_weights = self.network.get_weights()
            best_iter = -1
            best_val_cost = val_cost

        below_threshold_its = 0

        for i in range(iterations):
            err = 0.0
            batch_num = 0
            for batch_num, (xs, ys) in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should with SGD
                # We only do it within batches
                shuffle = numpy.random.permutation(xs.shape[0])
                xs[:] = xs[shuffle]
                ys[:] = ys[shuffle]
                # Update the model with this batch's data
                batch_err = self._train_fn(xs, ys, **kwargs)
                err += batch_err

                if batch_callback is not None:
                    batch_callback(batch_num, batch_err)

            # Go back and compute training cost
            training_cost = err / batch_num
            training_costs.append(training_cost)

            if validation_set is not None:
                # Compute the cost function on the validation set
                val_cost = self.compute_cost(validation_set[0], validation_set[1])
                val_costs.append(val_cost)
                if val_cost <= best_val_cost:
                    # We assume that, if the validation error remains the same, it's better to use the new set of
                    # weights (with, presumably, a better training error)
                    # Update our best estimate
                    best_weights = self.network.get_weights()
                    best_iter = i
                    best_val_cost = val_cost

                if i - best_iter >= stopping_iterations:
                    # We've gone on long enough without improving validation error
                    # Time to call a halt and use the best validation error we got
                    log.info("Stopping after %d iterations without improving validation cost" %
                             stopping_iterations)
                    break

                log.info("Completed iteration %d, training cost=%.5f, val cost=%.5f" % (i, training_cost, val_cost))

                if best_iter < i:
                    log.info("No improvement in validation cost")
            else:
                log.info("Completed iteration %d, training cost=%.5f" % (i, training_cost))

            if cost_plot_filename:
                # Plot the cost function as we train
                # Training cost is usually so high on the first iteration that it makes it impossible to see others
                columns = [(training_costs[1:], "Train cost")]
                if validation_set is not None:
                    columns.append((val_costs[1:], "Val cost"))
                ax, fig = plot_costs(None, *columns, return_figure=True)
                if best_iter is not None:
                    # Add a line at the most recent best val cost
                    ax.axvline(float(best_iter+1), color="b")
                    ax.text(float(best_iter+1)+0.1, best_val_cost*1.1, "Best val cost", color="b")
                # Write out to a file
                from matplotlib import pyplot as plt
                plt.savefig(cost_plot_filename)
                plt.close(fig)

            if iteration_callback is not None:
                iteration_callback(i, training_cost, val_cost, best_iter)

            # Check the proportional change between this iteration's training cost and the last
            if len(training_costs) > 2 and training_cost_prop_change_threshold is not None:
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
            # If val set wasn't given, the network just has the latest weights
            self.network.set_weights(best_weights)

    def compute_cost(self, xs, ys, **kwargs):
        return self._cost_fn(xs, ys, **kwargs)

