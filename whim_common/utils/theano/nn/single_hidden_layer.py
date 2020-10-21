from itertools import izip
import math
import numpy
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.printing
from theano import Param
from whim_common.utils.logging import get_console_logger
from whim_common.utils.plotting import plot_costs
from matplotlib import pyplot as plt
from whim_common.utils.probability.sample import balanced_array_sample


class SingleLayerNetworkTrainer(object):
    def __init__(self, network, regularize_bias=False, optimization="sgd"):
        """
        optimization selects the type of optimization algorithm used. The default, 'sgd', is
        standard stochastic gradient descent. Currently, the only alternative is 'adadelta',
        which implements AdaDelta updates.

        """
        self.network = network
        x, y = network.x, network.y

        # Training params
        self.learning_rate = T.scalar("learning_rate")
        self.reg_coef = T.scalar("reg_coef")
        self.class_weights = T.vector("class_weights", dtype="float64")
        # Needed for AdaDelta
        self.decay = T.scalar("decay")

        self.optimization = optimization

        # Cross-entropy loss function
        log_probs = T.log(network.class_probs[T.arange(y.shape[0]), y])
        xent = -T.mean(log_probs * self.class_weights[y])
        #xent = -self.network._mean_per_class_target_log_prob
        # The cost to minimize, including L2 regularization
        cost = xent + self.reg_coef * ((network.w0 ** 2.).mean() + (network.w1 ** 2.).mean())
        if regularize_bias:
            cost += self.reg_coef * ((network.b0 ** 2.).mean() + (network.b1 ** 2.).mean())

        parameters = [network.w0, network.w1, network.b0, network.b1]
        # Compute the gradient of the cost wrt the parameters
        gradients = [T.grad(cost, param) for param in parameters]

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
                Param(self.class_weights, default=numpy.ones(network.num_classes, dtype=numpy.float64)),
            ] + extra_params,
            outputs=T.sum(log_probs),
            updates=updates,
            givens=[(network.output_bias, 1), (network.hidden_bias, 1)],  # Bias enabled for training
            #on_unused_input="warn",
        )
        self._cost_fn = theano.function(
            inputs=[x, y,
                    Param(self.class_weights, default=numpy.ones(network.num_classes, dtype=numpy.float64)),
                    Param(self.reg_coef, default=0.01)],
            outputs=cost,
            givens=[(network.output_bias, 1), (network.hidden_bias, 1)],
            mode='FAST_RUN',
            #on_unused_input="warn",
        )
        self._costs = theano.function(
            inputs=[x, y, Param(self.class_weights, default=numpy.ones(network.num_classes, dtype=numpy.float64))],
            outputs=xent,
            givens=[(network.output_bias, 1), (network.hidden_bias, 1)],
            #on_unused_input="warn",
        )

    def get_class_counts(self, ys):
        return numpy.bincount(ys, minlength=self.network.num_classes)

    def train(self, xs, ys, iterations=10000, iteration_callback=None, learning_rate=None, regularization=None,
              batch_size=20, batch_callback=None, validation_set=None, stopping_iterations=10, log=None,
              class_weights=None, cost_plot_filename=None,
              training_cost_prop_change_threshold=None, undersample=None,
              print_predictions=False):
        """
        Train on data stored in Theano tensors. Uses minibatch training.

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

        If undersample is given it should be a float. The training data will be randomly undersampled to produce
        a set in which the expected number of instances of each class is undersample*min_freq, where min_freq
        is the number of instances of the least common observed class. A value of 1.0 will produce a roughly
        balanced set. Every class that is observed at all will be included at least once. The sampling is
        performed once at the beginning of training.

        """
        if log is None:
            log = get_console_logger("MLP train")

        if cost_plot_filename is not None:
            _fname, __, _ext = cost_plot_filename.rpartition(".")
            balanced_cost_plot_filename = "%s_balanced.%s" % (_fname, _ext)
            log.info("Outputting balanced costs to: %s" % balanced_cost_plot_filename)
        else:
            balanced_cost_plot_filename = None

        kwargs = {}
        cost_kwargs = {
            "reg_coef": 0.,   # Always compute the cost without regularization
        }
        if learning_rate is not None:
            kwargs["learning_rate"] = learning_rate
        if regularization is not None:
            kwargs["reg_coef"] = regularization
        log.info("Training params: learning rate=%s, reg coef=%s" % (learning_rate, regularization))
        log.info("Training with %s, batch size=%d" % (self.optimization, batch_size))
        if undersample is not None and undersample > 0.0:
            log.info("Undersampling the dataset with a ratio of %s" % undersample)

        # Work out how many batches to do
        if batch_size is None or batch_size == 0:
            num_batches = 1
        else:
            num_batches = xs.shape[0] / batch_size
            if xs.shape[0] % batch_size != 0:
                num_batches += 1

        if undersample is not None and undersample > 0.0:
            # Undersample the training data to produce a (more) balanced set
            balanced_indices = balanced_array_sample(ys, balance_ratio=undersample, min_inclusion=1)
            # Copy the data so we're not dealing with a view
            xs = numpy.copy(xs[balanced_indices])
            ys = numpy.copy(ys[balanced_indices])
            # Also sample the validation set similarly
            balanced_validation_indices = balanced_array_sample(validation_set[1],
                                                                balance_ratio=undersample,
                                                                min_inclusion=1)
            validation_set = (
                numpy.copy(validation_set[0][balanced_validation_indices]),
                numpy.copy(validation_set[1][balanced_validation_indices])
            )
            log.info("Sampled %d training and %d validation instances" % (xs.shape[0], validation_set[0].shape[0]))

        # Work out class weighting
        # Do this after undersampling: if both are used, we only want the weights to account for any imbalance
        #  left after undersampling
        if class_weights is not None:
            if class_weights == 'freq':
                # Use inverse frequency to weight class updates
                # This procedure is modelled directly on what liblinear does
                class_counts = self.get_class_counts(ys).astype(numpy.float64)
                # Replace zero-counts with 1s
                class_counts = numpy.maximum(class_counts, 1.0)
                class_weights = 1.0 / class_counts
                class_weights *= self.network.num_classes / class_weights.sum()
                log.info("Inverse-frequency class weighting")
            elif class_weights == 'log':
                # Use a different scheme, inversely proportional to the log of the class frequencies
                class_counts = self.get_class_counts(ys).astype(numpy.float64)
                class_counts = numpy.maximum(class_counts, 1.0)
                class_weights = 1.0 / (1.0 + numpy.log(class_counts))
                class_weights *= self.network.num_classes / class_weights.sum()
                log.info("Log-inverse-frequency class weighting")
            else:
                log.info("Custom vector class weighting")
            kwargs["class_weights"] = class_weights
            cost_kwargs["class_weights"] = class_weights
        else:
            log.info("No class weighting")

        # Keep a record of costs, so we can plot them
        val_costs = []
        training_costs = []
        # The costs using the balanced metric
        bal_val_costs = []
        bal_training_costs = []

        # Compute costs using the initialized network
        training_cost = self.compute_cost(xs, ys, **cost_kwargs)
        training_costs.append(training_cost)
        if validation_set is not None:
            val_cost = self.compute_cost(validation_set[0], validation_set[1], **cost_kwargs)
            val_costs.append(val_cost)
        else:
            val_cost = None

        log.info("Computing initial validation set metrics:")
        class_accuracies = self.network.per_class_accuracy(validation_set[0], validation_set[1])
        class_accuracies = class_accuracies[numpy.where(numpy.logical_not(numpy.isnan(class_accuracies)))]
        mean_class_accuracy = class_accuracies.mean()
        log.info("Per-class accuracy: %.4f%% (mean over %d classes)" %
                 (mean_class_accuracy, class_accuracies.shape[0]))
        # Also compute mean log prob of targets over val set
        mean_log_prob = self.network.mean_log_prob(validation_set[0], validation_set[1])
        log.info("Mean target log prob: %.4f" % mean_log_prob)
        mean_per_class_log_prob = self.network.mean_per_class_target_log_prob(validation_set[0], validation_set[1])
        log.info("Mean per-class mean target log prob: %.4f" % mean_per_class_log_prob)

        # Keep a copy of the best weights so far
        best_weights = best_iter = best_val_cost = None
        if validation_set is not None:
            best_weights = self.network.get_weights()
            best_iter = -1
            best_val_cost = val_cost

        below_threshold_its = 0

        # Count the instances we're learning from to give an idea of how hard a time the model's got
        training_class_counts = numpy.bincount(ys)
        training_class_counts = training_class_counts[training_class_counts.nonzero()]
        log.info("Training instances per class: min=%d, max=%d (%d unseen classes)" %
                 (int(training_class_counts.min()),
                  int(training_class_counts.max()),
                  self.network.num_classes - training_class_counts.shape[0]))

        for i in range(iterations):
            # Shuffle the training data between iterations, as one should with SGD
            shuffle = numpy.random.permutation(xs.shape[0])
            xs[:] = xs[shuffle]
            ys[:] = ys[shuffle]

            err = 0.0
            if num_batches > 1:
                for batch in range(num_batches):
                    # Update the model with this batch's data
                    batch_err = self._train_fn(
                        xs[batch*batch_size:(batch+1)*batch_size],
                        ys[batch*batch_size:(batch+1)*batch_size],
                        **kwargs
                    )
                    err += batch_err

                    if batch_callback is not None:
                        batch_callback(batch, num_batches, batch_err)
            else:
                # Batch training: no need to loop
                err = self._train_fn(xs, ys, **kwargs)

            # Go back and compute training cost
            training_cost = self.compute_cost(xs, ys, **cost_kwargs)
            training_costs.append(training_cost)
            # Training set error
            train_error = self.network.error(xs, ys)
            bal_training_costs.append(-self.network.mean_per_class_target_log_prob(xs, ys))

            if validation_set is not None:
                if print_predictions:
                    # Perform some predictions on a random sample of the val set
                    for randind in numpy.random.randint(validation_set[0].shape[0], size=5):
                        # Get the network's predictions
                        predictions = self.network.predict(validation_set[0][None, randind, :])
                        predictions = predictions[0, None]
                        log.info(
                            "Input: %s. Predictions: %s" % (
                                list(numpy.where(validation_set[0][randind] > 0)[0]),
                                list(predictions)
                            )
                        )
                # Compute the cost function on the validation set
                val_cost = self.compute_cost(validation_set[0], validation_set[1], **cost_kwargs)
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

                # Compute various metrics
                # Per-class accuracy on val set
                class_accuracies = self.network.per_class_accuracy(validation_set[0], validation_set[1])
                class_accuracies = class_accuracies[numpy.where(numpy.logical_not(numpy.isnan(class_accuracies)))]
                mean_class_accuracy = class_accuracies.mean()
                # Mean log prob of targets over val set
                mean_log_prob = self.network.mean_log_prob(validation_set[0], validation_set[1])
                mean_per_class_log_prob = self.network.mean_per_class_target_log_prob(validation_set[0],
                                                                                      validation_set[1])
                log.info("Completed iteration %d, training cost=%.5f, val cost=%.5f, training error=%.2f%%, "
                         "per-class accuracy: %.4f%%, mean tgt logprob: %.4f, per-class tgt logprob: %.4f" %
                         (i, training_cost, val_cost, train_error * 100.0, mean_class_accuracy, mean_log_prob,
                          mean_per_class_log_prob))
                bal_val_costs.append(-mean_per_class_log_prob)

                if best_iter < i:
                    log.info("No improvement in validation cost")
            else:
                log.info("Completed iteration %d, training cost=%.5f, training error=%.2f%%" %
                         (i, training_cost, train_error * 100.0))

            if cost_plot_filename:
                # Plot the cost function as we train
                columns = [(training_costs, "Train cost")]
                if validation_set is not None:
                    columns.append((val_costs, "Val cost"))
                ax = plot_costs(None, *columns)
                # Add a line at the most recent best val cost
                ax.axvline(float(best_iter+1), color="b")
                ax.text(float(best_iter+1)+0.1, best_val_cost*1.1, "Best val cost", color="b")
                plt.savefig(cost_plot_filename)

                bal_columns = [(bal_training_costs, "Train cost (balanced)")]
                if validation_set is not None:
                    bal_columns.append((bal_val_costs, "Val cost (balanced)"))
                plot_costs(balanced_cost_plot_filename, *bal_columns)

            if iteration_callback is not None:
                iteration_callback(i, training_cost, val_cost, train_error, best_iter)

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
            self.network.set_weights(best_weights)

    def compute_cost(self, xs, ys, **kwargs):
        return self._cost_fn(xs, ys, **kwargs)


class SingleLayerNetwork(object):
    """
    Theano neural network implementation, loosely based on tutorial.

    """
    TRAINER = SingleLayerNetworkTrainer

    def __init__(self, num_features, num_classes, num_hidden_units=100, normalize_features=False, autoencoder=False,
                 hidden_activation_fn=None, initialization='glorot'):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_hidden_units = num_hidden_units
        self.normalize_features = normalize_features
        if autoencoder:
            raise NotImplementedError("don't use SingleLayerNetwork any more to train an autoencoder. It has "
                                      "its own implementation, which is better")

        if hidden_activation_fn is None:
            hidden_activation_fn = nnet.sigmoid

        # Set up Theano network for the model
        # Features: (m, num_features)
        self.x = T.matrix("x", dtype="float64")
        # Classes (one-hot): (m, num_classes)
        self.y = T.vector("y", dtype="int64")

        if normalize_features:
            # Divide feature vector by its Euclidean norm before using the values as inputs
            self.inputs = ifelse(T.gt(self.x.sum(), 0), self.x / T.sqrt((self.x ** 2).sum()), self.x)
        else:
            self.inputs = self.x

        if initialization == 'gaussian':
            # Weights and bias, randomly initialized
            self.w0 = theano.shared(numpy.random.randn(num_features, num_hidden_units), name="w0")
            self.b0 = theano.shared(numpy.random.randn(num_hidden_units), name="b0")
            self.w1 = theano.shared(numpy.random.randn(num_hidden_units, num_classes), name="w1")
            self.b1 = theano.shared(numpy.random.randn(num_classes), name="b1")
        elif initialization == 'glorot':
            # Use Glorot & Bengio's initialization scheme, where the range of random weights depends on the number of
            # hidden units in this and the previous layer
            unif_width0 = math.sqrt(6.) / math.sqrt(num_features + num_hidden_units)
            self.w0 = theano.shared(numpy.random.uniform(-unif_width0, unif_width0, (num_features, num_hidden_units)),
                                    name="w0")
            unif_width1 = math.sqrt(6.) / math.sqrt(num_hidden_units + num_classes)
            self.w1 = theano.shared(numpy.random.uniform(-unif_width1, unif_width1, (num_hidden_units, num_classes)),
                                    name="w1")
            # Initialize biases to 0
            self.b0 = theano.shared(numpy.zeros(num_hidden_units), name="b0")
            self.b1 = theano.shared(numpy.zeros(num_classes), name="b1")
        elif initialization == 'squashed-gaussian':
            # Similar to initializing with a normalized Gaussian, but squashes to std to 1/sqrt(input_nodes)
            std0 = 1. / math.sqrt(num_features)
            self.w0 = theano.shared(numpy.random.normal(0., std0, (num_features, num_hidden_units)), name="w0")
            std1 = 1. / math.sqrt(num_hidden_units)
            self.w1 = theano.shared(numpy.random.uniform(0., std1, (num_hidden_units, num_classes)), name="w1")
            # Initialize biases to 0
            self.b0 = theano.shared(numpy.zeros(num_hidden_units), name="b0")
            self.b1 = theano.shared(numpy.zeros(num_classes), name="b1")
        else:
            raise ValueError("unknown initialization type '%s'. Choose gaussian, squashed-gaussian or glorot" %
                             initialization)

        # Parameter
        self.output_bias = T.scalar("output_bias", dtype="int64")
        self.hidden_bias = T.scalar("hidden_bias", dtype="int64")

        # Construct Theano expression graph
        self.hidden_activation = T.dot(self.inputs, self.w0) + \
                                 ifelse(T.gt(self.hidden_bias, 0), self.b0, T.zeros(self.b0.shape))
        self.output_activation = T.dot(hidden_activation_fn(self.hidden_activation), self.w1) + \
                                 ifelse(T.gt(self.output_bias, 0), self.b1, T.zeros(self.b1.shape))
        # Softmax activations to get a probability distribution over the classes
        self.class_probs = nnet.softmax(self.output_activation)

        # The predicted class is that with highest activation (no need to do the softmax for this)
        self.prediction = T.argmax(self.output_activation, axis=1)
        error = T.mean(T.neq(self.prediction, self.y))

        # Compile
        self._predict_fn = theano.function(
            inputs=[self.x, Param(self.output_bias, default=1), Param(self.hidden_bias, default=1)],
            outputs=self.prediction
        )
        self._prob_fn = theano.function(
            inputs=[self.x, Param(self.output_bias, default=1), Param(self.hidden_bias, default=1)],
            outputs=self.class_probs
        )
        self._error_fn = theano.function(
            inputs=[self.x, self.y, Param(self.output_bias, default=1), Param(self.hidden_bias, default=1)],
            outputs=error,
        )
        self.hidden_fn = theano.function(
            inputs=[self.x, Param(self.hidden_bias, default=1)],
            outputs=hidden_activation_fn(self.hidden_activation),
        )

        one_hot_predictions = T.eye(num_classes, num_classes, dtype="int8")[self.prediction]
        one_hot_targets = T.eye(num_classes, num_classes, dtype="int8")[self.y]

        # Average log-prob of correct answer
        # More useful metric than accuracy, since we can see differences even where the right answer's not top
        mean_target_log_prob = T.mean(T.log(self.class_probs[T.arange(self.y.shape[0]), self.y]))
        self.mean_log_prob = theano.function(
            inputs=[self.x, self.y, Param(self.output_bias, default=1), Param(self.hidden_bias, default=1)],
            outputs=mean_target_log_prob
        )

        # Similar thing, but averaged within classes first, then across
        num_targets = one_hot_targets.sum(axis=0)
        has_targets = T.neq(num_targets, 0.)
        per_class_target_log_prob = T.switch(has_targets,
                                             T.sum(T.log(self.class_probs) * one_hot_targets, axis=0) / num_targets,
                                             0.)
        self._mean_per_class_target_log_prob = T.sum(per_class_target_log_prob) / T.sum(has_targets)
        self.mean_per_class_target_log_prob = theano.function(
            inputs=[self.x, self.y, Param(self.output_bias, default=1), Param(self.hidden_bias, default=1)],
            outputs=self._mean_per_class_target_log_prob
        )

        ##### F-score computation
        # True positives per output class
        true_pos = T.cast(T.sum(one_hot_predictions & one_hot_targets, axis=0), dtype="float64")
        # Positive targets per output class
        pos = T.cast(T.sum(one_hot_targets, axis=0), dtype="float64")
        # Predicted positives per output class
        predicted_pos = T.cast(T.sum(one_hot_predictions, axis=0), dtype="float64")

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
        self._precisions_fn = theano.function(
            inputs=[self.x, self.y],
            outputs=precisions,
            givens=[(self.output_bias, 1), (self.hidden_bias, 1)],
        )
        self._recalls_fn = theano.function(
            inputs=[self.x, self.y],
            outputs=recalls,
            givens=[(self.output_bias, 1), (self.hidden_bias, 1)],
        )
        self._f_score_fn = theano.function(
            inputs=[self.x, self.y],
            outputs=[f_scores, precisions, recalls],
            givens=[(self.output_bias, 1), (self.hidden_bias, 1)],
        )

    def predict(self, xs, disable_output_bias=False, disable_hidden_bias=False):
        output_bias = 0 if disable_output_bias else 1
        hidden_bias = 0 if disable_hidden_bias else 1
        return self._predict_fn(xs, output_bias=output_bias, hidden_bias=hidden_bias)

    def probs(self, xs, disable_output_bias=False, disable_hidden_bias=False):
        output_bias = 0 if disable_output_bias else 1
        hidden_bias = 0 if disable_hidden_bias else 1
        return self._prob_fn(xs, output_bias=output_bias, hidden_bias=hidden_bias)

    def error(self, xs, ys):
        return self._error_fn(xs, ys)

    def class_recalls(self, xs, ys):
        """
        Since the model returns one answer per context, this isn't really recall, but per-class
        accuracy.

        """
        return self._recalls_fn(xs, ys)

    def per_class_accuracy(self, xs, ys):
        return self.class_recalls(xs, ys)

    def class_precisions(self, xs, ys):
        return self._precisions_fn(xs, ys)

    def class_f_scores(self, xs, ys):
        return self._f_score_fn(xs, ys)[0]

    def mean_f_score(self, xs, ys):
        f_scores = self.class_f_scores(xs, ys)
        return f_scores[numpy.where(numpy.logical_not(numpy.isnan(f_scores)))].mean()

    def mean_f_score_precision_recall(self, xs, ys):
        f_scores, precisions, recalls = self._f_score_fn(xs, ys)
        return (f_scores[numpy.where(numpy.logical_not(numpy.isnan(f_scores)))].mean(),
                precisions[numpy.where(numpy.logical_not(numpy.isnan(precisions)))].mean(),
                recalls[numpy.where(numpy.logical_not(numpy.isnan(recalls)))].mean(),)

    def get_weights(self):
        """
        Return a copy of all the weight arrays in a tuple.

        """
        return (self.w0.get_value().copy(),
                self.b0.get_value().copy(),
                self.w1.get_value().copy(),
                self.b1.get_value().copy())

    def set_weights(self, weights):
        """
        Set all weights from a tuple, like that returned by get_weights().

        """
        self.w0.set_value(weights[0])
        self.b0.set_value(weights[1])
        self.w1.set_value(weights[2])
        self.b1.set_value(weights[3])

    ####################################
    #### Pickling for model storage ####

    def __getstate__(self):
        return {
            "weights0": self.w0.get_value(),
            "weights1": self.w1.get_value(),
            "bias0": self.b0.get_value(),
            "bias1": self.b1.get_value(),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "num_hidden_units": self.num_hidden_units,
            "normalize_features": self.normalize_features,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(state["num_features"], state["num_classes"], state["num_hidden_units"],
                      normalize_features=state.get("normalize_features", False))
        self.w0.set_value(state["weights0"])
        self.w1.set_value(state["weights1"])
        self.b0.set_value(state["bias0"])
        self.b1.set_value(state["bias1"])


def train_network(num_features, num_classes, num_hidden_units, xs, ys,
                  iterations=10000, iteration_callback=None, learning_rate=None, regularization=None):
    network = SingleLayerNetwork(num_features, num_classes, num_hidden_units)
    trainer = SingleLayerNetworkTrainer(network)
    trainer.train(xs, ys, iterations=iterations, iteration_callback=iteration_callback,
                  learning_rate=learning_rate, regularization=regularization)
    return network