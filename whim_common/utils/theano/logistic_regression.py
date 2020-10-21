import numpy
from scipy.optimize.optimize import fmin_bfgs
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano import Param
from whim_common.utils.plotting import plot_costs


class LogisticRegression(object):
    """
    Theano LR implementation, based on tutorial

    """
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes

        # Set up Theano network for the model
        # Features: (m, num_features)
        x = T.matrix("x")
        # Classes (one-hot): (m, num_classes)
        y = T.ivector("y")
        # Weights and bias, randomly initialized
        self.theta = theano.shared(
            value=numpy.zeros(
                num_features * num_classes,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )
        w = self.theta.reshape((num_features, num_classes))
        # Don't include bias: we put 1s in the input instead
        #b = theano.shared(numpy.zeros(num_classes), name="b")
        # Other training params
        self.reg_coef = T.scalar("reg")

        # Construct Theano expression graph
        activation = T.dot(x, w)
        # Softmax activations to get a probability distribution over the classes
        class_probs = nnet.softmax(activation)
        # The predicted class is that with highest activation (no need to do the softmax for this)
        prediction = T.argmax(activation, axis=1)
        # Cross-entropy loss function
        #xent = nnet.categorical_crossentropy(class_probs, y)
        xent = -T.mean(T.log(class_probs)[T.arange(y.shape[0]), y])
        # The cost to minimize, including L2 regularization
        cost = xent + self.reg_coef * (w[1:, :] ** 2).sum()
        # Compute the gradient of the cost
        self.gw = T.grad(cost, w)
        self.gtheta = T.grad(cost, self.theta)
        # Error in terms of hard predictions (accuracy)
        error = T.mean(T.neq(prediction, y))

        # Compile
        self._predict_fn = theano.function(inputs=[x], outputs=prediction)
        self._prob_fn = theano.function(inputs=[x], outputs=class_probs)
        self._cost_fn = theano.function(
            inputs=[x, y],
            outputs=xent,
        )
        self._cost_fn_reg = theano.function(
            inputs=[x, y, Param(self.reg_coef, default=0.01)],
            outputs=cost,
        )
        self._error_fn = theano.function(
            inputs=[x, y],
            outputs=error
        )

        self.w = w
        self.x = x
        self.y = y
        self._cost_without_reg = xent

    def train(self, xs, ys, iterations=10000, iteration_callback=None,
              validation_xs=None, validation_ys=None, validation_frequency=1, learning_rate=0.1, regularization=0.01,
              plot_errors=None, plot_cost=None):
        """
        Train on data stored in Theano tensors.

        E.g.
        xs = rng.randn(N, num_features)
        ys = rng.randint(size=N, low=0, high=2)

        iteration_callback is called after each iteration with args (iteration, error array).

        """
        learning_rate_var = T.scalar("alpha")
        # Compute the training function
        _train_fn = theano.function(
            inputs=[self.x, self.y, Param(learning_rate_var, default=0.1), Param(self.reg_coef, default=0.01)],
            outputs=self._cost_without_reg,
            updates=[(self.theta, self.theta - learning_rate_var * self.gtheta)],
        )

        best_validation_error = numpy.inf

        validation_errors = []
        training_errors = []
        costs = []

        for i in range(iterations):
                training_cost = _train_fn(xs, ys, alpha=learning_rate, reg=regularization)

                # Only evaluate on val set every validation_frequencyth iteration
                if validation_xs is not None and (i+1) % validation_frequency == 0:
                    # Compute accuracy on validation set
                    validation_error = self.error(validation_xs, validation_ys)
                    # Compute accuracy on training set
                    training_error = self.error(xs, ys)
                    # Compute how much we've improved on the previous best validation error
                    if validation_error < best_validation_error:
                        validation_improvement = 0.0
                    else:
                        validation_improvement = (validation_error - best_validation_error) / best_validation_error * 100.0
                        best_validation_error = validation_error
                else:
                    validation_error = None
                    validation_improvement = None
                    training_error = None

                if iteration_callback is not None:
                    iteration_callback(i, training_cost, training_error, validation_error, validation_improvement)

                # Plot some graphs
                if plot_cost:
                    costs.append(training_cost)
                    plot_costs(plot_cost, (costs, "training cost"))
                if plot_errors and validation_error is not None:
                    validation_errors.append(validation_error)
                    training_errors.append(training_error)
                    plot_costs(plot_errors,
                               (training_errors, "training set error"),
                               (validation_errors, "val set error"))

    def train_bfgs(self, xs, ys, iterations=10000, iteration_callback=None,
                   validation_xs=None, validation_ys=None, validation_frequency=1, regularization=0.01,
                   plot_errors=None):
        """
        Train on data stored in Theano tensors.

        E.g.
        xs = rng.randn(N, num_features)
        ys = rng.randint(size=N, low=0, high=2)

        iteration_callback is called after each iteration with args (iteration, error array).

        """
        compute_grad = theano.function(
            inputs=[self.x, self.y],
            outputs=self.gtheta,
            givens={self.reg_coef: regularization}
        )

        # Prepare cost function to optimize and its gradient (jacobian)
        def train_fn(theta):
            self.theta.set_value(theta, borrow=True)
            return self._cost_fn_reg(xs, ys, reg_coef=regularization)

        def train_fn_grad(theta):
            self.theta.set_value(theta, borrow=True)
            return compute_grad(xs, ys)

        # Prepare a callback for between iterations
        best_validation_error = [numpy.inf]
        iteration_counter = [0]
        validation_errors = []
        training_errors = []

        def callback(new_theta):
            # Update the parameters of the model
            self.theta.set_value(new_theta, borrow=True)

            # Only evaluate on val set every validation_frequencyth iteration
            if validation_xs is not None and (iteration_counter[0]+1) % validation_frequency == 0:
                # Compute accuracy on validation set
                validation_error = self.error(validation_xs, validation_ys)
                # Compute accuracy on training set
                training_error = self.error(xs, ys)
                # Compute how much we've improved on the previous best validation error
                if validation_error < best_validation_error[0]:
                    validation_improvement = 0.0
                else:
                    validation_improvement = (validation_error - best_validation_error[0]) / best_validation_error[0] * 100.0
                    best_validation_error[0] = validation_error

                # Plot some graphs
                if plot_errors and validation_error is not None:
                    validation_errors.append(validation_error)
                    training_errors.append(training_error)
                    plot_costs(plot_errors,
                               (training_errors, "training set error"),
                               (validation_errors, "val set error"))
            else:
                validation_error = training_error = validation_improvement = None

            if iteration_callback is not None:
                # TODO Compute training cost?
                iteration_callback(iteration_counter[0], 0.0, training_error, validation_error, validation_improvement)

            iteration_counter[0] += 1

        # Call scipy's BFGS optimization function
        fmin_bfgs(train_fn, self.theta.get_value(), fprime=train_fn_grad, callback=callback, disp=True, maxiter=iterations)
        # The result is now in the model's parameters, thanks to the callback

    def predict(self, xs):
        return self._predict_fn(xs)

    def probs(self, xs):
        return self._prob_fn(xs)

    def compute_cost(self, xs, ys):
        return self._cost_fn(xs, ys)

    def error(self, xs, ys):
        return self._error_fn(xs, ys)

    #############################################################
    #### Implement certain parts of scikit-learn's interface ####
    @property
    def classes_(self):
        return numpy.arange(self.num_classes)

    @property
    def coef_(self):
        return self.theta.get_value().reshape((self.num_features, self.num_classes)).T

    def predict_log_proba(self, xs):
        """
        Implement part of scikit-learn's interface.

        """
        return self._prob_fn(xs)

    ####################################
    #### Pickling for model storage ####

    def __getstate__(self):
        return {
            "weights": self.theta.get_value(),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
        }

    def __setstate__(self, state):
        # Initialize using constructor
        self.__init__(state["num_features"], state["num_classes"])
        theta = state["weights"]
        self.theta.set_value(theta)