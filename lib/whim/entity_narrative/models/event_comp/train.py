from __future__ import absolute_import
from itertools import islice, dropwhile
import os
import random
import math
import shutil

import numpy
import theano
from theano import tensor as T
try:
    from matplotlib import pyplot as plt
    from whim_common.utils.plotting import plot_costs
except ImportError:
    # Plotting will not be available, as pyplot isn't installed
    plt = None
    plot_costs = None

from whim.entity_narrative import ModelLoadError
from whim.entity_narrative.models.base.train import NarrativeChainModelTrainer, ModelTrainingError
from whim_common.utils.base import remove_duplicates
from whim_common.utils.files import GroupedIntListsWriter, GroupedIntListsReader, \
    GroupedIntTuplesReader
from whim_common.utils.logging import get_console_logger
from whim_common.utils.progress import ProgressBarIter
from whim_common.utils.shuffle import limited_shuffle
from whim_common.utils.theano.nn.autoencoder import DenoisingAutoencoderIterableTrainer



class EventCompositionTrainer(NarrativeChainModelTrainer):
    def prepare_arguments(self, parser):
        """ Add arguments to an argparse parser for training. """
        parser.add_argument("event_model",
                            help="Name of a trained autoencoder-comp model that gives us our basic event "
                                 "representations that we will learn a similarity function of")
        parser.add_argument("--layer-sizes", default="100",
                            help="Comma-separated list of layer sizes (default: 100, single layer)")
        parser.add_argument("--corruption", type=float, default=0.2,
                            help="Level of drop-out noise to apply during training, 0.0-1.0 (default: 0.2)")
        parser.add_argument("--batch-size", type=int, default=1000,
                            help="Number of examples to include in a minibatch (default: 1000)")
        parser.add_argument("--iterations", type=int, default=10,
                            help="Number of iterations to train each layer for (default: 10)")
        parser.add_argument("--regularization", type=float, default=0.01,
                            help="L2 regularization coefficient (default: 0.01)")
        parser.add_argument("--lr", type=float, default=0.1,
                            help="SGD learning rate (default: 0.1)")
        parser.add_argument("--stage", type=int,
                            help="Perform just one learning stage, rather than going from scratch. "
                                 "1=autoencoder pretraining, 2=fine tuning")
        parser.add_argument("--tuning-lr", type=float, default=0.025,
                            help="SGD learning rate to use for fine-tuning (default: 0.025)")
        parser.add_argument("--tuning-min-lr", type=float, default=0.0001,
                            help="Learning rate drops off as we go through the dataset. Don't let it go "
                                 "lower than this value (default: 0.0001)")
        parser.add_argument("--tuning-regularization", type=float, default=0.01,
                            help="L2 regularization coefficient for fine-tuning stage (default: 0.01)")
        parser.add_argument("--tuning-iterations", type=int, default=3,
                            help="Number of iterations to fine tune for (default: 3)")
        parser.add_argument("--continue-tuning", action="store_true",
                            help="Permit a model that's already been fine-tuned to be tuned with some more iterations "
                                 "over the data")
        parser.add_argument("--restart-tuning", action="store_true",
                            help="Reload the stored stage 1 result and run stage 2 again from scratch")
        parser.add_argument("--random-init", action="store_true",
                            help="Don't pretrain the layers as autoencoders, but just randomly initialize weights "
                                 "and move on to tuning")
        parser.add_argument("--update-empty-vecs", action="store_true",
                            help="Vectors for empty arg slots are initialized to 0. Allow these to be learned during "
                                 "training (by default, left as 0)")
        parser.add_argument("--update-input-vecs", action="store_true",
                            help="Allow fine tuning to adjust the original word2vec vectors, rather than simply "
                                 "learning a composition function")
        parser.add_argument("--update-event-composition", action="store_true",
                            help="Allow fine tuning to adjust the weights that define the composition of predicates "
                                 "and arguments into an event representation")
        parser.add_argument("--first-it-last-layer", action="store_true",
                            help="On the first tuning iteration, only train the last layer (the logistic regression, "
                                 "which doesn't get initialized as an autoencoder)")
        parser.add_argument("--validation-reports", action="store_true",
                            help="During tuning, output stats on a validation set (actually a subset of the training "
                                 "set) every 1k batches")
        parser.add_argument("--event-tuning-iterations", type=int, default=0,
                            help="After tuning the event composition function, perform some further iterations tuning "
                                 "the whole network, including event representations. By default, this is not done "
                                 "at all")

    def train(self, model_name, corpus, log, opts, chain_features=None):
        from .model import EventCompositionNarrativeChainModel
        from whim.entity_narrative import ArgumentCompositionNarrativeChainModel

        #tmp_dir = self.get_training_temp_dir(model_name, create=True, remove=False)
        if opts.stage not in [1, 2, None]:
            raise ValueError("training stage must be 1 or 2, or unspecified for all")
        stage1 = opts.stage is None or opts.stage == 1
        stage2 = opts.stage is None or opts.stage == 2
        if opts.stage is not None:
            log.info("Only performing stage %d of training" % opts.stage)
        #basilwang 2020-10-07 manually set tmp_dir
        #indexed_corpus = os.path.join(tmp_dir, "training_chain_indices")
        indexed_corpus = "/Users/wanghuajie/Documents/Code/python/chains/"
        indexed_corpus = os.path.join(indexed_corpus, "training_chain_indices")
        backup_name = "%s-stage1" % model_name

        if stage1:
            log.info("STAGE 1")
            training_metadata = {
                "data": corpus.directory,
                "training_stage_completed": 0,
                "tuning_iterations": 0,
            }

            # Split up the layer size specification
            layer_sizes = [int(size) for size in opts.layer_sizes.split(",")]

            log.info("Loading event model '%s'" % opts.event_model)
            # basilwang 2020-10-07 we need use EventCompositionNarrativeChainModel otherwise can't find projection_model
            #event_model = ArgumentCompositionNarrativeChainModel.load(opts.event_model)
            event_model = EventCompositionNarrativeChainModel.load(opts.event_model)

            # Create and initialize the network(s)
            log.info("Initializing composition network with layer sizes [%d|%d]->%s->1" %
                     (event_model.vector_size, event_model.vector_size, "->".join(str(s) for s in layer_sizes)))
            # network = PairCompositionNetwork(event_model.projection_model, layer_sizes=layer_sizes)
            network = event_model.pair_projection_model
            model = EventCompositionNarrativeChainModel(
                network, event_model.pred_vocab, event_model.arg_vocab,
                training_metadata=training_metadata,
                predicative_adjectives=event_model.predicative_adjectives,
                transitivity=event_model.transitivity
            )
            model.save(model_name)

            # Store the whole corpus as a list of ints once now so we don't have to look up indices on every iteration
            if os.path.exists(indexed_corpus):
                log.info("Indexed corpus already exists. Delete %s if you want to reindex" % indexed_corpus)
            else:
                # basilwang 2020-10-07 manually set tmp_dir
                # Check whether the original event model has a cached indexed corpus from when it was trained
                # event_model_tmp_dir = event_model.get_trainer().get_training_temp_dir(opts.event_model,
                #                                                                       create=False, remove=False)
                event_model_indexed_corpus = os.path.join(indexed_corpus, "training_chain_indices")
                if os.path.exists(event_model_indexed_corpus):
                    # Don't copy automatically, as there's no guarantee original model was trained on same corpus
                    log.info("Indexed corpus found in original event model's tmp dir. If you want to reuse this run: \n"
                             "cp %s %s" % (event_model_indexed_corpus, indexed_corpus))
                log.info("Indexing full corpus in %s" % indexed_corpus)
                index_corpus(corpus, model, indexed_corpus)

            if opts.random_init:
                log.info("Randomly initializing layers and performing no pre-training")
                # Nothing to do: AEs are randomly initialized to start with, so just don't train them
            else:
                # Normal, autoencoder pre-training
                # Start the training algorithm
                log.info("Preraining with l2 reg=%f, lr=%f, corruption=%f, %d iterations per layer, "
                         "%d-instance minibatches" %
                         (opts.regularization, opts.lr, opts.corruption, opts.iterations, opts.batch_size))
                for layer in range(len(layer_sizes)):
                    corpus_it = PretrainCorpusIterator(indexed_corpus, model, layer, batch_size=opts.batch_size)
                    trainer = DenoisingAutoencoderIterableTrainer(network.layers[layer])
                    trainer.train(
                        ProgressBarIter(corpus_it, title="Pretraining layer %d" % layer),
                        iterations=opts.iterations,
                        corruption_level=opts.corruption,
                        log=log,
                        regularization=opts.regularization,
                        learning_rate=opts.lr,
                        loss="l2",
                    )

                    log.info("Finished pretraining layer %d" % layer)
                    log.info("Saving model: %s" % model_name)
                    model.save(model_name)

            log.info("Finished training stage 1 (pretraining)")
            model.training_metadata["training_stage_completed"] = 1
            model.save(model_name)
        elif stage2:
            if opts.restart_tuning:
                # Check that a stage 1 backup is available and use if
                if not os.path.exists(EventCompositionNarrativeChainModel.get_model_directory(backup_name)):
                    raise ModelTrainingError("couldn't find stored stage 1 backup (%s)" %
                                             EventCompositionNarrativeChainModel.get_model_directory(backup_name))
                log.info("Restoring from stage 1 backup model (%s)" % backup_name)
                shutil.rmtree(EventCompositionNarrativeChainModel.get_model_directory(model_name))
                shutil.copytree(EventCompositionNarrativeChainModel.get_model_directory(backup_name),
                                EventCompositionNarrativeChainModel.get_model_directory(model_name))
                # Should now find that when we load this model stage 2 hasn't been completed
            # Not running stage 1 of training, so load the model that we've already pretrained
            log.info("Loading stage 1 model")
            try:
                model = EventCompositionNarrativeChainModel.load(model_name)
            except ModelLoadError, e:
                raise ModelTrainingError("trying to run stage 2 of training, but no model named '%s' exists: "
                                         "stage 1 has not been run yet. (%s)" % (model_name, e))
            # Check whether the old indexed corpus is still available
            reindex = not os.path.exists(indexed_corpus)
            if reindex:
                log.info("Indexed corpus from stage 1 not available: reindexing in %s" % indexed_corpus)
                index_corpus(corpus, model, indexed_corpus)

        if stage2:
            log.info("STAGE 2")
            # Check we've done stage 1 and not stage 2
            completed = model.training_metadata.get("training_stage_completed", 0)
            if completed == 2:
                if opts.continue_tuning:
                    # Continue when stage 2's already been completed
                    log.info("Stage 2 already completed, but continuing to tune some more")
                    backup_num = 0
                    while os.path.exists(model.get_model_directory("%s-stage2-%d" % (model_name, backup_num))):
                        backup_num += 1
                    backup_name = "%s-stage2-%d" % (model_name, backup_num)
                else:
                    raise ModelTrainingError("model fine-tuning has already been completed on %s. To continue with "
                                             "further training iterations, use --continue-tuning. To rerun stage 2 "
                                             "from stored stage 1 result, use --restart-tuning" % model_name)
            elif completed == 0:
                raise ModelTrainingError("autoencoder pretraining (stage 1) has not been completed yet on this model")

            tuning_iterations_done = model.training_metadata.get("tuning_iterations", 0)
            if tuning_iterations_done > 0:
                log.info("Already completed %d tuning iterations" % tuning_iterations_done)

            if not opts.restart_tuning:
                # If we just restarted tuning by restoring from the backup, we don't need to save the backup again
                log.info("Keeping a copy of stage 1 model as %s" % backup_name)
                model.save(backup_name)

            log.info("Fine tuning with l2 reg=%s, lr=%s, %d-instance minibatches, %d its updating composition%s" %
                     (opts.tuning_regularization,
                      opts.tuning_lr,
                      opts.batch_size,
                      opts.tuning_iterations,
                      "" if not opts.event_tuning_iterations
                      else (", %d its updating full network" % opts.event_tuning_iterations)
                      ))
            tuner = PairProjectionFineTuner(
                model,
                learning_rate=opts.tuning_lr,
                min_learning_rate=opts.tuning_min_lr,
                regularization=opts.tuning_regularization,
                update_empty_vectors=opts.update_empty_vecs,
                update_input_vectors=opts.update_input_vecs,
                update_event_composition=opts.update_event_composition,
            )
            corpus_it = PairFineTuneCorpusIterator(indexed_corpus, model, batch_size=opts.batch_size)
            training_model_name = "%s-training" % model_name

            def _iteration_callback(iteration_num, *args):
                if iteration_num < opts.tuning_iterations - 1:
                    log.info("Saving between-iterations model: %s" % training_model_name)
                    model.training_metadata["tuning_iterations"] = tuning_iterations_done + iteration_num + 1
                    model.save(training_model_name)

            if opts.validation_reports:
                ###### Validation
                # Use a negative sample generator, which samples randomly from the dataset, to get "validation" samples
                validation_generator = NegativeSampleGenerator(indexed_corpus)
                # Use another one to draw independent samples to act as negatives
                negative_validation_generator = NegativeSampleGenerator(indexed_corpus)
                # Fetch a load of validation chains (actually in the training set, but never mind for now)
                # Randomly select two from each chain
                validation_chains = [
                    random.sample(chain, 2) for chain in
                    islice((chain for chain in validation_generator if len(chain) >= 2), 0, 5000, 5)
                ]
                negative_events = list(
                    islice((random.choice(chain)
                            for chain in negative_validation_generator if len(chain) >= 2), 0, 5000, 5)
                )
                # Put the chains into arrays so we can project them
                triple_num_events = len(validation_chains)
                triple_inputs = [
                    [
                        numpy.zeros(triple_num_events, dtype=numpy.int32) for i in range(4)  # Pred and each of 3 args
                    ] for j in range(3)  # Pos/neg lhs, pos rhs, neg rhs
                ]
                flat_triple_inputs = [array for position in triple_inputs for array in position]

                for p, (positive_pair, negative_event) in enumerate(zip(validation_chains, negative_events)):
                    # Use the first of the pair as the LHS of pos and neg
                    for array, val in zip(triple_inputs[0], positive_pair[0]):
                        array[p] = val
                    # Use the second as the RHS of the pos
                    for array, val in zip(triple_inputs[1], positive_pair[1]):
                        array[p] = val
                    # Use the negative sample as the RHS of the neg
                    for array, val in zip(triple_inputs[2], negative_event):
                        array[p] = val

                pair_coherence_fn = theano.function(
                    inputs=model.pair_projection_model.triple_inputs,
                    outputs=model.pair_projection_model.get_coherence_pair(),
                )

                def _batch_callback(iteration_num, batch_num):
                    # Every 100 batches, evaluate how the model's doing
                    if (batch_num > 0 and batch_num % 1000 == 0) or batch_num == -1:
                        # Project all of these inputs to get a score for each event pair
                        pos_coherences, neg_coherences = pair_coherence_fn(*flat_triple_inputs)
                        # Compute the difference in coherence between each positive pair and the corresponding negative
                        coherence_diffs = pos_coherences - neg_coherences
                        negative_diffs = numpy.minimum(coherence_diffs, 0.)

                        log.info("Coherences of validation sample (batch %d): "
                                 "mean pos=%g, mean neg=%g, mean diff=%g, mean of negative diff=%g" %
                                 (batch_num, pos_coherences.mean(), neg_coherences.mean(), coherence_diffs.mean(),
                                  negative_diffs.mean()))
                # Call it once before training starts
                _batch_callback(-1, -1)
            else:
                # By default, don't do anything between batches
                _batch_callback = None

            # Start the tuning process
            tuner.train(
                corpus_it,
                corpus_it.num_samples(),
                iterations=opts.tuning_iterations,
                log=log,
                iteration_callback=_iteration_callback,
                batch_callback=_batch_callback,
                first_it_last_layer=opts.first_it_last_layer,
            )

            if opts.event_tuning_iterations > 0:
                log.info("Performing %d iterations updating the full network, including event representations" %
                         opts.event_tuning_iterations)
                pre_full_tune_name = "%s-pre-full-tune" % model_name
                log.info("Saving tuned model before full tune as %s" % pre_full_tune_name)
                model.save(pre_full_tune_name)

                full_tuner = PairProjectionFineTuner(
                    model,
                    # Stick at the lowest learning rate
                    learning_rate=opts.tuning_min_lr,
                    min_learning_rate=opts.tuning_min_lr,
                    regularization=opts.tuning_regularization,
                    update_empty_vectors=opts.update_empty_vecs,
                    update_input_vectors=True,
                    update_event_composition=True,
                )
                full_tuner.train(
                    corpus_it,
                    corpus_it.num_samples(),
                    iterations=opts.event_tuning_iterations,
                    log=log,
                    iteration_callback=_iteration_callback,
                    batch_callback=_batch_callback,
                )
                model.training_metadata["full_tuning_iterations"] = \
                    model.training_metadata.get("full_tuning_iterations", 0) + opts.event_tuning_iterations

            log.info("Tuning complete: saving model '%s'" % model_name)
            model.training_metadata["training_stage_completed"] = 2
            model.training_metadata["tuning_iterations"] = tuning_iterations_done + opts.tuning_iterations
            model.save(model_name)

            # Remove the temporary between-iterations model
            ArgumentCompositionNarrativeChainModel.delete_model(training_model_name)

        return model


class PairProjectionFineTuner(object):
    """
    Once we've pre-trained our deep network using autoencoders to get a compact event projection,
    fine tune it so that events in the same chain tend to be close together.

    """
    def __init__(self, model, learning_rate=0.025, regularization=0.01, min_learning_rate=0.0001,
                 update_input_vectors=False, update_empty_vectors=False, update_event_composition=False):
        self.update_event_composition = update_event_composition
        self.min_learning_rate = min_learning_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.model = model
        self.network = model.pair_projection_model
        self.learning_rate_var = T.scalar("learning_rate", dtype=theano.config.floatX)

        event_network = self.network.event_network

        # Collect parameters to be tuned from all layers
        self.params = []
        self.regularized_params = []
        # Always add our own between-event composition function weights
        self.params.extend(
            sum([[layer.W, layer.b] for layer in model.pair_projection_model.layers], [])
        )
        self.regularized_params.extend([layer.W for layer in model.pair_projection_model.layers])

        self.params.append(self.model.pair_projection_model.prediction_weights)
        self.params.append(self.model.pair_projection_model.prediction_bias)
        self.regularized_params.append(self.model.pair_projection_model.prediction_weights)

        if update_event_composition:
            # Add the event-internal composition weights
            self.params.extend(sum([[layer.W, layer.b] for layer in event_network.layers], []))
            self.regularized_params.extend([layer.W for layer in event_network.layers])

        if update_empty_vectors:
            self.params.extend([
                event_network.empty_arg0_vector,
                event_network.empty_arg1_vector,
                event_network.empty_arg2_vector
            ])
        if update_input_vectors:
            self.params.extend([
                event_network.argument0_vectors,
                event_network.argument1_vectors,
                event_network.argument2_vectors,
            ])

        self.update_empty_vectors = update_empty_vectors
        self.update_input_vectors = update_input_vectors

        self.positive = T.vector("positive", dtype="int8")
        self.positive_col = self.positive.dimshuffle(0, 'x')

    def get_cost_updates(self, regularization=None, first_pass=False):
        if regularization is None:
            regularization = self.regularization

        # On the first pass, only update the parameters of the final, logistic regression layer
        if first_pass:
            params = [self.network.prediction_weights, self.network.prediction_bias]
            reg_params = [self.network.prediction_weights]
        else:
            params = self.params
            reg_params = self.regularized_params

        # Compute the similarity predicted by our current composition
        sim_per_input = self.network.prediction
        cost_per_input = -T.log(T.switch(self.positive_col, sim_per_input, 1. - sim_per_input))
        cost = T.mean(cost_per_input)

        if regularization > 0.:
            # Collect weights to be regularized from all layers
            reg_term = regularization * T.sum([T.sum(w ** 2) for w in reg_params])
            # Turn this into a mean by counting up the weight params
            reg_term /= T.cast(T.sum([T.prod(T.shape(w)) for w in reg_params]), theano.config.floatX)
            cost += reg_term

        # Now differentiate to get the updates
        gparams = [T.grad(cost, param) for param in params]
        updates = [(param, param - self.learning_rate_var * gparam) for param, gparam in zip(params, gparams)]

        return cost, updates

    def get_triple_cost_updates(self, regularization=None):
        if regularization is None:
            regularization = self.regularization

        # Compute the two similarities predicted by our current composition
        pos_coherences, neg_coherences = self.network.get_coherence_pair()
        # We want pos coherences to be higher than neg
        # Try to make pos - neg as high as possible (it ranges between -1 and 1)
        cost_per_input = -T.log(pos_coherences) - T.log(1. - neg_coherences)
        cost = T.mean(cost_per_input)

        if regularization > 0.:
            # Collect weights to be regularized from all layers
            reg_term = regularization * T.sum([T.sum(w ** 2) for w in self.regularized_params])
            # Turn this into a mean by counting up the weight params
            reg_term /= T.cast(T.sum([T.prod(T.shape(w)) for w in self.regularized_params]), theano.config.floatX)
            cost += reg_term

        # Now differentiate to get the updates
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - self.learning_rate_var * gparam) for param, gparam in zip(self.params, gparams)]

        return cost, updates

    def train(self, batch_iterator, total_samples, iterations=10000, validation_set=None, stopping_iterations=10,
              cost_plot_filename=None, iteration_callback=None, log=None, training_cost_prop_change_threshold=0.0005,
              batch_callback=None, first_it_last_layer=False):
        if log is None:
            log = get_console_logger("Autoencoder tune")

        log.info("Tuning params: learning rate=%s (->%s), regularization=%s" %
                 (self.learning_rate, self.min_learning_rate, self.regularization))
        if self.update_empty_vectors:
            log.info("Training empty vectors")
        if self.update_input_vectors:
            log.info("Updating basic word representations")

        ######## Compile functions
        network = self.model.pair_projection_model
        # Prepare cost/update functions for training
        cost, updates = self.get_triple_cost_updates()
        cost_without_reg, __ = self.get_triple_cost_updates(regularization=0.)
        # Prepare training functions
        cost_fn = theano.function(
            inputs=network.triple_inputs,
            outputs=cost_without_reg,
        )
        train_fn = theano.function(
            inputs=network.triple_inputs + [
                # Allow the learning rate to be set per update
                theano.Param(self.learning_rate_var, default=self.learning_rate),
            ],
            outputs=cost,
            updates=updates,
        )
        # Doesn't do anything now: used to do something different
        first_pass_train_fn = train_fn
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

        below_threshold_its = 0

        for i in range(iterations):
            err = 0.0
            batch_num = 0
            learning_rate = self.learning_rate
            seen_samples = 0
            tenth_progress = -1

            if i == 0 and first_it_last_layer:
                # On the first iteration, use the training function that only updates the final layer
                log.info("First pass: only updating final layer (logistic regression)")
                train = first_pass_train_fn
            else:
                train = train_fn

            for batch_num, batch_inputs in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should with SGD
                # Just shuffle within batches
                shuffle = numpy.random.permutation(batch_inputs[0].shape[0])
                for batch_data in batch_inputs:
                    batch_data[:] = batch_data[shuffle]

                # Update the model with this batch's data
                err += train(*batch_inputs, learning_rate=learning_rate)

                seen_samples += batch_inputs[0].shape[0]
                # Update the learning rate, so it falls away as we go through
                # Do this only on the first iteration. After that, LR should just stay at the min
                if i == 0:
                    learning_rate = max(self.min_learning_rate,
                                        self.learning_rate * (1. - float(seen_samples) / total_samples))

                current_tenth_progress = int(math.floor(10. * float(seen_samples) / total_samples))
                if current_tenth_progress > tenth_progress:
                    tenth_progress = current_tenth_progress
                    mean_cost_so_far = err / (batch_num+1)
                    log.info("%d%% of iteration: training cost so far = %.5g" %
                             (current_tenth_progress * 10, mean_cost_so_far))
                    if i == 0:
                        log.info("Learning rate updated to %g" % learning_rate)

                if batch_callback is not None:
                    batch_callback(i, batch_num)

            if batch_num == 0:
                raise ModelTrainingError("zero batches returned by training data iterator")
            training_costs.append(err / (batch_num+1))

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


class PretrainCorpusIterator(object):
    def __init__(self, indexed_corpus, model, layer_input=-1, batch_size=1):
        self.fine_tune_iterator = PairFineTuneCorpusIterator(indexed_corpus, model, batch_size=batch_size)
        self.layer_input = layer_input
        if layer_input == -1:
            # Compile the expression for the deepest hidden layer
            self.projection_fn = model.pair_projection_model.coherence_fn
        else:
            # Compile the theano expression for this layer's input
            self.projection_fn = model.pair_projection_model.get_layer_input_function(layer_input)

    def __len__(self):
        return 2*len(self.fine_tune_iterator)

    def num_samples(self):
        return self.fine_tune_iterator.num_samples()

    def __iter__(self):
        for preds_l, arg0s_l, arg1s_l, arg2s_l, preds_r_pos, arg0s_r_pos, arg1s_r_pos, arg2s_r_pos, \
                preds_r_neg, arg0s_r_neg, arg1s_r_neg, arg2s_r_neg in self.fine_tune_iterator:
            # Project this input to the relevant layer of the network
            # Don't care about whether the examples are positive or negative
            yield self.projection_fn(preds_l, arg0s_l, arg1s_l, arg2s_l,
                                     preds_r_pos, arg0s_r_pos, arg1s_r_pos, arg2s_r_pos)
            yield self.projection_fn(preds_l, arg0s_l, arg1s_l, arg2s_l,
                                     preds_r_neg, arg0s_r_neg, arg1s_r_neg, arg2s_r_neg)


class PairFineTuneCorpusIterator(object):
    """
    Iterator to prepare a training example for each chain for the fine-tuning stage.

    """
    def __init__(self, indexed_corpus, model, batch_size=1):
        self.indexed_corpus = indexed_corpus
        self.batch_size = batch_size
        self.model = model
        # Prepare an iterator to randomly sample negative samples
        self.negative_generator = NegativeSampleGenerator(indexed_corpus)
        self._samples = None

    def __len__(self):
        return int(math.ceil(float(self.num_samples()) / self.batch_size))

    def num_samples(self):
        if self._samples is None:
            # Count the positive examples in the file: one for every event in every chain with at least two events
            reader = GroupedIntListsReader(self.indexed_corpus)
            self._samples = sum(len(chain) for chain in reader if len(chain) >= 2)
        return self._samples

    def __iter__(self):
        array_length = self.batch_size
        preds_left = numpy.zeros(array_length, dtype=numpy.int32)
        arg0s_left = numpy.zeros(array_length, dtype=numpy.int32)
        arg1s_left = numpy.zeros(array_length, dtype=numpy.int32)
        arg2s_left = numpy.zeros(array_length, dtype=numpy.int32)
        preds_right_pos = numpy.zeros(array_length, dtype=numpy.int32)
        arg0s_right_pos = numpy.zeros(array_length, dtype=numpy.int32)
        arg1s_right_pos = numpy.zeros(array_length, dtype=numpy.int32)
        arg2s_right_pos = numpy.zeros(array_length, dtype=numpy.int32)
        preds_right_neg = numpy.zeros(array_length, dtype=numpy.int32)
        arg0s_right_neg = numpy.zeros(array_length, dtype=numpy.int32)
        arg1s_right_neg = numpy.zeros(array_length, dtype=numpy.int32)
        arg2s_right_neg = numpy.zeros(array_length, dtype=numpy.int32)

        sample_idx = 0

        reader = GroupedIntTuplesReader(self.indexed_corpus)
        for chain_events in reader:
            chain_events = remove_duplicates(chain_events)
            # Must have at least two events in the chain to be able to use it
            if len(chain_events) >= 2:
                # Use each event in the chain once (on the LHS - i.e. with a positive and negative companion)
                for left_event_idx, left_event in enumerate(chain_events):
                    preds_left[sample_idx], arg0s_left[sample_idx], \
                    arg1s_left[sample_idx], arg2s_left[sample_idx] = left_event

                    # Pick another event within the chain to use as a positive example
                    right_positive = random.choice(chain_events[:left_event_idx] + chain_events[left_event_idx+1:])
                    preds_right_pos[sample_idx], arg0s_right_pos[sample_idx], \
                    arg1s_right_pos[sample_idx], arg2s_right_pos[sample_idx] = right_positive

                    # Pick a negative sample that doesn't feature in this chain
                    right_negative = random.choice(self.negative_generator.get_sample(chain_events))
                    preds_right_neg[sample_idx], arg0s_right_neg[sample_idx], \
                    arg1s_right_neg[sample_idx], arg2s_right_neg[sample_idx] = right_negative

                    # If we've filled up the batch, yield it
                    sample_idx += 1
                    if sample_idx == self.batch_size:
                        yield preds_left, arg0s_left, arg1s_left, arg2s_left, \
                              preds_right_pos, arg0s_right_pos, arg1s_right_pos, arg2s_right_pos, \
                              preds_right_neg, arg0s_right_neg, arg1s_right_neg, arg2s_right_neg
                        sample_idx = 0

        # TODO Should return this last partial batch, but having a smaller batch is currently messing up training
        # TODO If you update this, allow for the triple option as well
        if False and sample_idx > 0:
            # We've partially filled a batch: yield this as the last item
            yield preds_left[:sample_idx], \
                  arg0s_left[:sample_idx], \
                  arg1s_left[:sample_idx], \
                  arg2s_left[:sample_idx], \
                  preds_right_pos[:sample_idx], \
                  arg0s_right_pos[:sample_idx], \
                  arg1s_right_pos[:sample_idx], \
                  arg2s_right_pos[:sample_idx], \
                  preds_right_neg[:sample_idx], \
                  arg0s_right_neg[:sample_idx], \
                  arg1s_right_neg[:sample_idx], \
                  arg2s_right_neg[:sample_idx]


class NegativeSampleGenerator(object):
    def __init__(self, indexed_corpus):
        self.indexed_corpus = indexed_corpus
        self.reader = GroupedIntTuplesReader(indexed_corpus)
        # Cycle round the dataset forever, shuffling with a buffer of 10k as we go
        self.getter = iter(
            limited_shuffle(infinite_cycle(self.reader), 1e4)
        )
        # Skip over a random number of lines at the beginning so we don't always start from the same place
        for i in range(random.randint(0, 1e5)):
            self.getter.next()

    def get_sample(self, exclude=[]):
        # Get the first non-empty chain that doesn't include the base example(s)
        return dropwhile(lambda chain: any(e in chain for e in exclude) or len(chain) == 0, self.getter).next()

    def __iter__(self):
        while True:
            yield self.get_sample()


def infinite_cycle(iterable):
    while True:
        for x in iterable:
            yield x


def index_corpus(corpus, model, index_file):
    with GroupedIntListsWriter(index_file) as writer:
        for doc in corpus:
            for chain_events in model.get_chain_input_vectors(doc.get_chains()):
                writer.write(chain_events)
