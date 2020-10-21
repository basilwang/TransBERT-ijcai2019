from __future__ import absolute_import
import copy
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
from whim_common.utils.shuffle import limited_shuffle
from whim_common.utils.theano.nn.autoencoder import DenoisingAutoencoderIterableTrainer
from whim_common.utils.vectors import vector_cosine_similarity


class ArgumentCompositionTrainer(NarrativeChainModelTrainer):
    def prepare_arguments(self, parser):
        """ Add arguments to an argparse parser for training. """
        parser.add_argument("word2vec_model",
                            help="Name of a trained word2vec to use as the input predicate and argument vectors "
                                 "to learn a composition of")
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
                                 "1=autoencoder pretraining, 2=fine tuning, 3=autoencoder pretraining for pairs, "
                                 "4=fine tuning for pairs")
        parser.add_argument("--tuning-lr", type=float, default=0.025,
                            help="SGD learning rate to use for fine-tuning (default: 0.025)")
        parser.add_argument("--tuning-min-lr", type=float, default=0.0001,
                            help="Learning rate drops off as we go through the dataset. Don't let it go "
                                 "lower than this value (default: 0.0001)")
        parser.add_argument("--tuning-regularization", type=float, default=0.01,
                            help="L2 regularization coefficient for fine-tuning stage (default: 0.01)")
        parser.add_argument("--tuning-iterations", type=int, default=3,
                            help="Number of iterations to fine tune for (default: 3)")
        parser.add_argument("--negative-samples", type=int, default=5,
                            help="Number of negative pair samples to include for every positive during fine tuning "
                                 "(default: 5)")
        parser.add_argument("--continue-tuning", action="store_true",
                            help="Permit a model that's already been fine-tuned to be tuned with some more iterations "
                                 "over the data")
        parser.add_argument("--restart-tuning", action="store_true",
                            help="Reload the stored stage 1 result and run stage 2 again from scratch")
        parser.add_argument("--identity-init", action="store_true",
                            help="Instead of initializing by training autoencoders, use initial weights such that "
                                 "the first layer simply sums the input vectors and subsequent layers maintain the "
                                 "same representation. Note that this requires all the layers to be the same size "
                                 "as the individual input vector components (i.e. the word2vec vector size)")
        parser.add_argument("--random-init", action="store_true",
                            help="Don't pretrain the layers as autoencoders, but just randomly initialize weights "
                                 "and move on to tuning")
        parser.add_argument("--update-empty-vecs", action="store_true",
                            help="Vectors for empty arg slots are initialized to 0. Allow these to be learned during "
                                 "training (by default, left as 0)")
        parser.add_argument("--update-input-vecs", action="store_true",
                            help="Allow fine tuning to adjust the original word2vec vectors, rather than simply "
                                 "learning a composition function")
        parser.add_argument("--tuning-objective", default="cosine",
                            help="Objective to use at tuning time. Choose from: cosine, dot, sqerr. (Default: cosine)")
        parser.add_argument("--scale-word2vec", action="store_true",
                            help="Scale the word2vec vectors so that each dimension has 0 mean and unit std dev "
                                 "before training begins")
        parser.add_argument("--pos-neg-diff", action="store_true",
                            help="Perform tuning using the difference between sim(p0, p1) and sim(p0, n) as an "
                                 "objective, where p0 is the target event, p1 another from the same chain and n "
                                 "a negative sample from another random chain")
        parser.add_argument("--validation-reports", action="store_true",
                            help="During tuning, output stats on a validation set (actually a subset of the training "
                                 "set) every 1k batches")

    def train(self, model_name, corpus, log, opts, chain_features=None):
        from .model import ArgumentCompositionNarrativeChainModel, EventVectorNetwork
        from whim.entity_narrative.models.word2vec.model import Word2VecNarrativeChainModel

        #tmp_dir = self.get_training_temp_dir(model_name, create=True, remove=False)
        if opts.stage not in [1, 2, None]:
            raise ValueError("training stage must be 1 or 2, or unspecified for all")
        stage1 = opts.stage is None or opts.stage == 1
        stage2 = opts.stage is None or opts.stage == 2
        if opts.stage is not None:
            log.info("Only performing stage %d of training" % opts.stage)
        # basilwang 2020-10-07 do not use tmp_dir
        indexed_corpus = "/Users/wanghuajie/Documents/Code/python/chains/chain_indices/arg_comp/"
        indexed_corpus = os.path.join(indexed_corpus, "training_chain_indices")
        backup_name = "%s-stage1" % model_name

        if stage1:
            log.info("STAGE 1")
            training_metadata = {
                "data": corpus.directory,
                "training_stage_completed": 0,
                "identity_init": opts.identity_init,
                "tuning_iterations": 0,
            }

            # Split up the layer size specification
            layer_sizes = [int(size) for size in opts.layer_sizes.split(",")]

            # Load the word2vec predicate and argument model that we're using as input
            log.info("Loading base word2vec model")
            word2vec_model = Word2VecNarrativeChainModel.load(opts.word2vec_model)
            if not word2vec_model.with_args:
                raise ModelTrainingError(
                    "base model '%s' wasn't trained to learn an argument representation, so can't be used as the "
                    "basis for a composition model" % opts.word2vec_model
                )
            word2vec = word2vec_model.word2vec

            if opts.identity_init and not all(size == word2vec.layer1_size for size in layer_sizes):
                raise ModelTrainingError("to initialize network to identity (sum) functions, all layer sizes "
                                         "must be equal to the word2vec hidden layer size (%d)" % word2vec.layer1_size)

            # Split up the predicate and argument vectors in the model
            pred_vocab, pred_matrix, arg_vocab, arg_matrix = split_pred_arg_vocab_and_matrices(
                # basilwang 2020-10-07 use wv
                word2vec.wv.vocab, word2vec.wv.syn0norm
                # word2vec.vocab, word2vec.syn0norm
            )

            # Create and initialize the network(s)
            log.info("Initializing network with layer sizes %d->%s" % (pred_matrix.shape[1]+3*arg_matrix.shape[1],
                                                                       "->".join(str(s) for s in layer_sizes)))
            network = EventVectorNetwork(pred_matrix, arg_matrix,
                                         layer_sizes=layer_sizes)
            model = ArgumentCompositionNarrativeChainModel(network, pred_vocab, arg_vocab,
                                                              training_metadata=training_metadata,
                                                              predicative_adjectives=word2vec_model.predicative_adjectives,
                                                              transitivity=word2vec_model.transitivity)
            model.save(model_name)

            # Store the whole corpus as a list of ints once now so we don't have to look up indices on every iteration
            if os.path.exists(indexed_corpus):
                log.info("Indexed corpus already exists. Delete %s if you want to reindex" % indexed_corpus)
            else:
                log.info("Indexing full corpus in %s" % indexed_corpus)
                index_corpus(corpus, model, indexed_corpus)

            if opts.scale_word2vec:
                log.info("Scaling word2vec dimensions to have 0 mean and unit std dev")
                # Compute the mean on each dimension over the dataset
                feature_means = numpy.zeros((pred_matrix.shape[1]+3*arg_matrix.shape[1]), dtype=theano.config.floatX)
                mean_it = CorpusIterator(indexed_corpus, model, layer_input=0, batch_size=100000)
                data_points = 0
                for vector in mean_it:
                    feature_means += vector.sum(axis=0)
                    data_points += vector.shape[0]
                feature_means /= data_points
                # Shift the word2vec vectors to have 0 mean
                log.info(" Shifting dimensions by, on average, %g" % numpy.mean(numpy.abs(feature_means)))
                model.projection_model.shift_input_vectors(-feature_means)
                # Compute the std dev on each dimension
                feature_std_dev = numpy.zeros((pred_matrix.shape[1]+3*arg_matrix.shape[1]), dtype=theano.config.floatX)
                for vector in mean_it:
                    # Mean is now 0
                    # Square each dimension, to take the mean square
                    feature_std_dev += (vector ** 2.).sum(axis=0)
                feature_std_dev /= data_points
                feature_std_dev = numpy.sqrt(feature_std_dev)
                feature_scaler = 1. / feature_std_dev
                # Scale each dimension to unit std dev -- divide by the std dev
                log.info(" Scaling dimensions by, on average, %g" % numpy.mean(feature_scaler))
                model.projection_model.scale_input_vectors(feature_scaler)

            if opts.identity_init:
                log.info("Simply initializing layers to sum, then identity")
                # Simply initialize the weights to identity matrices
                # The first layer is 4 Is stacked on top of each other: simply sum the 4 input vectors
                identity = numpy.eye(word2vec.layer1_size, dtype=theano.config.floatX)
                layer0_w = numpy.vstack((identity, identity, identity, identity))
                zero_bias = numpy.zeros(word2vec.layer1_size, dtype=theano.config.floatX)
                model.projection_model.layers[0].set_weights(
                    (layer0_w, zero_bias, numpy.zeros(word2vec.layer1_size*4, dtype=theano.config.floatX))
                )

                # The other layers start out just as identity matrices
                for layer in model.projection_model.layers[1:]:
                    layer.set_weights((identity, zero_bias, zero_bias))
            elif opts.random_init:
                log.info("Randomly initializing layers and performing no pre-training")
                # Nothing to do: AEs are randomly initialized to start with, so just don't train them
            else:
                # Normal, autoencoder pre-training
                # Start the training algorithm
                log.info("Pretraining with l2 reg=%f, lr=%f, corruption=%f, %d iterations per layer, "
                         "%d-instance minibatches" %
                         (opts.regularization, opts.lr, opts.corruption, opts.iterations, opts.batch_size))
                for layer in range(len(layer_sizes)):
                    log.info("Pretraining layer %d" % layer)
                    corpus_it = CorpusIterator(indexed_corpus, model, layer, batch_size=opts.batch_size)
                    trainer = DenoisingAutoencoderIterableTrainer(network.layers[layer])
                    trainer.train(
                        corpus_it,
                        iterations=opts.iterations,
                        corruption_level=opts.corruption,
                        log=log,
                        regularization=opts.regularization,
                        learning_rate=opts.lr,
                        loss="l2",
                    )

                    log.info("Finished training layer %d" % layer)
                    log.info("Saving model: %s" % model_name)
                    model.save(model_name)

            log.info("Finished training stage 1 (pretraining)")
            model.training_metadata["training_stage_completed"] = 1
            model.save(model_name)
        elif stage2:
            if opts.restart_tuning:
                # Check that a stage 1 backup is available and use if
                if not os.path.exists(ArgumentCompositionNarrativeChainModel.get_model_directory(backup_name)):
                    raise ModelTrainingError("couldn't find stored stage 1 backup (%s)" %
                                             ArgumentCompositionNarrativeChainModel.get_model_directory(backup_name))
                log.info("Restoring from stage 1 backup model (%s)" % backup_name)
                shutil.rmtree(ArgumentCompositionNarrativeChainModel.get_model_directory(model_name))
                shutil.copytree(ArgumentCompositionNarrativeChainModel.get_model_directory(backup_name),
                                ArgumentCompositionNarrativeChainModel.get_model_directory(model_name))
                # Should now find that when we load this model stage 2 hasn't been completed
            # Not running stage 1 of training, so load the model that we've already pretrained
            log.info("Loading stage 1 model")
            try:
                model = ArgumentCompositionNarrativeChainModel.load(model_name)
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

            log.info("Fine tuning with l2 reg=%f, lr=%f, %d-instance minibatches, %d iterations" %
                     (opts.tuning_regularization,
                      opts.tuning_lr,
                      opts.batch_size,
                      opts.tuning_iterations))
            if opts.pos_neg_diff:
                log.info("Using objective based on difference between positive and negative similarity samples")
            tuner = EventProjectionFineTuner(
                model,
                learning_rate=opts.tuning_lr,
                min_learning_rate=opts.tuning_min_lr,
                regularization=opts.tuning_regularization,
                #positive_sample_weight=opts.negative_samples,
                update_empty_vectors=opts.update_empty_vecs,
                update_input_vectors=opts.update_input_vecs,
                objective=opts.tuning_objective,
                diff_objective=opts.pos_neg_diff,
            )
            corpus_it = FineTuneCorpusIterator(indexed_corpus, model, batch_size=opts.batch_size,
                                               negative_samples=opts.negative_samples,
                                               pos_neg_triples=opts.pos_neg_diff)
            training_model_name = "%s-training" % model_name

            def _iteration_callback(iteration_num, *args):
                if iteration_num < opts.tuning_iterations - 1:
                    log.info("Saving between-iterations model: %s" % training_model_name)
                    model.training_metadata["tuning_iterations"] = tuning_iterations_done + iteration_num + 1
                    model.save(training_model_name)

            if opts.validation_reports:
                ###### Validation
                # TODO Really we should hold this set out of the training set
                # Use a negative sample generator, which samples randomly from the dataset, to get "validation" samples
                validation_generator = NegativeSampleGenerator(indexed_corpus)
                # Use another one to draw independent samples to act as negatives
                negative_validation_generator = NegativeSampleGenerator(indexed_corpus)
                # Fetch a load of validation chains (actually in the training set, but never mind for now)
                validation_chains = list(
                    islice((chain for chain in validation_generator if len(chain) >= 2), 0, 5000, 5)
                )
                negative_events = list(
                    islice((random.choice(chain)
                            for chain in negative_validation_generator if len(chain) >= 2), 0, 5000, 5)
                )
                # Put the chains into arrays so we can project them
                input_sizes = [len(chain) for chain in validation_chains]
                num_events = sum(input_sizes)
                val_preds = numpy.zeros(num_events+len(negative_events), dtype=numpy.int32)
                val_arg0s = numpy.zeros(num_events+len(negative_events), dtype=numpy.int32)
                val_arg1s = numpy.zeros(num_events+len(negative_events), dtype=numpy.int32)
                val_arg2s = numpy.zeros(num_events+len(negative_events), dtype=numpy.int32)
                ptr = 0
                for chain_events in validation_chains:
                    for pred, arg0, arg1, arg2 in chain_events:
                        val_preds[ptr], val_arg0s[ptr], val_arg1s[ptr], val_arg2s[ptr] = pred, arg0, arg1, arg2
                        ptr += 1
                for pred, arg0, arg1, arg2 in negative_events:
                    val_preds[ptr], val_arg0s[ptr], val_arg1s[ptr], val_arg2s[ptr] = pred, arg0, arg1, arg2
                    ptr += 1
                ########## End of validation prep

                def _batch_callback(iteration_num, batch_num):
                    # Every 1000 batches, evaluate how the model's doing
                    if batch_num % 1000 == 0 or batch_num == -1:
                        # Project all of these inputs to get a vector for each event
                        event_vectors = model.projection_model.project(val_preds, val_arg0s, val_arg1s, val_arg2s)
                        #u_event_vectors = model.projection_model.unnormalized_project(val_preds, val_arg0s, val_arg1s, val_arg2s)
                        #print u_event_vectors
                        # Sum up the projected vectors from each chain, excluding the last one
                        so_far = 0
                        similarity = 0.
                        negative_similarity = 0.
                        ratios = []
                        for chain_num, chain_input_size in enumerate(input_sizes):
                            # Sum up the chain's vectors
                            chain_vector = numpy.mean(event_vectors[so_far:so_far+chain_input_size-1], axis=0)
                            # Compute their similarity (by dot product) with the last event in the chain
                            pos_sim = vector_cosine_similarity(chain_vector, event_vectors[so_far+chain_input_size-1])
                            neg_sim = vector_cosine_similarity(chain_vector, event_vectors[num_events+chain_num])
                            similarity += pos_sim
                            negative_similarity += neg_sim
                            ratios.append((1+pos_sim) / (1+neg_sim))
                            so_far += chain_input_size
                        similarity /= len(validation_chains)
                        negative_similarity /= len(validation_chains)
                        log.info("Mean dot product of validation sample (batch %d): "
                                 "pos=%g, neg=%g, mean ratio=%g, mean diff=%g" %
                                 (batch_num, similarity, negative_similarity, float(numpy.mean(ratios)),
                                  similarity - negative_similarity))
                _batch_callback(-1, -1)
            else:
                _batch_callback = None

            # Start the tuning process
            tuner.train(
                corpus_it,
                len(corpus_it),
                iterations=opts.tuning_iterations,
                log=log,
                iteration_callback=_iteration_callback,
                batch_callback=_batch_callback,
            )
            log.info("Tuning complete: saving model '%s'" % model_name)
            model.training_metadata["training_stage_completed"] = 2
            model.training_metadata["tuning_iterations"] = tuning_iterations_done + opts.tuning_iterations
            model.save(model_name)

            # Remove the temporary between-iterations model
            ArgumentCompositionNarrativeChainModel.delete_model(training_model_name)

        return model


class EventProjectionFineTuner(object):
    """
    Once we've pre-trained our deep network using autoencoders to get a compact event projection,
    fine tune it so that events in the same chain tend to be close together.

    """
    def __init__(self, model, learning_rate=0.025, regularization=0.01, min_learning_rate=0.0001,
                 positive_sample_weight=1, objective="cosine", update_input_vectors=False,
                 update_empty_vectors=False, diff_objective=False):
        self.diff_objective = diff_objective
        self.objective = objective
        self.positive_sample_weight = float(positive_sample_weight)
        self.min_learning_rate = min_learning_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.model = model
        self.network = model.projection_model
        self.learning_rate_var = T.scalar("learning_rate", dtype=theano.config.floatX)

        # Collect parameters to be tuned from all layers
        self.params = sum([[layer.W, layer.b] for layer in self.model.projection_model.layers], [])
        if update_empty_vectors:
            self.params.extend([
                self.model.projection_model.empty_arg0_vector,
                self.model.projection_model.empty_arg1_vector,
                self.model.projection_model.empty_arg2_vector
            ])
        if update_input_vectors:
            self.params.extend([
                self.model.projection_model.argument0_vectors,
                self.model.projection_model.argument1_vectors,
                self.model.projection_model.argument2_vectors,
            ])
        self.update_empty_vectors = update_empty_vectors
        self.update_input_vectors = update_input_vectors

        self.positive = T.vector("positive", dtype="int8")

    def get_cosine(self):
        # Use the deepest projection layer as our target representation
        projection_a, projection_b = self.model.projection_model.get_projection_pair(normalize=True)
        # No need to normalize: the projection is already normalized
        # Now take their dot product, giving us the cosine similarity between the two projections
        return T.sum(projection_a * projection_b, axis=1)

    def get_cosine_pair(self):
        # Use the deepest projection layer as our target representation
        projection_a, projection_b, projection_c = self.model.projection_model.get_projection_triple(normalize=True)
        # No need to normalize: the projection is already normalized
        # Now take their dot product, giving us the cosine similarity between the two projections
        # Do this for sim(A,B) and sim(A,C): generally one should be low and the other high
        return T.sum(projection_a * projection_b, axis=1), T.sum(projection_a * projection_c, axis=1)

    def get_dot(self):
        # Use the deepest projection layer as our target representation
        projection_a, projection_b = self.model.projection_model.get_projection_pair(normalize=False)
        # Now take their dot product
        return T.sum(projection_a * projection_b, axis=1)

    def get_squared_error(self):
        projection_a, projection_b = self.model.projection_model.get_projection_pair(normalize=False)
        return T.sum((projection_a - projection_b) ** 2., axis=1)

    def get_cost_updates(self, regularization=None):
        if regularization is None:
            regularization = self.regularization
        # First compute the cosine similarity of the two vectors
        # We want this to be large for positive cases and small for negative, so the cost is simply the negated cosine
        if self.objective == "cosine":
            cost_per_input = -T.log(0.5 + T.switch(self.positive, 0.5, -0.5) * self.get_cosine())
        elif self.objective == "dot":
            # Dot prod ends up between -num_dims and num_dims
            # Scale down to 0 to 1
            projection_a, projection_b = self.model.projection_model.get_projection_pair(normalize=False)
            cost_per_input = -T.log(0.5 + T.switch(self.positive, 0.5, -0.5) *
                                    T.mean(projection_a * projection_b, axis=1))
        elif self.objective == "sqerr":
            cost_per_input = T.switch(self.positive, 1., -1.) * self.get_squared_error()
        else:
            raise ValueError("unknown objective '%s'" % self.objective)
        cost = T.mean(T.switch(self.positive, self.positive_sample_weight, 1.) * cost_per_input)

        if regularization > 0.:
            # Collect weights to be regularized from all layers
            weights = [layer.W for layer in self.model.projection_model.layers]
            reg_term = regularization * T.sum([T.sum(w ** 2) for w in weights])
            # Turn this into a mean by counting up the weight params
            reg_term /= T.sum([T.prod(T.shape(w)) for w in weights])
            cost += reg_term

        # Now differentiate to get the updates
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - self.learning_rate_var * gparam) for param, gparam in zip(self.params, gparams)]

        return cost, updates

    def get_diff_cost_updates(self, regularization=None):
        if regularization is None:
            regularization = self.regularization
        # First compute the cosine similarity of the two vectors
        # We want this to be large for positive cases and small for negative, so the cost is simply the negated cosine
        if self.objective == "cosine":
            # Get the similarity of inputA to both inputB and inputC
            pos_sim, neg_sim = self.get_cosine_pair()
            sim_diff = pos_sim - neg_sim
            # If pos > neg, we're happy, no cost
            # Otherwise, pull in the direction of making pos > neg
            cost_per_input = -T.log(1 + 0.5 * T.where(sim_diff > 0, 0, sim_diff))
        elif self.objective == "dot":
            raise NotImplementedError
        elif self.objective == "sqerr":
            raise NotImplementedError
        else:
            raise ValueError("unknown objective '%s'" % self.objective)
        cost = T.mean(cost_per_input)

        if regularization > 0.:
            # Collect weights to be regularized from all layers
            weights = [layer.W for layer in self.model.projection_model.layers]
            reg_term = regularization * T.sum([T.sum(w ** 2) for w in weights])
            # Turn this into a mean by counting up the weight params
            reg_term /= T.sum([T.prod(T.shape(w)) for w in weights])
            cost += reg_term

        # Now differentiate to get the updates
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - self.learning_rate_var * gparam) for param, gparam in zip(self.params, gparams)]

        return cost, updates

    def train(self, batch_iterator, total_samples, iterations=10000, validation_set=None, stopping_iterations=10,
              cost_plot_filename=None, iteration_callback=None, log=None, training_cost_prop_change_threshold=0.0005,
              batch_callback=None):
        if log is None:
            log = get_console_logger("Autoencoder tune")

        log.info("Tuning params: learning rate=%s (->%s), regularization=%.2f, objective=%s" %
                 (self.learning_rate, self.min_learning_rate, self.regularization, self.objective))
        if self.update_empty_vectors:
            log.info("Training empty vectors")
        if self.update_input_vectors:
            log.info("Updating basic word representations")

        ######## Compile functions
        network = self.model.projection_model
        if self.diff_objective:
            cost, updates = self.get_diff_cost_updates()
            cost_fn = theano.function(
                inputs=[
                    network.predicate_input_a, network.arg0_input_a, network.arg1_input_a, network.arg2_input_a,
                    network.predicate_input_b, network.arg0_input_b, network.arg1_input_b, network.arg2_input_b,
                    network.predicate_input_c, network.arg0_input_c, network.arg1_input_c, network.arg2_input_c,
                ],
                outputs=cost,
            )
            train_fn = theano.function(
                inputs=[
                    network.predicate_input_a, network.arg0_input_a, network.arg1_input_a, network.arg2_input_a,
                    network.predicate_input_b, network.arg0_input_b, network.arg1_input_b, network.arg2_input_b,
                    network.predicate_input_c, network.arg0_input_c, network.arg1_input_c, network.arg2_input_c,
                    # Allow the learning rate to be set per update
                    theano.Param(self.learning_rate_var, default=self.learning_rate),
                ],
                outputs=cost,
                updates=updates,
            )
        else:
            # Prepare cost/update functions for training
            cost, updates = self.get_cost_updates()
            # Prepare training functions
            cost_fn = theano.function(
                inputs=[
                    network.predicate_input_a, network.arg0_input_a, network.arg1_input_a, network.arg2_input_a,
                    network.predicate_input_b, network.arg0_input_b, network.arg1_input_b, network.arg2_input_b,
                    self.positive,
                ],
                outputs=cost,
            )
            train_fn = theano.function(
                inputs=[
                    network.predicate_input_a, network.arg0_input_a, network.arg1_input_a, network.arg2_input_a,
                    network.predicate_input_b, network.arg0_input_b, network.arg1_input_b, network.arg2_input_b,
                    self.positive,
                    # Allow the learning rate to be set per update
                    theano.Param(self.learning_rate_var, default=self.learning_rate),
                ],
                outputs=cost,
                updates=updates,
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

        below_threshold_its = 0

        for i in range(iterations):
            err = 0.0
            batch_num = 0
            learning_rate = self.learning_rate
            seen_samples = 0
            #pbar = get_progress_bar(total_samples, title="Iteration %d" % i, counter=True)
            tenth_progress = -1

            for batch_num, batch_inputs in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should with SGD
                # Just shuffle within batches
                shuffle = numpy.random.permutation(batch_inputs[0].shape[0])
                for batch_data in batch_inputs:
                    batch_data[:] = batch_data[shuffle]

                # Update the model with this batch's data
                err += train_fn(*batch_inputs, learning_rate=learning_rate)

                seen_samples += batch_inputs[0].shape[0]
                # Update the learning rate, so it falls away as we go through
                learning_rate = max(self.min_learning_rate,
                                    self.learning_rate * (1. - float(seen_samples) / total_samples))

                current_tenth_progress = math.floor(10. * float(seen_samples) / total_samples)
                if current_tenth_progress > tenth_progress:
                    tenth_progress = current_tenth_progress
                    log.info("Learning rate at %.1f%% = %g" % (100. * float(seen_samples) / total_samples,
                                                               learning_rate))

                #if seen_samples < total_samples:
                #    pbar.update(seen_samples)

                if batch_callback is not None:
                    batch_callback(i, batch_num)

            #pbar.finish()

            if batch_num == 0:
                raise ModelTrainingError("zero batches returned by training data iterator")
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



class CorpusIterator(object):
    """
    Iterator to prepare inputs for autoencoder pretraining

    """
    def __init__(self, indexed_corpus, model, layer_input=-1, batch_size=1):
        self.indexed_corpus = indexed_corpus
        self.batch_size = batch_size
        self.model = model
        self.layer_input = layer_input
        if layer_input == -1:
            # Compile the expression for the deepest hidden layer
            self.projection_fn = model.projection_model.project
        else:
            # Compile the theano expression for this layer's input
            self.projection_fn = model.projection_model.get_layer_input_function(layer_input)

    def __iter__(self):
        preds = numpy.zeros(self.batch_size, dtype=numpy.int32)
        arg0s = numpy.zeros(self.batch_size, dtype=numpy.int32)
        arg1s = numpy.zeros(self.batch_size, dtype=numpy.int32)
        arg2s = numpy.zeros(self.batch_size, dtype=numpy.int32)
        data_point_index = 0

        reader = GroupedIntListsReader(self.indexed_corpus)
        for chain_events in reader:
            for pred, arg0, arg1, arg2 in chain_events:
                preds[data_point_index] = pred
                arg0s[data_point_index] = arg0
                arg1s[data_point_index] = arg1
                arg2s[data_point_index] = arg2
                data_point_index += 1

                # If we've filled up the batch, yield it
                if data_point_index == self.batch_size:
                    # Get the vector (up to the current point in the network) for this event
                    yield self.projection_fn(preds, arg0s, arg1s, arg2s)
                    data_point_index = 0

        if data_point_index > 0:
            # We've partially filled a batch: yield this as the last item
            yield self.projection_fn(preds[:data_point_index], arg0s[:data_point_index], arg1s[:data_point_index],
                                     arg2s[:data_point_index])


class FineTuneCorpusIterator(object):
    """
    Iterator to prepare a training example for each chain for the fine-tuning stage.

    """
    def __init__(self, indexed_corpus, model, batch_size=1, negative_samples=5, pos_neg_triples=False):
        self.pos_neg_triples = pos_neg_triples
        self.negative_samples = negative_samples
        self.indexed_corpus = indexed_corpus
        self.batch_size = batch_size
        self.model = model
        # Prepare an iterator to randomly sample negative samples
        self.negative_generator = NegativeSampleGenerator(indexed_corpus)
        self._length = None

    def __len__(self):
        if self._length is None:
            # If using pos-neg diff objective, just yield a single sample per chain
            if self.pos_neg_triples:
                # Count the positive examples in the file
                reader = GroupedIntListsReader(self.indexed_corpus)
                self._length = 0
                for chain in reader:
                    if len(chain) >= 2:
                        self._length += 1
            else:
                # Count the positive examples in the file
                reader = GroupedIntListsReader(self.indexed_corpus)
                self._length = 0
                for chain in reader:
                    if len(chain) >= 2:
                        self._length += len(chain) * (len(chain) - 1)
                # Account for the negative samples
                self._length *= (self.negative_samples+1)
        return self._length

    def __iter__(self):
        if self.pos_neg_triples:
            array_length = self.batch_size
        else:
            array_length = self.batch_size*(self.negative_samples+1)
        preds_a = numpy.zeros(array_length, dtype=numpy.int32)
        arg0s_a = numpy.zeros(array_length, dtype=numpy.int32)
        arg1s_a = numpy.zeros(array_length, dtype=numpy.int32)
        arg2s_a = numpy.zeros(array_length, dtype=numpy.int32)
        preds_b = numpy.zeros(array_length, dtype=numpy.int32)
        arg0s_b = numpy.zeros(array_length, dtype=numpy.int32)
        arg1s_b = numpy.zeros(array_length, dtype=numpy.int32)
        arg2s_b = numpy.zeros(array_length, dtype=numpy.int32)
        preds_c = numpy.zeros(array_length, dtype=numpy.int32)
        arg0s_c = numpy.zeros(array_length, dtype=numpy.int32)
        arg1s_c = numpy.zeros(array_length, dtype=numpy.int32)
        arg2s_c = numpy.zeros(array_length, dtype=numpy.int32)
        # Always the same pattern of positives and negatives at regular intervals
        positive = numpy.zeros(self.batch_size*(self.negative_samples+1), dtype=numpy.int8)

        data_point_index = 0
        array_cursor = 0

        reader = GroupedIntTuplesReader(self.indexed_corpus)
        for chain_events in reader:
            # Remove duplicate events: these don't help the training, as they'll always be identical vectors!
            chain_events = remove_duplicates(chain_events)
            # Must have at least two events in the chain to be able to use it
            if len(chain_events) >= 2:
                if self.pos_neg_triples:
                    # Sample a single pair of positive events
                    positive_pair = random.sample(chain_events, 2)
                    # Sample a negative
                    negative_chain = self.negative_generator.get_sample(positive_pair)
                    negative_event = random.choice(negative_chain)

                    preds_a[data_point_index], arg0s_a[data_point_index], \
                    arg1s_a[data_point_index], arg2s_a[data_point_index] = positive_pair[0]
                    preds_b[data_point_index], arg0s_b[data_point_index], \
                    arg1s_b[data_point_index], arg2s_b[data_point_index] = positive_pair[1]
                    preds_c[data_point_index], arg0s_c[data_point_index], \
                    arg1s_c[data_point_index], arg2s_c[data_point_index] = negative_event

                    data_point_index += 1
                    if data_point_index == self.batch_size:
                        positive[::self.negative_samples+1] = 1
                        yield preds_a, arg0s_a, arg1s_a, arg2s_a, \
                              preds_b, arg0s_b, arg1s_b, arg2s_b, \
                              preds_c, arg0s_c, arg1s_c, arg2s_c
                        data_point_index = 0
                else:
                    # Train on every event in the chain
                    for event0_index, event0 in enumerate(chain_events):
                        # Use each other event as a positive example
                        for event1 in chain_events[:event0_index] + chain_events[event0_index+1:]:
                            # Set this as the left event for the positive sample and all of the negatives
                            preds_a[array_cursor:array_cursor+self.negative_samples+1], \
                            arg0s_a[array_cursor:array_cursor+self.negative_samples+1], \
                            arg1s_a[array_cursor:array_cursor+self.negative_samples+1], \
                            arg2s_a[array_cursor:array_cursor+self.negative_samples+1] = event0
                            preds_b[array_cursor], arg0s_b[array_cursor], arg1s_b[array_cursor], arg2s_b[array_cursor] = \
                                event1

                            # Added a positive sample, add negatives too
                            for i in range(1, self.negative_samples+1):
                                negative_chain = self.negative_generator.get_sample([event0, event1])
                                negative_event = random.choice(negative_chain)
                                preds_b[array_cursor+i], arg0s_b[array_cursor+i], arg1s_b[array_cursor+i], arg2s_b[array_cursor+i] \
                                    = negative_event

                            # If we've filled up the batch, yield it
                            data_point_index += 1
                            if data_point_index == self.batch_size:
                                # Get the vector (up to the current point in the network) for this event
                                positive[::self.negative_samples+1] = 1
                                yield preds_a, arg0s_a, arg1s_a, arg2s_a, preds_b, arg0s_b, arg1s_b, arg2s_b, positive
                                data_point_index = 0

                            array_cursor = data_point_index*(self.negative_samples+1)

        # TODO Should return this last partial batch, but having a smaller batch is currently messing up training
        # TODO If you update this, allow for the triple option as well
        if False and data_point_index > 0:
            # We've partially filled a batch: yield this as the last item
            positive[::self.negative_samples+1] = 1
            yield preds_a[:array_cursor], \
                  arg0s_a[:array_cursor], \
                  arg1s_a[:array_cursor], \
                  arg2s_a[:array_cursor], \
                  preds_b[:array_cursor], \
                  arg0s_b[:array_cursor], \
                  arg1s_b[:array_cursor], \
                  arg2s_b[:array_cursor], \
                  positive[:array_cursor]


class NegativeSampleGenerator(object):
    def __init__(self, indexed_corpus):
        self.indexed_corpus = indexed_corpus
        self.reader = GroupedIntTuplesReader(indexed_corpus)
        # Cycle round the dataset forever, shuffling with a buffer of 10k as we go
        self.getter = iter(
            limited_shuffle(
                (
                    remove_duplicates(chain) for chain in infinite_cycle(self.reader)
                ), 1e4
            )
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


def split_pred_arg_vocab_and_matrices(vocab_dict, matrix):
    pred_vocab, pred_id2word, pred_id_map = {}, [], []
    arg_vocab, arg_id2word, arg_id_map = {}, [], []

    for word, v in vocab_dict.items():
        # Don't update this vocab item, but a copy
        v = copy.deepcopy(v)
        # Distinguish arg words from predicates
        if word.startswith("arg:"):
            arg_word = word[4:]
            # Keep a note of the old index for copying vectors
            arg_id_map.append(v.index)
            # Update the index of this vocab item
            v.index = len(arg_id2word)
            # Add the word to the new vocabulary
            arg_vocab[arg_word] = v
            arg_id2word.append(v)
        else:
            # Keep a note of the old index for copying vectors
            pred_id_map.append(v.index)
            # Update the index of this vocab item
            v.index = len(pred_id2word)
            # Add the word to the new predicate-only vocab
            pred_vocab[word] = v
            pred_id2word.append(v)

    # Now split up the actual matrix
    pred_matrix = numpy.zeros((len(pred_id2word), matrix.shape[1]), dtype=matrix.dtype)
    arg_matrix = numpy.zeros((len(arg_id2word), matrix.shape[1]), dtype=matrix.dtype)
    # Set the vectors by selecting appropriate rows from the old matrix
    pred_matrix[:, :] = matrix[pred_id_map, :]
    arg_matrix[:, :] = matrix[arg_id_map, :]

    return pred_vocab, pred_matrix, arg_vocab, arg_matrix
