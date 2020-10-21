from itertools import islice
import random

import numpy
import theano.tensor as T
import theano
from cam.whim.entity_narrative.chains.document import Event
from whim_common.utils.vectors import vector_cosine_similarity


class HillClimbingNextEventSampler(object):
    def __init__(self, model, context_chain, limit_pred_vocab=None, limit_arg_vocab=None, exclude_seen_preds=False,
                 alpha=0.01):
        self.alpha = alpha
        self.model = model
        base_event_network = model.pair_projection_model.event_network

        # Get the input indices for the context events
        context_arrays = self.get_context_vectors(context_chain)
        self.context_entity, self.context_events = context_chain
        self.context_size = len(self.context_events)
        # Project all the context events into the event space so we get the LHS vectors
        context_vectors = base_event_network.project(*context_arrays)
        context_vectors = context_vectors[None, :, :]
        context_vector_input = theano.shared(context_vectors)

        if limit_pred_vocab:
            # Work out what predicate indices we should use
            self.pred_indices = [
                v.index for v in
                list(sorted(self.model.pred_vocab.values(), key=lambda v: v.count, reverse=True))[:limit_pred_vocab]
            ]
        else:
            self.pred_indices = list(range(model.pair_projection_model.event_network.pred_vocab_size))
        self.pred_vocab_size = len(self.pred_indices)
        self.pred_vocab_indices = numpy.tile(
            numpy.array(self.pred_indices, dtype=numpy.int32).reshape((len(self.pred_indices), 1)),
            (1, self.context_size)
        )
        self.pred_index2vocab = dict((index, vocab_num) for (vocab_num, index) in enumerate(self.pred_indices))

        if limit_arg_vocab:
            # Work out what arg indices we should use
            self.arg_indices = [
                v.index for v in
                list(sorted(self.model.arg_vocab.values(), key=lambda v: v.count, reverse=True))[:limit_arg_vocab]
            ] + [-1]
        else:
            self.arg_indices = list(range(model.pair_projection_model.event_network.arg_vocab_size)) + [-1]
        self.arg_vocab_size = len(self.arg_indices)
        self.arg_vocab_indices = numpy.tile(
            numpy.array(self.arg_indices, dtype=numpy.int32).reshape((len(self.arg_indices), 1)),
            (1, self.context_size)
        )
        self.arg_index2vocab = dict((index, vocab_num) for (vocab_num, index) in enumerate(self.arg_indices))

        # Dimensions of inputs:
        #  0: possible input values, 1: context events

        # Create variables for the unobserved inputs (RHS), which we're sampling
        self.unseen_pred = T.vector("sampled_pred", dtype="int32")
        self.unseen_arg0 = T.vector("sampled_arg0", dtype="int32")
        self.unseen_arg1 = T.vector("sampled_arg1", dtype="int32")
        self.unseen_arg2 = T.vector("sampled_arg2", dtype="int32")

        self.right_pred_matrix = T.matrix("right_pred", dtype="int32")
        self.right_arg0_matrix = T.matrix("right_arg0", dtype="int32")
        self.right_arg1_matrix = T.matrix("right_arg1", dtype="int32")
        self.right_arg2_matrix = T.matrix("right_arg2", dtype="int32")

        # Create a new projection for the RHS, using the given vars as inputs
        rhs_projection, __, __, __ = base_event_network.copy_projection_function(
            predicate_input=self.right_pred_matrix,
            arg0_input=self.right_arg0_matrix,
            arg1_input=self.right_arg1_matrix,
            arg2_input=self.right_arg2_matrix
        )
        # Now compose the two projections by making a copy of the composition network that uses the trained weights
        prediction, __, __, __, activation = model.pair_projection_model.copy_coherence_function(
            input_a=T.extra_ops.repeat(context_vector_input, rhs_projection.shape[0], 0),
            input_b=rhs_projection,
        )
        scores = T.mean(prediction, axis=1)

        self.dist_given_others = theano.function(
            inputs=[self.right_pred_matrix, self.right_arg0_matrix, self.right_arg1_matrix, self.right_arg2_matrix],
            outputs=scores
        )

        self.exclude_preds = []
        self.exclude_args = [[], [], []]

        if exclude_seen_preds:
            # Get the predicate for each context event and make sure we don't sample it again
            for event in self.context_events:
                predicate_str = self.model.get_model_predicate_repr(self.context_entity, event)
                # Convert to vocab indices
                if predicate_str in self.model.pred_vocab:
                    self.exclude_preds.append(self.model.pred_vocab[predicate_str].index)

    def get_context_vectors(self, context_chain):
        inputs = list(self.model.get_chain_input_vectors([context_chain]))[0]
        input_arrays = [numpy.array(pos_indices, dtype=numpy.int32) for pos_indices in zip(*inputs)]
        return input_arrays

    def sample(self, burn_in=20, num_nearest_neighbors=100):
        return list(islice(self.sample_iter(batch=burn_in, num_nearest_neighbors=num_nearest_neighbors), 1))[0]

    def sample_iter(self, batch=20, num_nearest_neighbors=100):
        # Choose random initial values for the new event
        pred = numpy.array([random.choice(self.pred_indices)], dtype=numpy.int32)
        arg0 = numpy.array([random.choice(self.arg_indices)], dtype=numpy.int32)
        arg1 = numpy.array([random.choice(self.arg_indices)], dtype=numpy.int32)
        arg2 = numpy.array([random.choice(self.arg_indices)], dtype=numpy.int32)
        args = [arg0, arg1, arg2]

        pred_vecs = self.model.pair_projection_model.event_network.predicate_vectors.get_value()
        arg0_vecs = self.model.pair_projection_model.event_network.argument0_vectors.get_value()
        arg1_vecs = self.model.pair_projection_model.event_network.argument1_vectors.get_value()
        arg2_vecs = self.model.pair_projection_model.event_network.argument2_vectors.get_value()
        arg_vecs = [arg0_vecs, arg1_vecs, arg2_vecs]

        args_empty = [arg0[0] == -1, arg1[0] == -1, arg2[0] == -1]
        sample_args = [True] * 3

        def resample_pred():
            # Find nearest neighbours to the previous sample
            nearest_neighbours = [
                self.pred_indices[index] for index in reversed(numpy.argsort(
                    vector_cosine_similarity(pred_vecs[self.pred_indices], pred_vecs[pred[0]])
                )) if self.pred_indices[index] not in self.exclude_preds
            ][:num_nearest_neighbors]

            dist = self.dist_given_others(
                numpy.tile(
                    numpy.array(nearest_neighbours, dtype=numpy.int32).reshape((len(nearest_neighbours), 1)),
                    (1, self.context_size)
                ),
                numpy.tile(arg0, (len(nearest_neighbours), self.context_size)),
                numpy.tile(arg1, (len(nearest_neighbours), self.context_size)),
                numpy.tile(arg2, (len(nearest_neighbours), self.context_size))
            )
            dist = numpy.maximum(0., dist - dist[0]).astype(numpy.float64)
            # If nothing gives a higher coherence than the previous sample, don't sample again
            if dist.sum() > 0.:
                dist /= dist.sum()
                # Choose the maximal coherence from our nearest neighbours
                pred[0] = numpy.random.choice(nearest_neighbours, p=dist)

        def update_samples_from_pred():
            # Don't sample a value for the position that's filled by the entity, according to sampled predicate
            pred_str = self.model.pred_id2word[pred[0]]
            if pred_str.startswith("adj"):
                # Don't sample anything for any of the positions in this case
                sample_args[0] = sample_args[1] = sample_args[2] = False
                args_empty[0] = args_empty[1] = args_empty[2] = True
            else:
                sample_args[0] = sample_args[1] = sample_args[2] = True
                verb, pos = pred_str.split(":")
                if pos == "subj":
                    sample_args[0] = False
                    args_empty[0] = True
                elif pos == "obj":
                    sample_args[1] = False
                    args_empty[1] = True
                else:
                    sample_args[2] = False
                    args_empty[2] = True

        def resample_arg(arg):
            previous_sample = args[arg][0]
            # If we've not yet sampled a non-empty value, choose a vector at random to take NNs to
            previous_sample_id = previous_sample if previous_sample != -1 else random.choice(self.arg_indices[:-1])
            # Find nearest neighbours to the previous sample
            nearest_neighbours = [
                self.arg_indices[index] for index in reversed(numpy.argsort(
                    vector_cosine_similarity(arg_vecs[arg][self.arg_indices], arg_vecs[arg][previous_sample_id])
                )) if self.arg_indices[index] not in self.exclude_args[arg]
            ][:num_nearest_neighbors] + [-1]

            projection_args = [
                numpy.tile(pred, (len(nearest_neighbours), self.context_size)),
                numpy.tile(numpy.array([-1], dtype=numpy.int32)
                           if args_empty[0] else arg0, (len(nearest_neighbours), self.context_size)),
                numpy.tile(numpy.array([-1], dtype=numpy.int32)
                           if args_empty[1] else arg1, (len(nearest_neighbours), self.context_size)),
                numpy.tile(numpy.array([-1], dtype=numpy.int32)
                           if args_empty[2] else arg2, (len(nearest_neighbours), self.context_size))
            ]
            # Replace the appropriate position with the candidate nearest neighbours
            projection_args[arg+1] = numpy.tile(
                numpy.array(nearest_neighbours, dtype=numpy.int32).reshape((len(nearest_neighbours), 1)),
                (1, self.context_size)
            )
            dist = self.dist_given_others(*projection_args)
            # If we previously had an empty arg, take the coherence of empty as our base coherence
            # Otherwise use the coherence of the previously selected arg (which should be first in the NNs)
            sample_population = dist >= dist[-1 if args_empty[arg] else 0]
            dist = numpy.maximum(0., dist - dist[-1 if args_empty[arg] else 0]).astype(numpy.float64)
            # If nothing gives a higher coherence than the previous sample, don't sample again
            if dist.sum() > 0.:
                # Push up all non-zero coherences, so that there's some chance of selecting the same arg again,
                #  even if there are others with slightly higher coherence
                dist[sample_population] += self.alpha
                dist /= dist.sum()
                # Choose the maximal coherence from our nearest neighbours
                new_sample = numpy.random.choice(nearest_neighbours, p=dist)
                if new_sample == -1:
                    # Chosen to leave arg empty
                    # Don't update assignment, just mark as empty: keep previous assignment to take NNs to next time
                    args_empty[arg] = True
                else:
                    args[arg][0] = new_sample
                    args_empty[arg] = False

        update_samples_from_pred()

        # Run the sampler
        # Keep yielding samples indefinitely
        while True:
            # Do batch samples without yielding anything
            for i in range(batch):
                # Sample the different positions in a random order
                for position in random.sample(range(4), 4):
                    if position == 3:
                        resample_pred()
                        # Now we've got a new predicate, update which args we're sampling
                        update_samples_from_pred()
                    elif sample_args[position]:
                        resample_arg(position)

            indices = [int(pred[0]), int(arg0[0]), int(arg1[0]), int(arg2[0])]
            yield self.indices_to_event(indices, self.context_entity, args_empty), indices

    def indices_to_event(self, indices, entity, args_empty):
        # Convert the four indices into an event
        pred = self.model.pred_id2word[indices[0]]
        args = [self.model.arg_id2word[s] if s != -1 and not empty else None for (s, empty)
                in zip(indices[1:], args_empty)]

        # Split up the predicate to work out where to put the entity
        if pred.startswith("adj:"):
            __, verb, obj = pred.split(":")
            subj = entity
            iobj = None
        else:
            verb, pos = pred.split(":")
            subj, obj, iobj = args
            # Don't generally know what the preposition is
            prep = "_"
            if pos == "subj":
                subj = entity
            elif pos == "obj":
                obj = entity
            else:
                # Where the entity is in arg2 position, we get the preposition from the predicate
                prep = pred.partition(":prep_")[2]
                iobj = entity
            if iobj is not None:
                iobj = (prep, iobj)

        return Event(verb, verb, subject=subj, object=obj, iobject=iobj)
