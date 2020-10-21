"""
Tools for sampling next events given a context.

"""
import copy
from itertools import islice, dropwhile
import math
from operator import itemgetter
import warnings
import numpy
import theano
from theano import tensor as T
from theano.tensor import extra_ops
from whim_common.data.coref import Entity
from cam.whim.entity_narrative.chains.document import Event
from cam.whim.entity_narrative.models.arg_comp.model import ArgumentCompositionNarrativeChainModel
from cam.whim.entity_narrative.shell.commands import ModelShell
from whim_common.utils.base import remove_duplicates
from whim_common.utils.vectors import magnitude


class NextEventSampler(object):
    """
    The constructor options pred_given, etc, allow you to fix certain parts of the otherwise unseen next
    event. These then influence the sampling of the other parts. E.g. instead of being completely unknown
    at the start, our next event might be eat(?, ?, with fork).

    These need to be specified in the constructor, as they affect how the sampling equations are built up.
    If you're specifying different components to keep fixed in subsequent sampling runs, you'll need to
    rebuild the sampler. If you're just changing the actual values of the given positions (that is, the
    words that they're fixed to), you can use set_given(). This will raise an error if you try to change
    a value that wasn't a given before.

    """
    def __init__(self, model, learning_rate=0.1, pred_given=None, arg0_given=None, arg1_given=None, arg2_given=None):
        self.learning_rate = learning_rate
        self.model = model
        self.network = model.pair_projection_model
        self.event_network = self.network.event_network

        self.pred_given = pred_given
        self.arg0_given = arg0_given
        self.arg1_given = arg1_given
        self.arg2_given = arg2_given

        self.learning_rate_var = T.scalar("learning_rate", dtype=theano.config.floatX)

        # Create variables for the unobserved inputs (RHS), which we're sampling
        self.pred_size = (1, self.network.event_network.pred_vector_size)
        self.rhs_pred = theano.shared(
            numpy.zeros(self.pred_size, dtype=theano.config.floatX),
            borrow=True,
        )
        self.arg_size = (1, self.network.event_network.arg_vector_size)
        self.rhs_arg0 = theano.shared(
            numpy.zeros(self.arg_size, dtype=theano.config.floatX),
            borrow=True,
        )
        self.rhs_arg1 = theano.shared(
            numpy.zeros(self.arg_size, dtype=theano.config.floatX),
            borrow=True,
        )
        self.rhs_arg2 = theano.shared(
            numpy.zeros(self.arg_size, dtype=theano.config.floatX),
            borrow=True,
        )
        self.arg_vectors = [self.rhs_arg0, self.rhs_arg1, self.rhs_arg2]
        self.input_vector_size = self.pred_size[1] + 3*self.arg_size[1]
        rhs_vector = T.concatenate([self.rhs_pred, self.rhs_arg0, self.rhs_arg1, self.rhs_arg2], axis=1)
        # Rebuild the prediction function so that it uses our new vectors on the RHS
        # Repeat them over the first dimension, a single RHS vector is compared to all LHS vectors (context)
        rhs_projection = theano.clone(self.event_network.projection_layer, replace={
            self.event_network.input_vector: extra_ops.repeat(rhs_vector,
                                                              self.event_network.predicate_input_a.shape[0], axis=0)
        })
        prediction = theano.clone(self.network.prediction, replace={self.network.input_b: rhs_projection},
                                  share_inputs=True)
        # The prediction value is the coherence output, which we will use as our objective to maximize
        # Average it over the context inputs (comparing each to the single RHS vector)
        chain_coherence = T.mean(prediction)

        # The optimization fn updates the RHS vectors to maximize the mean coherence with the LHS
        self.params = []
        # Only optimize the positions that haven't been fixed
        if pred_given is None:
            self.params.append(self.rhs_pred)
        if arg0_given is None:
            self.params.append(self.rhs_arg0)
        if arg1_given is None:
            self.params.append(self.rhs_arg1)
        if arg2_given is None:
            self.params.append(self.rhs_arg2)
        if len(self.params) == 0:
            raise ValueError("all RHS event components have been fixed, so there's nothing left to sample!")

        cost = -T.log(chain_coherence)
        # Differentiate cost w.r.t. the RHS vectors to get the updates
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - self.learning_rate_var * gparam) for param, gparam in zip(self.params, gparams)]

        self.optimize = theano.function(
            inputs=[
                self.event_network.predicate_input_a,
                self.event_network.arg0_input_a,
                self.event_network.arg1_input_a,
                self.event_network.arg2_input_a,
                theano.Param(self.learning_rate_var, default=self.learning_rate)
            ],
            outputs=[cost, chain_coherence],
            updates=updates,
        )
        self.score_vector = theano.function(
            inputs=[
                self.event_network.predicate_input_a,
                self.event_network.arg0_input_a,
                self.event_network.arg1_input_a,
                self.event_network.arg2_input_a,
            ],
            outputs=chain_coherence,
        )

        self.positions = [self.rhs_pred, self.rhs_arg0, self.rhs_arg1, self.rhs_arg2]
        self.givens = [pred_given, arg0_given, arg1_given, arg2_given]
        self.vector_vocabs = [
            self.model.pair_projection_model.event_network.predicate_vectors.get_value(),
            self.model.pair_projection_model.event_network.argument0_vectors.get_value(),
            self.model.pair_projection_model.event_network.argument1_vectors.get_value(),
            self.model.pair_projection_model.event_network.argument2_vectors.get_value(),
        ]

        # Set the non-updated input vectors to the right values
        if not all(x is None for x in [pred_given, arg0_given, arg1_given, arg2_given]):
            if pred_given is not None:
                if pred_given not in self.model.pred_vocab:
                    warnings.warn("predicate '%s' not in vocabulary: not constraining sample on predicate" % pred_given)
                self.set_given(0, pred_given)
            if arg0_given is not None:
                if arg0_given == "--":
                    # Special value meaning fix to empty
                    self.set_given(1, None)
                else:
                    if arg0_given not in self.model.arg_vocab:
                        warnings.warn("arg '%s' not in vocabulary: not constraining sample on arg0" % arg0_given)
                    self.set_given(1, arg0_given)
            if arg1_given is not None:
                if arg1_given == "--":
                    self.set_given(2, None)
                else:
                    if arg1_given not in self.model.arg_vocab:
                        warnings.warn("arg '%s' not in vocabulary: not constraining sample on arg1" % arg1_given)
                    self.set_given(2, arg1_given)
            if arg2_given is not None:
                if arg2_given == "--":
                    self.set_given(3, None)
                else:
                    if arg2_given not in self.model.arg_vocab:
                        warnings.warn("arg '%s' not in vocabulary: not constraining sample on arg2" % arg2_given)
                    self.set_given(3, arg2_given)

    def randomize_vectors(self):
        # Randomly initialize the RHS vectors
        stddev = numpy.sqrt(1. / self.input_vector_size)
        # Don't reinitialize vectors that are declared fixed
        if self.pred_given is None:
            self.rhs_pred.set_value(
                numpy.asarray(numpy.random.normal(0., stddev, size=self.pred_size), dtype=theano.config.floatX)
            )
        if self.arg0_given is None:
            self.rhs_arg0.set_value(
                numpy.asarray(numpy.random.normal(0., stddev, size=self.arg_size), dtype=theano.config.floatX)
            )
        if self.arg1_given is None:
            self.rhs_arg1.set_value(
                numpy.asarray(numpy.random.normal(0., stddev, size=self.arg_size), dtype=theano.config.floatX)
            )
        if self.arg2_given is None:
            self.rhs_arg2.set_value(
                numpy.asarray(numpy.random.normal(0., stddev, size=self.arg_size), dtype=theano.config.floatX)
            )

    def zero_vectors(self):
        if self.pred_given is None:
            self.rhs_pred.set_value(numpy.zeros(self.pred_size, dtype=theano.config.floatX))
        if self.arg0_given is None:
            self.rhs_arg0.set_value(numpy.zeros(self.arg_size, dtype=theano.config.floatX))
        if self.arg1_given is None:
            self.rhs_arg1.set_value(numpy.zeros(self.arg_size, dtype=theano.config.floatX))
        if self.arg2_given is None:
            self.rhs_arg2.set_value(numpy.zeros(self.arg_size, dtype=theano.config.floatX))

    def zero_vector(self, arg_num):
        self.arg_vectors[arg_num].set_value(numpy.zeros(self.arg_size, dtype=theano.config.floatX))

    def set_vector(self, position, vector):
        vec_var = self.positions[position]
        vec_var.set_value(vector)

    def set_vectors(self, vectors):
        for pos, vec in enumerate(vectors):
            self.set_vector(pos, vec)

    def get_vectors(self):
        return [pos.get_value() for pos in self.positions]

    def get_context_vectors(self, context_chain):
        inputs = list(self.model.get_chain_input_vectors([context_chain]))[0]
        input_arrays = [numpy.array(pos_indices, dtype=numpy.int32) for pos_indices in zip(*inputs)]
        return input_arrays

    def set_given(self, position, word):
        if self.givens[position] is None:
            # This position wasn't set as a given value when the network was built
            # We can't change which positions are sampled now without rebuilding
            raise ValueError("position %d was not configured as a given (i.e. it was set to be sampled) when "
                             "sampler was created. Cannot set as a given without rebuilding sampler")
        else:
            if word is None:
                if position == 0:
                    # Can't keep the predicate as zero, as it can't be empty
                    raise ValueError("cannot set predicate to empty")
                # Keep fixed to the empty vector
                self.zero_vector(position-1)
                # Use a special value to indicate empty (None indicates not set)
                self.givens[position] = "--"
            else:
                vocab = self.model.pred_vocab if position == 0 else self.model.arg_vocab
                self.set_vector(position, self.vector_vocabs[position][numpy.newaxis, vocab[word].index])
                # Update the word so we know what to put in the event after sampling
                self.givens[position] = word

    def sample_next_input_vectors(self, context_chain, max_iterations=1000, cost_change_threshold=1e-4):
        # Get the inputs that will go on the LHS of the coherence function
        input_arrays = self.get_context_vectors(context_chain)
        # Randomly initialize the RHS vectors
        self.randomize_vectors()

        costs = []
        # Perform a load of SGD updates to the RHS vectors to bring them close to all the LHS vectors
        for i in range(max_iterations):
            cost, score = self.optimize(*input_arrays)
            costs.append(cost)

            # Check whether we've reached the threshold in absolute change in cost
            if len(costs) > 1 and abs(costs[-1] - costs[-2]) <= cost_change_threshold:
                break

        return self.get_vectors(), score

    def sample_next_input_event_parts(self, context_chain, max_iterations=1000, cost_change_threshold=1e-4):
        sampled_vectors, score = self.sample_next_input_vectors(
            context_chain,
            max_iterations=max_iterations,
            cost_change_threshold=cost_change_threshold
        )

        # Find the words closest to these vectors
        # The first one is a predicate
        if self.pred_given is not None:
            predicate = self.pred_given
        else:
            # Check which args are constrained: they can't be filled with the chain entity
            def _bad_predicate(pred):
                if self.arg0_given is not None and pred.endswith(":subj"):
                    return True
                if self.arg1_given is not None and pred.endswith(":obj"):
                    return True
                if self.arg2_given is not None and ":prep_" in pred:
                    return True
                return False

            predicate_vectors = self.model.pair_projection_model.event_network.predicate_vectors.get_value()
            similarities = numpy.dot(predicate_vectors, sampled_vectors[0][0])
            # Go through the predicates in order of similarity until we find one that fits our constraints
            predicate = dropwhile(_bad_predicate,
                                  (self.model.pred_id2word[idx] for idx in reversed(similarities.argsort()))).next()

        arg_vector_list = [
            self.event_network.argument0_vectors.get_value(),
            self.event_network.argument1_vectors.get_value(),
            self.event_network.argument2_vectors.get_value()
        ]
        args = []
        for arg_num, (arg_vec, arg_vectors) in enumerate(zip(sampled_vectors[1:], arg_vector_list)):
            if self.givens[arg_num+1] is not None:
                if self.givens[arg_num+1] == "--":
                    # Arg was fixed to empty
                    args.append(None)
                else:
                    args.append(self.givens[arg_num+1])
            else:
                similarities = numpy.dot(arg_vectors, arg_vec[0])
                args.append(self.model.arg_id2word[similarities.argmax()])

        return [predicate] + args, score

    def sample_next_input_event(self, context_chain, max_iterations=1000, cost_change_threshold=1e-4):
        entity, events = context_chain
        parts, score = self.sample_next_input_event_parts(context_chain,
                                                          max_iterations=max_iterations,
                                                          cost_change_threshold=cost_change_threshold)
        pred = parts[0]
        args = parts[1:]

        # Split up the predicate to work out where to put the entity
        if pred.startswith("adj"):
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

        return Event(verb, verb, subject=subj, object=obj, iobject=iobj), score

    def sample_next_input_events(self, context_chain, n, max_iterations=1000, cost_change_threshold=1e-4,
                                 rescore=False):
        scored_events = [self.sample_next_input_event(context_chain,
                                                      max_iterations=max_iterations,
                                                      cost_change_threshold=cost_change_threshold) for i in range(n)]

        if rescore:
            candidate_next_events = [event for event, score in scored_events]

            # Get rid of duplicate events -- we may have sampled the same thing multiple times
            def dup_key(event):
                # Key for deciding whether two events are duplicates
                return event.to_string_entity_text({context_chain[0]: "E0"})
            candidate_next_events = remove_duplicates(candidate_next_events, key=dup_key)

            # Run the events back through the coherence function to get their actual coherence scores, after
            # nearest neighbour, etc
            candidate_scores = self.model.score_choices(context_chain[0], context_chain[1], candidate_next_events)
            scored_events = list(zip(candidate_next_events, candidate_scores))

        scored_events.sort(key=itemgetter(1), reverse=True)
        return scored_events


class NextEventProjectionSampler(object):
    """
    Use SGD to sample a coherent next event by way of its argument composition vector (deepest
    representation layer before composition).

    """
    def __init__(self, projection_size, rhs_projection, optimize_fn, score_vector_fn, model,
                 neighbour_finder=None, learning_rate=0.1, num_samples=1):
        self.model = model
        self.num_samples = num_samples
        self.neighbour_finder = neighbour_finder
        self.learning_rate = learning_rate
        self.projection_size = projection_size
        self.rhs_projection = rhs_projection

        self.optimize = optimize_fn
        self.score_vector = score_vector_fn

    @staticmethod
    def from_model(model, neighbour_finder=None, learning_rate=0.1, num_samples=1, slimline_model=None):
        learning_rate_var = T.scalar("learning_rate", dtype=theano.config.floatX)
        network = model.pair_projection_model

        # Create variables for the unobserved input vector (RHS), which we're sampling
        projection_size = network.event_network.projection_size
        rhs_projection = theano.shared(
            numpy.zeros((num_samples, projection_size), dtype=theano.config.floatX),
            borrow=True,
        )
        # The prediction value is the coherence output, which we will use as our objective to maximize
        # Compute it as the composition of the observed LHS event(s) and the unobserved RHS event
        # Repeat over the first dimension, so a single RHS vector is compared to all LHS vectors (context)
        prediction = theano.clone(network.prediction, replace={
            network.input_b: extra_ops.repeat(rhs_projection,
                                                   network.event_network.predicate_input_a.shape[0], axis=0),
            network.event_network.predicate_input_a: T.tile(network.event_network.predicate_input_a, (num_samples,)),
            network.event_network.arg0_input_a: T.tile(network.event_network.arg0_input_a, (num_samples,)),
            network.event_network.arg1_input_a: T.tile(network.event_network.arg1_input_a, (num_samples,)),
            network.event_network.arg2_input_a: T.tile(network.event_network.arg2_input_a, (num_samples,)),
        }, share_inputs=True)
        # Average it over the context inputs (comparing each to the single RHS vector)
        chain_coherence = T.mean(prediction)

        # The optimization fn updates the RHS vector to maximize the mean coherence with the LHS
        params = [rhs_projection]
        cost = -T.log(chain_coherence)
        # Differentiate cost w.r.t. the RHS vectors to get the updates
        gparams = [T.grad(cost, param) for param in params]
        updates = [(param, param - learning_rate_var * gparam) for param, gparam in zip(params, gparams)]

        optimize = theano.function(
            inputs=[
                network.event_network.predicate_input_a,
                network.event_network.arg0_input_a,
                network.event_network.arg1_input_a,
                network.event_network.arg2_input_a,
                theano.Param(learning_rate_var, default=learning_rate)
            ],
            outputs=[cost, chain_coherence],
            updates=updates,
        )
        score_vector = theano.function(
            inputs=[
                network.event_network.predicate_input_a,
                network.event_network.arg0_input_a,
                network.event_network.arg1_input_a,
                network.event_network.arg2_input_a,
            ],
            outputs=chain_coherence,
        )

        if slimline_model is not None:
            # Use the slimline version of the model to keep a reference to for sampling purposes
            model = slimline_model

        return NextEventProjectionSampler(projection_size, rhs_projection, optimize, score_vector, model,
                                          neighbour_finder, learning_rate, num_samples)

    def randomize_vector(self):
        # Randomly initialize the RHS vector
        stddev = numpy.sqrt(1. / self.projection_size)
        self.rhs_projection.set_value(
            numpy.asarray(numpy.random.normal(0., stddev, size=(self.num_samples, self.projection_size)), dtype=theano.config.floatX)
        )

    def zero_vectors(self):
        self.rhs_projection.set_value(numpy.zeros((self.num_samples, self.projection_size), dtype=theano.config.floatX))

    def get_vectors(self):
        return self.rhs_projection.get_value()[:, :]

    def get_context_vectors(self, context_chain):
        inputs = list(self.model.get_chain_input_vectors([context_chain]))[0]
        input_arrays = [numpy.array(pos_indices, dtype=numpy.int32) for pos_indices in zip(*inputs)]
        return input_arrays

    def sample_next_input_vectors(self, context_chain, max_iterations=1000, cost_change_threshold=1e-4):
        # Get the inputs that will go on the LHS of the coherence function
        input_arrays = self.get_context_vectors(context_chain)
        # Randomly initialize the RHS vectors
        self.randomize_vector()

        costs = []
        score = 0.
        # Perform a load of SGD updates to the RHS vectors to bring them close to all the LHS vectors
        for i in range(max_iterations):
            cost, score = self.optimize(*input_arrays)
            costs.append(cost)

            # Check whether we've reached the threshold in absolute change in cost
            if len(costs) > 1 and abs(costs[-1] - costs[-2]) <= cost_change_threshold:
                break

        return self.get_vectors(), score

    def sample_next_input_events(self, context_chain, max_iterations=1000, cost_change_threshold=1e-4):
        if self.neighbour_finder is None:
            raise ValueError("no neighbour finder has been given: can't sample next events")

        entity, context_events = context_chain
        sampled_vectors, score = self.sample_next_input_vectors(
            context_chain,
            max_iterations=max_iterations,
            cost_change_threshold=cost_change_threshold
        )
        # Use nearest neighbour search to find possible events from this vector
        neighbours = []
        for i in range(self.num_samples):
            # Get all the neighbours in the same hash bucket
            neighbours.extend(self.neighbour_finder.get_hash_bucket(sampled_vectors[i]))

        # Instead of just scoring these by vector similarity to the sampled vector, actually compute the
        # coherence of the events with the context
        if neighbours:
            neighbour_events = [(sampled_entity, sampled_event)
                                for (v, (sampled_event_source, (sampled_entity, sampled_event))) in neighbours]

            # Get rid of duplicate events -- there will be lots in the bucket
            def dup_key((entity, event)):
                # Key for deciding whether two events are duplicates
                return event.to_string_entity_text({entity: "E0"})
            neighbour_events = remove_duplicates(neighbour_events, key=dup_key)

            # Replace the protagonist in each sampled event with the context protagonist
            candidate_next_events = []
            for (sampled_entity, sampled_event) in neighbour_events:
                entity_position = sampled_event.get_entity_position(sampled_entity)
                event = copy.deepcopy(sampled_event)
                event.substitute_entity(entity_position, entity)
                candidate_next_events.append(event)
            # Get the actual coherence score of the sampled neighbour in the chain context
            candidate_scores = self.model.score_choices(entity, context_events, candidate_next_events)
            return list(zip(candidate_next_events, candidate_scores))
        else:
            return []
