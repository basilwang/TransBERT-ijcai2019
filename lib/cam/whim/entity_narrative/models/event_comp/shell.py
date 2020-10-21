from itertools import islice
from operator import itemgetter
from cam.whim.entity_narrative.models.event_comp.sample import NextEventSampler, NextEventProjectionSampler
from cam.whim.entity_narrative.shell.commands import ModelShell
from whim_common.utils.base import str_to_bool


class EventCompositionShell(ModelShell):
    def do_predict_next(self, line, **kwargs):
        """
        Predict the next event given a context chain

        iterations: number of sampling iterations (default 100)
        pred, arg0, arg1, arg2: fix the predicate/arg0/arg1/arg2 to be a given word and sample the other components
        unique_pred: only show one instance of each predicate

        """
        iterations = int(kwargs.pop("its", 100))
        given_pred = kwargs.pop("pred", None)
        given_arg0 = kwargs.pop("arg0", None)
        given_arg1 = kwargs.pop("arg1", None)
        given_arg2 = kwargs.pop("arg2", None)
        unique_pred = str_to_bool(kwargs.pop("unique_pred", False))

        entities, events, __ = ModelShell.parse_entities_and_events(line)

        sampler = NextEventSampler(
            self.model, pred_given=given_pred, arg0_given=given_arg0, arg1_given=given_arg1, arg2_given=given_arg2
        )
        # Get 50 events, to allow the rescoring to reject rubbish ones
        scored_events = sampler.sample_next_input_events((entities[0], events), 50, max_iterations=iterations,
                                                         rescore=True)

        if unique_pred:
            # Filter events that have the same predicate as a prediction we've already made
            def _filter_unique_predicates(it):
                seen_verbs_uniq = set()
                for e in it:
                    if e[0].verb_lemma not in seen_verbs_uniq:
                        yield e
                        # Don't yield events with this verb in future
                        seen_verbs_uniq.add(e[0].verb_lemma)
            scored_events = list(_filter_unique_predicates(scored_events))

        for event, score in scored_events[:5]:
            print event, score

    def do_predict_next2(self, line, **kwargs):
        """
        Like predict_next, but instead of directly sampling the next event's surface vector it samples
        its deepest projection and then finds nearest neighbours to that.

        """
        iterations = int(kwargs.pop("its", 100))
        entities, events, __ = ModelShell.parse_entities_and_events(line)

        try:
            sampler = self.env["sampler"]
        except KeyError:
            print "No predictor has been prepared. Use load_predictor to prepare one"
            return

        scored_events = list(sampler.sample_next_input_events((entities[0], events), max_iterations=iterations))
        scored_events.sort(key=itemgetter(1), reverse=True)
        for event, score in scored_events[:10]:
            print event, score

    def do_load_predictor(self, line, **kwargs):
        port = int(kwargs.pop("redis_port", 6379))
        topn = int(kwargs.pop("n", 5))

        # Load a nearest neighbour finder for the model: will fail if one's not been created
        finder = self.model.get_nearest_neighbour_finder(self.model_name, port=port)
        print "Loaded nearest neighbour finder using Redis port %d" % port
        sampler = NextEventProjectionSampler.from_model(self.model, finder, num_samples=topn)
        print "Prepared next event sampler to give %d top predictions" % topn

        self.env["finder"] = finder
        self.env["sampler"] = sampler

    def do_neighbours(self, line, **kwargs):
        """
        Like predict_next, but instead of directly sampling the next event's surface vector it samples
        its deepest projection and then finds nearest neighbours to that.

        """
        port = int(kwargs.pop("redis_port", 6379))
        max_num = int(kwargs.pop("max", 10))

        # Load a nearest neighbour finder for the model: will fail if one's not been created
        finder = self.model.get_nearest_neighbour_finder(self.model_name, port=port)
        # Read an event from the shell input
        entities, events, __ = ModelShell.parse_entities_and_events(line, max_events=1)
        # Project this event into the event space
        projection = self.model.project_events([(entities[0], events[0])])[0, :]
        # Search for neighbours to this projected event
        for (entity, event), source, result_vector, score in islice(finder.predict_next_event(projection),
                                                                    max_num):
            print event.to_string_entity_text({entity: "X"}), score

    def do_hash(self, line, **kwargs):
        """
        Like predict_next, but instead of directly sampling the next event's surface vector it samples
        its deepest projection and then finds nearest neighbours to that.

        """
        port = int(kwargs.pop("redis_port", 6379))

        # Load a nearest neighbour finder for the model: will fail if one's not been created
        finder = self.model.get_nearest_neighbour_finder(self.model_name, port=port)
        # Read an event from the shell input
        entities, events, __ = ModelShell.parse_entities_and_events(line, max_events=1)
        # Project this event into the event space
        projection = self.model.project_events([(entities[0], events[0])])[0, :]
        print finder.hash.hash_vector(projection)
