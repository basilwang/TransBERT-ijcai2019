from cam.whim.entity_narrative.models.base.vectorspace.shell import VectorSpaceModelShell
from cam.whim.entity_narrative.shell.commands import ModelShell


class ArgumentCompositionShell(VectorSpaceModelShell):
    def do_predict_args(self, line, **kwargs):
        entities, events, line = ModelShell.parse_entities_and_event(line)
        print "Predicting args for predicate %s" % self.model.get_model_predicate_repr(entities[0], events[0])
        arg_predictions = self.model.predict_args(entities[0], events[0], limit=10)
        for arg_num, predictions in enumerate(arg_predictions):
            print "Predictions for arg %d:" % arg_num
            for word, score in predictions:
                print "  %s (%f)" % (word, score)

    def do_test(self, line, **kwargs):
        entities, events, line = ModelShell.parse_entities_and_event(line)
        pred = self.model.get_model_predicate_repr(entities[0], events[0])
        print "Testing reconstruction for predicate %s" % pred
        pred_index = self.model.pred_vocab[pred].index
        self.model.projection_model.test([pred_index], [-1], [-1], [-1])
