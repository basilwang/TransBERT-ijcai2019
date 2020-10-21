from itertools import islice
from cam.whim.entity_narrative.shell.commands import ModelShell


class NarrativePredictorShell(object):
    """
    Mixin for shells of prediction models (those inheriting from NarrativeChainPredictor)

    """
    def do_predict(self, line, **kwargs):
        entities, events, line = ModelShell.parse_event_context(line)
        # Make ordered predictions
        predictions = self.model.predict_next_event(entities[0], events)
        for event, score in islice(predictions, 10):
            print event, score