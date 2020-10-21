

class NarrativeChainPredictor(object):
    """
    Abstract class for narrative chain models that are able to make predictions of next events
    given a context.

    """
    # This should be overridden as True if the model provides predictions of event including arguments
    PREDICTIONS_WITH_ARGS = False

    def predict_next_event(self, entity, context_events):
        """
        Iterable of events, starting with the highest weighted. Doesn't need to be a list:
        could be, for example, an unbounded generator.

        :param entity: chain entity
        :param context_events: events so far
        :return: pairs of (event, score)
        """
        raise NotImplementedError("predict_next_event() should be implemented by %s" % type(self).__name__)
