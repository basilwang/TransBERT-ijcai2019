from cam.whim.entity_narrative.shell.commands import ModelShell


class VectorSpaceModelShell(ModelShell):
    def do_project_chain(self, line, **kwargs):
        entities, events, line = ModelShell.parse_event_context(line)
        # Get the model's projection of these events as a single chain
        vectors = self.model.project_chains([(entities[0], events)])
        print vectors[0]

    def do_project(self, line, **kwargs):
        entities, events, line = ModelShell.parse_event_context(line)
        # Get the model's projection of these events
        vectors = self.model.project_events([(entities[0], e) for e in events])
        for event, vector in zip(events, vectors):
            print "%s: %s" % (event, vector)
