from whim_common.utils.urwid import PALETTE
import urwid


def _exit(*args):
    raise urwid.ExitMainLoop()


class PopUpDialog(urwid.WidgetWrap):
    """A dialog that appears with nothing but a close button """
    signals = ['close']

    def __init__(self, text):
        close_button = urwid.Button("OK")
        urwid.connect_signal(close_button, 'click',
                             lambda button: self._emit("close"))
        w = urwid.Pile([urwid.Text(text), urwid.Divider(), close_button])
        w = urwid.Filler(w)
        w = urwid.LineBox(w)
        super(PopUpDialog, self).__init__(urwid.AttrWrap(w, 'popbg'))

    @staticmethod
    def show(text):
        lines = text.splitlines()
        height = len(lines) + 4
        width = max(len(l) for l in lines) + 4

        pop_up = PopUpDialog(text)
        urwid.connect_signal(pop_up, 'close', _exit)
        main = urwid.Overlay(pop_up, urwid.SolidFill(u'\N{MEDIUM SHADE}'),
                             align='center', width=width,
                             valign='middle', height=height)
        urwid.MainLoop(main, palette=PALETTE, pop_ups=True).run()


class ThingWithAPopUp(urwid.PopUpLauncher):
    def __init__(self, thing, message, left=0, top=0):
        super(ThingWithAPopUp, self).__init__(thing)
        self.message = message

        lines = message.splitlines()
        self.height = len(lines) + 4
        self.width = max(len(l) for l in lines) + 2
        self.left = left
        self.top = top

        self._next_callback = lambda button: self.close_pop_up()

        urwid.connect_signal(self.original_widget, 'click', lambda button: self.open_pop_up())

    def create_pop_up(self):
        pop_up = PopUpDialog(self.message)
        urwid.connect_signal(pop_up, 'close', self._next_callback)
        return pop_up

    def get_pop_up_parameters(self):
        return {'left': self.left, 'top': self.top, 'overlay_width': self.width, 'overlay_height': self.height}

    def open_pop_up(self, callback=None):
        def _callback(button):
            self.close_pop_up()
            if callback is not None:
                callback(button)

        self._next_callback = _callback
        super(ThingWithAPopUp, self).open_pop_up()


def multiple_choice_question(text, options, messages, footer=None, context_widgets=[]):
    body = [urwid.Text("Choose next event:"), urwid.Divider()]
    selected_wrapper = []

    def _keypress(key):
        if key == 'esc':
            _exit()

    def _callback(button, (val, popup_anchor)):
        # Callback for handling the selected value from the list
        selected_wrapper.append(val)
        popup_anchor.open_pop_up(callback=_exit)

    for opt_num, (option, message) in enumerate(zip(options, messages)):
        button = urwid.Button(option)
        popup_anchor = ThingWithAPopUp(button, message, left=25, top=1)
        urwid.connect_signal(button, 'click', _callback, (opt_num, popup_anchor))
        body.append(urwid.AttrMap(popup_anchor, None, focus_map='reversed'))

    bottom_row = [urwid.Text("Press Esc to stop")]
    if footer is not None:
        bottom_row.append(urwid.Text(footer, align='right'))

    top_widgets = [
        urwid.Text(text),
        urwid.Divider(),
    ]
    if context_widgets:
        top_widgets.extend(context_widgets)
        top_widgets.append(urwid.Divider())

    main = urwid.LineBox(
        urwid.Frame(
            urwid.ListBox(urwid.SimpleFocusListWalker(body)),
            header=urwid.Pile(top_widgets),
            footer=urwid.Columns(bottom_row)
        )
    )
    urwid.MainLoop(main, palette=PALETTE, unhandled_input=_keypress, pop_ups=True).run()

    if selected_wrapper:
        return selected_wrapper[0]
    else:
        return None

