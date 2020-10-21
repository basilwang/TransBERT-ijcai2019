from whim_common.utils.urwid import PALETTE
import urwid


def selection_menu(title, choices):
    """
    Show a list of choices and allow the user to select one.

    """
    selected_wrapper = []

    def _callback(button, val):
        # Callback for handling the selected value from the list
        selected_wrapper.append(val)
        raise urwid.ExitMainLoop()

    def _keypress(key):
        if key == 'esc':
            raise urwid.ExitMainLoop()

    body = [urwid.Text(title), urwid.Divider()]
    for (display,val) in choices:
        button = urwid.Button(display)
        urwid.connect_signal(button, 'click', _callback, val)
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))
    main = urwid.LineBox(urwid.ListBox(urwid.SimpleFocusListWalker(body)))

    urwid.MainLoop(main, palette=PALETTE, unhandled_input=_keypress).run()

    if selected_wrapper:
        # Get the selected option
        return selected_wrapper.pop()
    else:
        return None

