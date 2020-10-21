import matplotlib
matplotlib.use('PDF', warn=True)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt


COLOURS = ['r', 'g', 'b', 'y', 'm', 'c', '#B1C7CA', '#90F0A0', '#800000', '#FFC0C0', '#008000', '#000080',
           '#404000', '#00FFFF', '#FF00FF', '#FFC0FF']

MARKERS = ["o", "*", "v", "^", "s", "+", "x", "<", ">", "1", "4", "p", "D", "d"]

def _get_marker(num):
    if num < len(MARKERS):
        return MARKERS[num]
    else:
        return "$%d$" % (num - len(MARKERS) + 1)


RED_GREEN_CMAP = LinearSegmentedColormap(
    "red_green",
    {
        "red": [(0.0,  1.0, 1.0),
                (0.5,  1.0, 1.0),
                (0.8,  0.8, 0.8),
                (1.0,  0.0, 0.0)],
        "green": [(0.0,  0.0, 0.0),
                  (0.2,  0.8, 0.8),
                  (0.5,  1.0, 1.0),
                  (1.0,  1.0, 1.0)],
        "blue": [(0.0, 0.0, 0.0),
                 (0.2,  0.8, 0.8),
                 (0.5,  1.0, 1.0),
                 (0.8,  0.8, 0.8),
                 (1.0,  0.0, 0.0)],
    }
)

plt.register_cmap(cmap=RED_GREEN_CMAP)


def plot_costs(output_filename, *data_columns, **kwargs):
    return_figure = kwargs.get("return_figure", False)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for col, data_column in enumerate(data_columns):
        data, name = data_column

        # Plot this y-data as a line
        ax.plot(list(range(len(data))), data,
                marker="o", linestyle="-", color=COLOURS[col % len(COLOURS)],
                label=name)

    lgd = ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

    if output_filename is not None:
        # Output the plot to a file
        plt.savefig(output_filename, bbox_inches="tight", additional_artists=[lgd])
    if return_figure:
        return ax, fig
    return ax


def plot_learning_curves(output_filename, data_sizes, *data_columns, **kwargs):
    axis = kwargs.get("axis", None)
    size = kwargs.get("size", (8, 6))
    pair_scaler = 2 if kwargs.get("pairs", False) else 1
    if axis is None:
        fig = plt.figure(figsize=size)
        axis = fig.add_subplot(111)

    for col, data_column in enumerate(data_columns):
        data, name = data_column

        # Plot this y-data as a line, with the data sizes as the x values
        axis.plot(data_sizes[:len(data)], data,
                  marker=_get_marker(col / pair_scaler / len(COLOURS)), linestyle=["-", "--"][col % pair_scaler],
                  color=COLOURS[(col / pair_scaler) % len(COLOURS)],
                  label=name)

    lgd = axis.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

    # Output the plot to a file
    if output_filename is not None:
        plt.savefig(output_filename, bbox_inches="tight", additional_artists=[lgd])
    return axis, lgd


def plot_multiple_learning_curves(output_filename, data_sizes, *data_groups, **kwargs):
    y_labels = kwargs.get("y_labels", [None] * len(data_groups))

    fig = plt.figure(figsize=(8.0, 6.0*len(data_groups)))
    artists = []

    for group_num, (data_group, y_label) in enumerate(zip(data_groups, y_labels)):
        axis = fig.add_subplot(len(data_groups), 1, group_num+1)
        axis.set_ylabel(y_label)
        __, lgd = plot_learning_curves(None, data_sizes, *data_group, axis=axis)
        artists.append(lgd)

    plt.savefig(output_filename, bbox_inches="tight", additional_artists=artists)