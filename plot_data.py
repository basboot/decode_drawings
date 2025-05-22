import matplotlib.pyplot as plt
import numpy as np


# Helper function to plot a single subplot
def _plot_single_subplot_in_grid(ax, x_data, y_data, num_frames, color_indices, cmap, norm_factor,
                               xlabel, ylabel, title, marker, marker_size, equal_axis=False):
    """Helper function to draw a single subplot with common styling."""
    if num_frames > 1:
        # Plot lines connecting the points
        for i in range(num_frames - 1):
            ax.plot([x_data[i], x_data[i + 1]], [y_data[i], y_data[i + 1]],
                     color=cmap(color_indices[i] / norm_factor), linestyle='-')
    # Plot scatter points
    ax.scatter(x_data, y_data, c=color_indices, cmap=cmap, marker=marker, s=marker_size, zorder=2, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if equal_axis:
        ax.axis('equal')

# Main function to create and display the grid of plots
def plot_data_grid(plot_configurations, num_frames, video_id_str, show_plot = True):
    color_indices = np.arange(num_frames)
    cmap = plt.colormaps.get_cmap('inferno')
    # Normalization factor for cmap, handles num_frames=1 case
    norm_factor = (num_frames - 1.0) if num_frames > 1 else 1.0


    """
    Creates a 2x2 grid of plots based on the provided configurations,
    saves it to a PDF, and displays it.
    """
    plt.figure(figsize=(15, 10)) # Adjusted figure size for 2x2 layout

    for i, config in enumerate(plot_configurations):
        ax = plt.subplot(2, 2, i + 1) # Subplot index from 1 to 4
        _plot_single_subplot_in_grid(
            ax,
            config["x_data"],
            config["y_data"],
            num_frames,
            color_indices,
            cmap,
            norm_factor,
            config["xlabel"],
            config["ylabel"],
            config["title"],
            config["marker"],
            config["marker_size"],
            config.get("equal_axis", False) # Use .get for optional 'equal_axis'
        )

    plt.tight_layout()
    plt.savefig(f"output/decode{video_id_str}.pdf", format="pdf", bbox_inches="tight")
    if show_plot:
        plt.show()
