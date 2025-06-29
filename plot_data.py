import matplotlib.pyplot as plt
import numpy as np
import math


# Helper function to plot a single subplot
def _plot_single_subplot_in_grid(ax, x_data, y_data, num_frames, color_indices, cmap, norm_factor,
                               xlabel, ylabel, title, marker, marker_size, color=None, equal_axis=False, scatter_only=False, mirror_y = False):
    if mirror_y:
        y_data = [-y for y in y_data]
    
    if num_frames > 1 and not scatter_only:
        # Plot lines connecting the points
        for i in range(len(x_data) - 1):
            ax.plot([x_data[i], x_data[i + 1]], [y_data[i], y_data[i + 1]],
                     color=cmap(color_indices[i] / norm_factor) if color is None else color, linestyle='-')
    # Plot scatter points
    # TODO: fix single color
    ax.scatter(x_data, y_data, c=color_indices if color is None else color, cmap=cmap, marker=marker, s=marker_size, zorder=2, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if equal_axis:
        ax.axis('equal')

# Main function to create and display the grid of plots
def plot_data_grid(plot_configurations, num_frames, video_id_str, show_plot = True, output_dir = "output"):
    color_indices = np.arange(num_frames)
    cmap = plt.colormaps.get_cmap('inferno')
    # Normalization factor for cmap, handles num_frames=1 case
    norm_factor = (num_frames - 1.0) if num_frames > 1 else 1.0


    n_plots = len(plot_configurations)
    n_rows = math.ceil(n_plots / 2) # 2 plots per row

    plt.figure(figsize=(15, n_rows * 5)) # Adjust figure to number of plots

    for i, config in enumerate(plot_configurations):
        ax = plt.subplot(n_rows, 2, i + 1) 
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
            config["color"],
            config.get("equal_axis", False), # Use .get for optional 'equal_axis',
            config.get("scatter_only", False), # Use .get for optional 
            config.get("mirror_y", False) # Use .get for optional 
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/decode{video_id_str}.pdf", format="pdf", bbox_inches="tight")
    if show_plot:
        plt.show()
