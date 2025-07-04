import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.utilities.utils import rot


def plot_elliptic_extent(m, p, ax=None, color='b', alpha=1., label=None, linestyle=None, show_center=True, fill=False,
                         show_orientation=False, zorder=None):
    """
    Add matplotlib ellipse patch based on location and extent information about vehicle
    :param m: Kinematic information as 4D array [x, y, velocity_x, velocity_y]
    :param p: extent information as 3D array [orientation, length, width]. Orientation in radians.
    :param ax: matplotlib axis to plot on or None (will use .gca() if None)
    :param color: Color to plot the ellipse and marker in
    :param alpha: Alpha value for plot
    :param label: Label to apply to plot or None to not add a label
    :param linestyle: Linestyle parameter passed to matplotlib
    :param show_center: If True, will additionally add an x for the center location
    :param fill: Whether to fill the ellipse
    """
    if ax is None:
        ax = plt.gca()
    theta, l1, l2 = p
    theta = np.rad2deg(theta)
    # patches.Ellipse takes angle counter-clockwise
    el = patches.Ellipse(xy=m[:2], width=l1, height=l2, angle=theta, fill=fill, color=color, label=label,
                         alpha=alpha, linestyle=linestyle, zorder=zorder)
    if show_center:
        ax.scatter(m[0], m[1], color=color, marker='x', zorder=zorder)

    if show_orientation:
        direction_vector = rot(p[0]) @ np.array([l1 / 2, 0]) + m[:2]
        ax.plot([m[0], direction_vector[0]], [m[1], direction_vector[1]], color=color, zorder=zorder)

    ax.add_patch(el)


def mark_trajectory_turns_in_plot(traj_type, curve_linewidth: int = 14, color='k'):
    """
    Adds marks for the standard trajectory in the plot (12-17, 23-28, 41-46, 52-57).

    Note that the parameters should generally not be used, except for edge cases, in order to ensure that the
    visualization is the same between all figures.
    """
    xlim = np.array(plt.gca().get_xlim())
    ylim = plt.gca().get_ylim()

    if traj_type == 'butterfly':
        plt.plot([12, 17], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
        plt.plot([23, 28], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
        plt.plot([41, 46], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
        plt.plot([52, 57], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
    elif traj_type == 'nimitz':
        plt.plot([6, 9], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
        plt.plot([17, 22], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
        plt.plot([26, 31], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
    else:
        pass

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)


def plot_elliptic_state(full_state, ax=None, color='b', alpha=1., label=None, linestyle=None, show_center=True,
                        fill=False, show_orientation=None, zorder=None):
    """Wraps around plot_elliptic_extent. This method takes a 7D state and extracts m and p from it"""
    m = full_state[:2]
    p = full_state[4:]
    if show_orientation is None:
        # show_orientation = False
        show_orientation = not fill

    return plot_elliptic_extent(m, p, ax=ax, color=color, alpha=alpha, label=label, linestyle=linestyle,
                                show_center=show_center, fill=fill, show_orientation=show_orientation, zorder=zorder)
