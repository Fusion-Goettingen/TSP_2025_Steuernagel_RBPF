"""
Run a single MC run of a setting, plotting the errors and showing the estimation results for all trackers
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector

from src.evaluation.generate_trajectory_data import get_trajectory_data
from src.utilities.metrics import gwd_full_state
from src.utilities.visuals import plot_elliptic_state, mark_trajectory_turns_in_plot
from src.evaluation.evaluation_management import EvaluationManager
from src.utilities.utils import rot


def custom_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


class VisHelper:
    def __init__(self):
        self.flag = True

    def disable_flag(self, unused_parameter):
        self.flag = False


def main(tracker_names,
         mode,
         seed=None,
         scatter_measurements=True,
         steps=None,
         plot_errors=True,
         save_dir=None,
         steps_zoom=None,
         add_arrow_at_init_loc=False,
         ):
    seed = np.random.randint(1, 9999999) if seed is None else int(seed)
    print(f"Running with seed {seed}")
    rng = np.random.default_rng(seed)

    manager = EvaluationManager(mode=mode, seed=seed)
    settings = manager.get_settings()
    tracker_dict = manager.get_tracker_dict(tracker_names=tracker_names,
                                            settings=settings)

    current_run_trackers = deepcopy(tracker_dict)
    tracker_states = {tracker_id: [] for tracker_id in tracker_names}
    true_states, measurements = get_trajectory_data(rng=rng, **settings)
    n_steps = len(true_states)

    errors = {tracker_id: np.zeros((n_steps,)) for tracker_id in tracker_names}
    runtimes = {tracker_id: np.zeros((n_steps,)) for tracker_id in tracker_names}

    for step_ix, measurements_and_gt in enumerate(zip(measurements, true_states)):
        Z, gt = measurements_and_gt

        for tracker_id in tracker_names:
            # update
            pre_update = time.time()
            current_run_trackers[tracker_id]["instance"].update(Z)
            runtimes[tracker_id][step_ix] += time.time() - pre_update
            # extract state
            tracker_states[tracker_id].append(current_run_trackers[tracker_id]["instance"].get_state())
            # save error
            errors[tracker_id][step_ix] = gwd_full_state(tracker_states[tracker_id][-1], gt)
            # predict
            pre_predict = time.time()
            current_run_trackers[tracker_id]["instance"].predict()
            runtimes[tracker_id][step_ix] += time.time() - pre_predict

    # print runtimes
    avg_runtimes = {t: np.average(runtimes[t]) for t in tracker_names}
    print("Runtimes per step in ms:")
    for tracker_id in tracker_names:
        print(f"{tracker_id}: {avg_runtimes[tracker_id] * 1000:.1f}")

    # plot errors
    avg_errors = {t: np.average(errors[t]) for t in tracker_names}
    if plot_errors:
        for t in tracker_names:
            plt.plot(errors[t], c=tracker_dict[t]["color"], label=f"{t}: {avg_errors[t]:.2f}")
        plt.legend()
        mark_trajectory_turns_in_plot(traj_type=settings["trajectory_type"])
        plt.show()

    print("\nErrors for methods during steps")
    errors = {t: errors[t][steps] for t in tracker_names} if steps is not None else errors
    for t in tracker_names:
        print(f"Errors for {t}:")
        print(errors[t].round(2))
        print(f"Average: {np.average(errors[t]):.2f}")
        print("---")
    for i in range(len(true_states)):
        if steps is not None and i not in steps:
            continue
        plot_elliptic_state(true_states[i], color='grey', fill=True, alpha=0.65)
        if add_arrow_at_init_loc and i == 0:
            rmat = rot(true_states[i][4])
            dx, dy = rmat @ np.array([true_states[i][5] * 0.5, 0])
            offset_side = rmat @ np.array([0, true_states[i][6]])
            x, y = true_states[i][:2] + offset_side
            width = 0.15
            plt.arrow(x, y, dx, dy,
                      width=width,
                      head_width=5 * width,
                      head_length=8 * width,
                      length_includes_head=True)
        if scatter_measurements:
            plt.scatter(*measurements[i].T, c='black', alpha=0.9, s=5)
        for tracker_id in tracker_names:
            if steps is None:
                add_label = i == 0
            else:
                add_label = i == steps[0]
            plot_elliptic_state(tracker_states[tracker_id][i],
                                color=tracker_dict[tracker_id]["color"],
                                label=tracker_id if add_label else None,
                                show_orientation=False,
                                )

    # ==== INSET AXIS
    if steps_zoom is not None:
        if "nimitz" in mode:
            inset_loc = 'center'
            inset_zoom = 2.5
            inset_corners = [3, 4]
        elif "butterfly" in mode:
            inset_loc = 'upper right'
            inset_zoom = 1.25
            inset_corners = [2, 3, 1, 4]
        else:
            raise ValueError(f"ax inset only implemented for {mode}")
        org_ax = plt.gca()
        axins = zoomed_inset_axes(org_ax, inset_zoom, loc=inset_loc)

        plt.sca(axins)
        for i in steps_zoom:
            plot_elliptic_state(true_states[i], color='grey', fill=True, alpha=0.65)
            plt.scatter(*measurements[i].T, c='black', alpha=0.9, s=5)
            for tracker_id in tracker_names:
                plot_elliptic_state(tracker_states[tracker_id][i],
                                    color=tracker_dict[tracker_id]["color"],
                                    label=False,
                                    show_orientation=False,
                                    )
        plt.xticks([])
        plt.yticks([])
        if "butterfly" in mode:
            custom_mark_inset(org_ax, axins,
                              loc1a=inset_corners[0],
                              loc2a=inset_corners[1],
                              loc1b=inset_corners[2],
                              loc2b=inset_corners[3],
                              fc="none", ec="0.5", linestyle='--')
        elif "nimitz" in mode:
            mark_inset(org_ax, axins,
                       loc1=inset_corners[0],
                       loc2=inset_corners[1],
                       fc="none", ec="0.5", linestyle='--')
        plt.sca(org_ax)
    # ====
    plt.xlabel("$m_1$ / m")
    plt.ylabel("$m_2$ / m")
    plt.axis('equal')
    if "nimitz" in mode:
        plt.ylim(np.array(plt.ylim()) + 25)
    elif "butterfly" in mode:
        plt.ylim(np.array(plt.ylim()) + 15)
    plt.legend(loc='upper left')
    if save_dir is None:
        plt.show()
    else:
        save_dir = save_dir + "/" if save_dir[-1] != "/" else save_dir
        plt.savefig(f"{save_dir}example_{mode}_{seed}", bbox_inches='tight')
        plt.close()


def generate_example_qualitative_results():
    plt.style.use("../../data/stylesheets/default.mplstyle")
    save_dir = "../../output/results/qualitative_results"
    trackers = ["VBRM", "MEM-RBPF"]
    steps = None

    init_arrow = True
    main(
        tracker_names=trackers,
        mode="easy-random-nimitz",
        seed=18168,
        plot_errors=False,
        steps=steps,
        save_dir=save_dir,
        steps_zoom=list(range(8, 13)),
        add_arrow_at_init_loc=init_arrow,
    )

    main(
        tracker_names=trackers,
        mode="easy-random-butterfly-variable",
        seed=42660,
        plot_errors=False,
        steps=steps,
        save_dir=save_dir,
        steps_zoom=list(range(40, 46)),
        add_arrow_at_init_loc=init_arrow,
    )

    sys.exit(0)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=100000)
    generate_example_qualitative_results()
