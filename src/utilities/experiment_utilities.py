"""
Contains functions used during the conducted experiments, both for result acquisition and visualization.
"""
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.evaluation.generate_trajectory_data import get_trajectory_data
from src.utilities.metrics import gwd_full_state, iou_full_state, errors_per_state_component
from src.utilities.visuals import mark_trajectory_turns_in_plot


def get_error_data(tracker_dict: dict,
                   rng: np.random.Generator,
                   settings: dict,
                   n_monte_carlo_runs: int,
                   use_tqdm=True,
                   extended_metrics=False,
                   ):
    """
    Given trackers and needed settings, run the trackers on the trajectories, evaluate the errors and runtimes, and
    return the results
    :param tracker_dict: Dictionary of trackers, each entry containing a dict which contains "instance", a not init.
    Tracker instance, and color, a valid matplotlib color string
    :param rng: RNG object used throughout the experiment
    :param settings: Setting used to get the reference trajectory data to be run on
    :param n_monte_carlo_runs: Number of monte carlo runs to perform
    :return: (errors, runtimes) as dicts
    """
    theta_bound = np.pi  # saved thetas will be computed modulo this, choose 1 or 2pi

    # get one singular run already in order to know how long each of the tracks will be
    true_states, measurements = get_trajectory_data(rng=rng, **settings)
    n_steps = len(true_states)  # get lengths

    # prepare error_dicts
    tracker_names = np.array(list(tracker_dict.keys()))
    errors = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
    runtimes = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
    if extended_metrics:
        gt_name = "True"
        ious = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        err_m = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        err_theta = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        err_l1 = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        err_l2 = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        l1s = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        l2s = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        thetas = {tracker_id: np.zeros((n_monte_carlo_runs, n_steps)) for tracker_id in tracker_names}
        l1s[gt_name] = np.zeros((n_monte_carlo_runs, n_steps))
        l2s[gt_name] = np.zeros((n_monte_carlo_runs, n_steps))
        thetas[gt_name] = np.zeros((n_monte_carlo_runs, n_steps))
    for run_ix in tqdm(range(n_monte_carlo_runs), disable=(not use_tqdm)):
        current_run_trackers = deepcopy(tracker_dict)
        this_run_states = {tracker_id: [] for tracker_id in tracker_names}
        true_states, measurements = get_trajectory_data(rng=rng, **settings)

        for step_ix, measurements_and_gt in enumerate(zip(measurements, true_states)):
            Z, gt = measurements_and_gt
            if extended_metrics:
                l1s[gt_name][run_ix, step_ix] = gt[5]
                l2s[gt_name][run_ix, step_ix] = gt[6]
                thetas[gt_name][run_ix, step_ix] = gt[4] % theta_bound
            for tracker_id in tracker_names:
                start_time = time.time()
                # update
                current_run_trackers[tracker_id]["instance"].update(Z)
                # extract state
                this_run_states[tracker_id].append(current_run_trackers[tracker_id]["instance"].get_state())

                # save error
                errors[tracker_id][run_ix, step_ix] = gwd_full_state(this_run_states[tracker_id][-1], gt)
                if extended_metrics:
                    ious[tracker_id][run_ix, step_ix] = iou_full_state(this_run_states[tracker_id][-1], gt)
                    e_m, e_t, e_l1, e_l2 = errors_per_state_component(this_run_states[tracker_id][-1], gt)
                    err_m[tracker_id][run_ix, step_ix] = e_m
                    err_theta[tracker_id][run_ix, step_ix] = e_t
                    err_l1[tracker_id][run_ix, step_ix] = e_l1
                    err_l2[tracker_id][run_ix, step_ix] = e_l2
                    l1s[tracker_id][run_ix, step_ix] = this_run_states[tracker_id][-1][5]
                    l2s[tracker_id][run_ix, step_ix] = this_run_states[tracker_id][-1][6]
                    thetas[tracker_id][run_ix, step_ix] = this_run_states[tracker_id][-1][4] % theta_bound

                # predict
                current_run_trackers[tracker_id]["instance"].predict()
                runtimes[tracker_id][run_ix, step_ix] = (time.time() - start_time) * 1000
        pass
    if extended_metrics:
        return errors, runtimes, ious, err_m, err_theta, err_l1, err_l2, l1s, l2s, thetas
    else:
        return errors, runtimes


def plot_error_over_time(tracker_dict, errors, settings, runtimes, tracker_names=None,
                         filename=None, verbose=False, outside_legend=True, mark_turns=False,
                         include_median_for=None, order_legend_descending=False, wide=False,
                         ylabel=None,
                         disable_show=False,
                         gt_key_for_percent_visuals=None,
                         mean_init_ix=0):
    """Create a plot of the errors of a set of trackers, as a line plot over time"""
    if tracker_names is None:
        tracker_names = np.array(list(tracker_dict.keys()))
    else:
        tracker_names = np.array(tracker_names)

    to_pop = []
    for t_name in tracker_names:
        if t_name not in tracker_dict.keys():
            print(f"{t_name} not in trackers, skipping")
            to_pop.append(t_name)
    for name in to_pop:
        tracker_names.pop(tracker_names.index(name))
    if len(tracker_names) == 0:
        print("All trackers skipped or none given, skipping")
        return

    if include_median_for == 'ALL':
        include_median_for = tracker_names

    longest_name_length = np.max([len(t) for t in tracker_names])
    # iterate over trackers_elliptical
    if verbose:
        print("GWD:")

    # default order is to just take them as-is
    order = list(range(len(tracker_names)))
    if order_legend_descending:
        # generate a new order mask with descending mean errors
        mean_errors_in_original_order = [
            np.average(np.average(errors[t_id], axis=0)[mean_init_ix:])
            for t_id in tracker_names
        ]
        order = np.argsort(mean_errors_in_original_order)[::-1]
    if wide:
        fig, axs = plt.subplots(1, 1, figsize=(24, 8))
    for t_id in tracker_names[order]:
        color = tracker_dict[t_id]["color"]
        tracker_error_avg = errors[t_id]
        if gt_key_for_percent_visuals is not None:
            # divide by ground truth and convert to %
            tracker_error_avg = 100 * tracker_error_avg / errors[gt_key_for_percent_visuals]
        if include_median_for is None or t_id not in include_median_for:
            avg_over_runs = np.average(tracker_error_avg, axis=0)
            plt.plot(avg_over_runs, color=color, label=f"{t_id}: {np.average(avg_over_runs[mean_init_ix:]):5.2f}")
            if verbose:
                print(f"\t{t_id:{longest_name_length}s}: {np.average(avg_over_runs):5.2f}")
        else:
            avg_over_runs = np.average(tracker_error_avg, axis=0)
            plt.plot(avg_over_runs, color=color, label=f"{t_id} [Mean]: {np.average(avg_over_runs):5.2f}",
                     linestyle='--')
            median_over_runs = np.median(tracker_error_avg, axis=0)
            plt.plot(median_over_runs, color=color, label=f"{t_id} [Median]: {np.average(median_over_runs):5.2f}")
    if gt_key_for_percent_visuals is not None:
        plt.axhline(100, c='k')
    if verbose:
        print("Time / ms:")
    for t_id in tracker_names[order]:
        tracker_time_avg = runtimes[t_id]
        time_avg_over_runs = np.average(tracker_time_avg, axis=0)
        if verbose:
            print(f"\t{t_id:{longest_name_length}s}: {np.average(time_avg_over_runs):5.2f}")

    plt.ylim([0, plt.ylim()[1]])

    if mark_turns:
        mark_trajectory_turns_in_plot(settings["trajectory_type"])

    # show plot
    if outside_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        # plt.legend(loc='upper right')
        if ylabel is None:
            plt.legend()
        else:
            if "gwd" in ylabel.lower():
                plt.legend(loc='upper left')
            else:
                plt.legend(loc="lower right")
    plt.xlabel("Step in trajectory")
    ylabel = r"Squared GWD / m$^2$" if ylabel is None else ylabel
    plt.ylabel(ylabel)
    # plt.title(title)
    if not disable_show:
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.draw()
            plt.savefig(filename)
            plt.close()
