"""
Script to compute results for quantitative experiments.
Settings are defined at the end of the file, below "if __name__ == '__main__':"
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from src.evaluation.evaluation_management import EvaluationManager
from src.utilities.experiment_utilities import get_error_data, plot_error_over_time


def generate_json_data(mode,
                       tracker_names,
                       target_dir,
                       n_runs,
                       seed=None):
    seed = np.random.randint(1, 9999999) if seed is None else int(seed)
    print(f"Running with seed {seed} for mode {mode}")
    rng = np.random.default_rng(seed)

    manager = EvaluationManager(mode=mode, seed=seed)
    settings = manager.get_settings()
    trackers = manager.get_tracker_dict(tracker_names=tracker_names,
                                        settings=settings)

    gwds, runtimes, ious, err_m, err_theta, err_l1, err_l2, l1_list, l2_list, theta_list = get_error_data(
        tracker_dict=trackers,
        rng=rng,
        settings=settings,
        n_monte_carlo_runs=n_runs,
        extended_metrics=True
    )
    tqdm.write("")
    data_dict = {
        "metadata": {
            "seed": seed,
            "n_runs": n_runs,
            "settings": settings,
            "date": datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        },
        "tracker_dict": trackers,
        "errors": gwds,
        "runtimes": runtimes,
        "ious": ious,
        "err_m": err_m,
        "err_theta": err_theta,
        "err_l1": err_l1,
        "err_l2": err_l2,
        "l1": l1_list,
        "l2": l2_list,
        "theta": theta_list,
    }
    if target_dir[-1] != "/":
        target_dir = target_dir + "/"
    target_file = os.path.join(target_dir, f"last_{mode}.npy")
    np.save(target_file, data_dict)
    tqdm.write(f"Saved data to {target_file}")


def print_runtime_stats(runtimes):
    print("Runtime analysis:")
    for method_k in runtimes:
        # runtimes[method_k].shape == (n_runs, n_steps)
        d = runtimes[method_k]
        print(f"\t{method_k}: Mean = {np.mean(d):.2f}, "
              f"from {np.min(d):.2f} to {np.max(d):.2f} "
              f"with a std. dev. of {np.std(d):.2f}")


def print_semi_axis_details(tracker_dict, errors, settings, runtimes, tracker_names=None,
                            filename=None, verbose=False, outside_legend=True, mark_turns=False,
                            include_median_for=None, order_legend_descending=False, wide=False,
                            ylabel=None,
                            disable_show=False,
                            gt_key=None):
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

    for t_id in tracker_names:
        color = tracker_dict[t_id]["color"]
        tracker_error = np.array(errors[t_id])  # n_runs x n_steps

        for i in range(tracker_error.shape[1]):
            step = tracker_error[:, i]
            print(i)
            print("\tmin", step.min())
            print("\tmax", step.max())
            print("\tavg", step.mean())
            print("\tstd", step.std())

        plt.plot(np.min(tracker_error, axis=0), c='k')
        plt.plot(np.max(tracker_error, axis=0), c='k')
        plt.plot(np.mean(tracker_error, axis=0), c='b')
        plt.show()


def print_percent_best_data(err_data, stat_fct=np.mean):
    if len(err_data.keys()) < 2:
        return
        # average for each scenario:

    names = list(err_data.keys())
    data = stat_fct(err_data[names[0]], axis=1)
    for n in names[1:]:
        data = np.vstack([data, stat_fct(err_data[n], axis=1)])
    n_scenarios = data.shape[1]
    best_algo_id = np.argmin(data, axis=0)
    longest_name_length = np.max([len(t) for t in names])
    print("Number of times algorithm was best:")
    for i, n in enumerate(names):
        n_times_best = np.sum(best_algo_id == i)
        percent_value = 100 * n_times_best / n_scenarios
        print(f"\t{n:{longest_name_length}s}: "
              f"{n_times_best:{int(np.log10(n_scenarios)) + 1}d}/{n_scenarios}\t({percent_value:2.2f}%)")


def plot_json_data_individual(mode,
                              source_dir,
                              image_dir,
                              print_axis_details=False,
                              print_percent_best=False,
                              print_runtimes=False,
                              ):
    data_file = os.path.join(source_dir, f"last_{mode}.npy")
    data: dict = np.load(data_file, allow_pickle=True).astype(dict).item()
    if print_runtime or print_axis_details or print_percent_best:
        print(f"Results of {data['metadata']['n_runs']} runs for seed {data['metadata']['seed']} from {data_file}")

    include_init_in_avg = True

    kwargs = dict(
        tracker_dict=data["tracker_dict"],
        settings=data["metadata"]["settings"],
        runtimes=data["runtimes"],
        outside_legend=False,
        mark_turns=True,
        order_legend_descending=True,
        disable_show=True,
        mean_init_ix=0 if include_init_in_avg else 1,
        include_median_for=None,
    )
    if print_runtimes:
        print_runtime_stats(runtimes=data["runtimes"])

    if print_axis_details:
        print_semi_axis_details(errors=data["l1"],
                                ylabel="Size / $m$",
                                gt_key="True",
                                **kwargs)
    if print_percent_best:
        print_percent_best_data(data["errors"])

    # GWD
    plt.close()
    plot_error_over_time(errors=data["errors"], **kwargs)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(image_dir, f"last_gwd_{mode}"), bbox_inches='tight')
        plt.close()

    # THETA
    plot_error_over_time(errors=data["err_theta"], ylabel="Angular error / rad", **kwargs)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(image_dir, f"last_theta_{mode}"), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=100000)
    plt.style.use("../../data/stylesheets/default.mplstyle")
    force_generation = True
    visuals_mode = "all"  # 'gwd' 'errors' 'convergence' 'all'
    save_images = True
    print_runtime = False

    seed = 42
    n_monte_carlo_runs = 500
    # select experiments settings to run for:
    mode = [
        # === Standard experiments:
        "easy-random-nimitz",
        "hard-random-nimitz",
        "easy-random-butterfly-variable",
        "hard-random-butterfly-variable",

        # === Number of particles:
        # "easy-random-nimitz-b",

        # === Motion process noise analysis
        "easy-random-cv", "easy-random-cv-motion", "easy-random-cv-motionvelo",
        "hard-random-cv", "hard-random-cv-motion", "hard-random-cv-motionvelo",
    ]

    # select trackers to use:
    trackers = [
        # === Standard experiments:
        "MEM-EKF*",
        "VBRM",
        "MEM-RBPF",
        "PAKF",

        #  === Evaluating number of particles:
        # "MEM-RBPF #10", "MEM-RBPF #25", "MEM-RBPF #50", "MEM-RBPF #100"

        # === Optionally, compare different resampling approaches:
        # "MEM-RBPF [multinomial]", "MEM-RBPF [stratified]", "MEM-RBPF [systematic]", "MEM-RBPF [residual]",
    ]

    # general settings
    toplevel_dir = "../../output/results/"
    data_directory = toplevel_dir + "result_data/"
    image_directory = toplevel_dir + "output_images/" if save_images else None


    def generate(mode_to_use):
        if isinstance(mode_to_use, str):
            generate_json_data(
                tracker_names=trackers,
                mode=mode_to_use,
                target_dir=data_directory,
                n_runs=n_monte_carlo_runs,
                seed=seed,
            )
        else:
            for m in mode_to_use:
                generate(m)


    def visualize(mode_to_use):
        if isinstance(mode_to_use, str):
            plot_json_data_individual(mode_to_use,
                                      data_directory,
                                      image_directory,
                                      print_percent_best=False,
                                      print_runtimes=print_runtime,
                                      )
        else:
            for m in mode_to_use:
                visualize(m)


    if force_generation:
        print(f"Forcing re-generation of results for mode '{mode}'...\n")
        generate(mode)
        visualize(mode)
    else:
        try:
            visualize(mode)
        except FileNotFoundError:
            print(f"For the given settings, no saved data was found. Generating results...\n")
            generate(mode)
            visualize(mode)
