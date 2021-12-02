"""compute:
Worsecrossbars' main module and entrypoint.
"""
import argparse
import gc
import json
import logging
import platform
import signal
import sys
from multiprocessing import Process
from pathlib import Path

import numpy as np

from worsecrossbars.backend.mlp_trainer import create_datasets
from worsecrossbars.backend.simulation import run_simulation
from worsecrossbars.backend.simulation import train_models
from worsecrossbars.backend.simulation import training_validation_metrics
from worsecrossbars.plotting.curves_plotting import accuracy_curves
from worsecrossbars.plotting.curves_plotting import training_validation_curves
from worsecrossbars.utilities.dropbox_upload import DropboxUpload
from worsecrossbars.utilities.initial_setup import main_setup
from worsecrossbars.utilities.io_operations import create_output_structure
from worsecrossbars.utilities.io_operations import read_external_json
from worsecrossbars.utilities.io_operations import read_webhook
from worsecrossbars.utilities.io_operations import user_folders
from worsecrossbars.utilities.json_handlers import validate_json
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier
from worsecrossbars.utilities.parameter_validator import validate_parameters


def stop_handler(signum, _):
    """This function handles stop signals transmitted by the Kernel when the script terminates
    abruptly/unexpectedly."""

    logging.error(
        "Simulation terminated unexpectedly due to Signal %s",
        signal.Signals(signum).name,
    )
    if command_line_args.teams:
        sims = json_object["simulations"]
        teams.send_message(
            f"Using parameters:\n{sims}\nSignal:{signal.Signals(signum).name}",
            title="Simulation terminated unexpectedly",
            color="b90e0a",
        )
    gc.collect()
    sys.exit(1)


def worker(mnist_dataset, simulation_parameters, _output_folder, _teams=None):
    """A worker, an async class that handles the heavy-lifting computation-wise."""

    number_hidden_layers = simulation_parameters["number_hidden_layers"]
    fault_type = simulation_parameters["fault_type"]
    noise_variance = simulation_parameters["noise_variance"]

    logging.info("Attempting simulation with following parameters: %s", simulation_parameters)

    if _teams:
        _teams.send_message(
            f"Using parameters:\n{simulation_parameters}",
            title="Started simulation",
            color="ffca33",
        )

    percentages = np.arange(0, 1.01, 0.01).round(2)

    weights_list, histories_list = train_models(
        mnist_dataset, simulation_parameters, epochs=10, batch_size=100
    )

    # Computing training and validation loss and accuracy by averaging over all the models trained
    # in the previous step
    training_validation_data = training_validation_metrics(histories_list)

    logging.info(
        "[%dHL_%s_%.2fNV] Done training. Computing loss and accuracy.",
        number_hidden_layers,
        fault_type,
        noise_variance,
    )

    # Saving training/validation data to file
    with open(
        str(
            Path.home().joinpath(
                "worsecrossbars",
                "outputs",
                _output_folder,
                "training_validation",
                f"training_validation_{fault_type}_{number_hidden_layers}HL"
                + f"_{noise_variance}NV.json",
            )
        ),
        "w",
        encoding="utf-8",
    ) as file:
        output_object = {
            "training_validation_data": [data.tolist() for data in training_validation_data],
            "fault_type": fault_type,
            "number_hidden_layers": number_hidden_layers,
            "noise_variance": noise_variance,
        }
        json.dump(output_object, file)

    logging.info(
        "[%dHL_%s_%.2fNV] Saved training and validation data.",
        number_hidden_layers,
        fault_type,
        noise_variance,
    )

    # Running a variety of simulations to average out stochastic variance
    accuracies = run_simulation(weights_list, percentages, mnist_dataset, simulation_parameters)

    # Saving accuracies array to file
    with open(
        str(
            Path.home().joinpath(
                "worsecrossbars",
                "outputs",
                _output_folder,
                "accuracies",
                f"accuracies_{fault_type}_{number_hidden_layers}HL_{noise_variance}NV.json",
            )
        ),
        "w",
        encoding="utf-8",
    ) as file:
        output_object = {
            "percentages": percentages.tolist(),
            "accuracies": accuracies.tolist(),
            "fault_type": fault_type,
            "number_hidden_layers": number_hidden_layers,
            "noise_variance": noise_variance,
        }
        json.dump(output_object, file)

    logging.info(
        "[%dHL_%s_%.2fNV] Saved accuracy data.",
        number_hidden_layers,
        fault_type,
        noise_variance,
    )

    if _teams:
        _teams.send_message(
            f"Using parameters:\n{simulation_parameters}",
            title="Finished simulation",
            color="1fd513",
        )


def main():
    """Main point of entry for the computing-side of the package."""

    pool = []

    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    mnist_dataset = create_datasets(training_validation_ratio=3)

    for simulation_parameters in json_object["simulations"]:
        validate_parameters(simulation_parameters)
        if command_line_args.teams is None:
            process = Process(
                target=worker, args=[mnist_dataset, simulation_parameters, output_folder]
            )
        else:
            process = Process(
                target=worker, args=[mnist_dataset, simulation_parameters, output_folder, teams]
            )
        process.start()
        pool.append(process)

    for process in pool:
        process.join()

    for accuracy_plot_parameters in json_object["accuracy_plots_parameters"]:
        accuracy_curves(
            accuracy_plot_parameters["plots_data"],
            output_folder,
            xlabel=accuracy_plot_parameters["xlabel"],
            title=accuracy_plot_parameters["title"],
            filename=accuracy_plot_parameters["filename"],
        )

    for tv_plot_parameters in json_object["training_validation_plots_parameters"]:
        training_validation_curves(
            tv_plot_parameters["plots_data"],
            output_folder,
            title=tv_plot_parameters["title"],
            filename=tv_plot_parameters["filename"],
            value_type=tv_plot_parameters["value_type"],
        )

    if command_line_args.dropbox:
        dbx.upload()
        logging.info("Uploaded simulation outcome to Dropbox.")
        if command_line_args.teams:
            teams.send_message(
                f"Simulations {output_folder} uploaded successfully.",
                title="Uploaded to Dropbox",
                color="0060ff",
            )
    sys.exit(0)


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        metavar="CONFIG_FILE",
        nargs="?",
        help="Provide the config file needed for simulations",
        type=str,
        default="",
    )
    parser.add_argument(
        "--setup",
        dest="setup",
        metavar="INITIAL_SETUP",
        help="Run the inital setup",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-w",
        dest="wipe_current",
        metavar="WIPE_CURRENT",
        help="Wipe the current output (or config)",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-d",
        dest="dropbox",
        metavar="DROPBOX",
        help="Enable Dropbox integration",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-t",
        dest="teams",
        metavar="MSTEAMS",
        help="Enable MS Teams integration",
        type=bool,
        default=True,
    )

    command_line_args = parser.parse_args()

    if command_line_args.setup:

        main_setup(command_line_args.wipe_current)
        sys.exit(0)

    else:

        # Create user and output folders.
        user_folders()
        output_folder = create_output_structure(command_line_args.wipe_current)

        logging.basicConfig(
            filename=str(
                Path.home().joinpath("worsecrossbars", "outputs", output_folder, "logs", "run.log")
            ),
            filemode="w",
            format="[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s",
            level=logging.INFO,
            datefmt="%d-%b-%y %H:%M:%S",
        )

        # Get the JSON supplied, parse it, validate it against a known schema.
        json_path = Path.cwd().joinpath(command_line_args.config)
        json_object = read_external_json(str(json_path))
        validate_json(json_object)

        if command_line_args.teams:
            teams = MSTeamsNotifier(read_webhook())

        # Attach Signal Handler
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        if platform.system() == "Darwin" or platform.system() == "Linux":
            signal.signal(signal.SIGHUP, stop_handler)

        # GoTo main
        main()
