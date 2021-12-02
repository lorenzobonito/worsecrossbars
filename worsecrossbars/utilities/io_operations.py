"""io_operations:
A module that holds multiple utility functions used to
interact with the File System.
"""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def read_external_json(file_path: str) -> dict:
    """Read the contents of a JSON file and dump it in an object.

    Args:
      file_path: The file location on the File System, as a string.

    Returns:
      json_object: Dictionary containing the data stored in the given JSON file
    """

    if not os.path.exists(file_path):
        print("The provided file does not exist! Exiting...")
        sys.exit(1)
    else:
        try:
            with open(file_path, encoding="utf8") as json_file:
                json_object = json.load(json_file)
        except IsADirectoryError:
            print(
                "The given config file argument is invalid. Please, ensure the compute module",
                "is being run on a valid .json file. Call the module with the flag -h for help.",
            )
            sys.exit(1)
        return json_object


def user_folders() -> None:
    """Creates the basic folder structure in the user's HOME folder."""

    home_dir = Path.home()
    home_dir.joinpath("worsecrossbars", "config").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "utils").mkdir(parents=True, exist_ok=True)


def create_output_structure(wipe_current: bool = True) -> str:
    """Create an output folder to store the results of the simulation.

    Args:
      wipe_current: Delete all other folders in "outputs".

    Returns:
      folder: The name of the output folder.
    """

    home_dir = Path.home()
    time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    folder = f"{time_now.__str__()}"

    if wipe_current:
        directory = str(home_dir.joinpath("worsecrossbars", "outputs"))
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

    home_dir.joinpath("worsecrossbars", "outputs", folder, "accuracies").mkdir(
        parents=True, exist_ok=True
    )
    home_dir.joinpath("worsecrossbars", "outputs", folder, "logs").mkdir(
        parents=True, exist_ok=True
    )
    home_dir.joinpath("worsecrossbars", "outputs", folder, "plots", "accuracies").mkdir(
        parents=True, exist_ok=True
    )
    home_dir.joinpath("worsecrossbars", "outputs", folder, "plots", "training_validation").mkdir(
        parents=True, exist_ok=True
    )
    home_dir.joinpath("worsecrossbars", "outputs", folder, "training_validation").mkdir(
        parents=True, exist_ok=True
    )

    return folder


def store_webhook() -> None:
    """Asks the user for a MSTeam Webhook URL and stores it."""

    working_file = Path.home().joinpath("worsecrossbars", "config", "msteams.json")
    webhook_url = input("Please enter the MS Teams Webhook URL: ")
    jsonified_webhook = {"msteams_webhook": webhook_url}
    with open(str(working_file), "w", encoding="utf8") as outfile:
        json.dump(jsonified_webhook, outfile)


def read_webhook() -> str:
    """Reads the Webhook URL from the user's HOME directory.

    Returns:
      webhook_url: String, contains the webhook URL.
    """

    working_file = Path.home().joinpath("worsecrossbars", "config", "msteams.json")

    if not os.path.exists(working_file):
        print("Please run this module with --setup before using Internet options!")
        sys.exit(0)
    else:
        with open(str(working_file), encoding="utf8") as json_file:
            webhook_url = json.load(json_file)["msteams_webhook"]
        return webhook_url


def store_dropbox_keys() -> None:
    """Asks the user for Dropbox keys, and stores them."""

    working_file = Path.home().joinpath("worsecrossbars", "config", "app_keys.json")
    app_key = input("Enter the APP KEY from your Dropbox App: ").strip()
    app_secret = input("Enter the APP SECRET from your Dropbox App: ").strip()
    app_keys = {"APP_KEY": app_key, "APP_SECRET": app_secret}

    with open(str(working_file), "w", encoding="utf8") as outfile:
        json.dump(app_keys, outfile)
