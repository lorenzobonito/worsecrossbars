"""initial_setup:
Module used to configure the system to use worsecrossbars.
"""
import os
from pathlib import Path
from urllib.error import URLError

import wget

from worsecrossbars.utilities import auth_dropbox
from worsecrossbars.utilities import io_operations


def main_setup(overwrite_configs: bool = False):
    """A function to download the correct fonts, and prep the system to use worsecrossbars.

    Args:
      overwrite_configs: Delete the folder and ask the user for new configs,
        regardless of their presence.
    """
    io_operations.user_folders()

    try:
        wget.download(
            "https://github.com/nicogig/ComputerModern/raw/main/cmunrm.ttf",
            out=str(Path.home().joinpath("worsecrossbars", "utils")),
        )
    except URLError as err:
        print(
            "The Computer Modern font could not be downloaded because wget"
            + f" terminated unexpectedly with error {err.reason}.\nPlotting will"
            + " use the standard Matplotlib font. Please consult the Docs for further guidance."
        )
    print("\n")

    if overwrite_configs:
        auth_dropbox.authenticate()
        io_operations.read_webhook()
    else:
        if not os.path.exists(
            Path.home().joinpath("worsecrossbars", "config", "user_secrets.json")
        ):
            auth_dropbox.authenticate()
        if not os.path.exists(Path.home().joinpath("worsecrossbars", "config", "msteams.json")):
            io_operations.store_webhook()
