"""dropbox_upload:
An internal module to upload a user's output folder
to Dropbox.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import dropbox


class DropboxUpload:
    """A Dropbox handler that deals with authentication and upload.

    Args:
      folder: The output folder of the module.
    """

    def __init__(self, folder) -> None:
        if not os.path.exists(
            Path.home().joinpath("worsecrossbars", "config", "user_secrets.json")
        ):
            print("Please run this module with --setup before using Internet options!")
            sys.exit(0)
        else:
            self.output_folder = folder
            with open(
                str(Path.home().joinpath("worsecrossbars", "config", "user_secrets.json")),
                encoding="utf8",
            ) as json_file:
                self.dropbox_secrets = json.load(json_file)
                self.auth_checked = True
            with open(
                str(Path.home().joinpath("worsecrossbars", "config", "app_keys.json")),
                encoding="utf8",
            ) as json_file:
                self.app_keys = json.load(json_file)

    def __call__(self):
        pass

    def upload(self):
        """Uploads the output folder to Dropbox."""

        if self.auth_checked:
            dbx = dropbox.Dropbox(
                oauth2_refresh_token=self.dropbox_secrets["dropbox_refresh"],
                app_key=self.app_keys["APP_KEY"],
                app_secret=self.app_keys["APP_SECRET"],
            )
            shutil.make_archive(
                f"output_{self.output_folder}",
                "zip",
                str(Path.home().joinpath("worsecrossbars", "outputs", self.output_folder)),
            )
            with open(f"output_{self.output_folder}.zip", "rb") as file:
                data = file.read()
            try:
                _ = dbx.files_upload(
                    data,
                    f"/output_{self.output_folder}.zip",
                    mute=True,
                    mode=dropbox.files.WriteMode("overwrite"),
                )
            except dropbox.exceptions.ApiError as err:
                print("*** API error", err)
            os.remove(f"./output_{self.output_folder}.zip")
