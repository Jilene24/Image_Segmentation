import os
import subprocess
import sys


class EnvironmentSetup:
    """Class for setting up the environment and dependencies."""

    @staticmethod
    def install_dependencies():
        dependencies = [
            "segmentation-models-pytorch",
            "-U git+https://github.com/albumentations-team/albumentations",
            "--upgrade opencv-contrib-python"
        ]
        for dependency in dependencies:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])

    @staticmethod
    def clone_repository(repo_url: str):
        """Clone the specified GitHub repository."""
        if not os.path.exists('Human-Segmentation-Dataset-master'):
            subprocess.check_call(["git", "clone", repo_url])
        else:
            print("Repository already cloned.")
