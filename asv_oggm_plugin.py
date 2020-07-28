import subprocess
import requests
import tempfile
import os
import logging

from asv.plugins.conda import _find_conda, Conda
from asv.console import log
from asv import util

logging.getLogger("requests").setLevel(logging.WARNING)

OGGM_CONDA_ENV_URL = ("https://raw.githubusercontent.com/OGGM/"
                      "OGGM-dependency-list/master/Linux-64/{0}")
OGGM_CONDA_ENVS = {
    "36": "oggmdev-1.2.0.202002022248_20200202_py36.yml",
    "37": "oggmdev-1.2.0.202002022248_20200202_py37.yml",
}

class OggmVirtualenv(Conda):
    tool_name = "oggm_conda"

    def _setup(self):
        log.info("Creating oggm conda environment for {0}".format(self.name))

        env_file = tempfile.NamedTemporaryFile(mode="w", delete=False,
                                               suffix=".yml")
        try:
            pyver = str(self._python).replace(".", "")[:2]
            oggm_env = OGGM_CONDA_ENVS[pyver]
            req = requests.get(OGGM_CONDA_ENV_URL.format(oggm_env))
            req.raise_for_status()

            for line in req.text.splitlines():
                if line.startswith("prefix:"):
                    continue
                elif line.startswith("name:"):
                    env_file.write("name: {0}\n".format(self.name))
                else:
                    env_file.write(line + "\n")

            env_file.close()

            self._conda_channels = ["conda-forge", "defaults"]
            self._conda_environment_file = env_file.name

            return super()._setup()
        except Exception as exc:
            if os.path.isfile(env_file.name):
                with open(env_file.name, "r") as f:
                    text = f.read()
                log.info("oggm conda env create failed: in {} with:\n{}"
                         .format(self._path, text))
            raise
        finally:
            os.unlink(env_file.name)
