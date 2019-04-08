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
                      "OGGM-dependency-list/master/linux-64/{0}")
OGGM_CONDA_ENVS = {
    "36": "oggmdev-1.1.0.201903261531_20190326_py36.yml",
    "37": "oggmdev-1.1.0.201903261531_20190326_py37.yml",
}

class OggmVirtualenv(Conda):
    tool_name = "oggm_conda"

    def _has_requirement(self, line):
        for key, _ in self._requirements.items():
            key += "="
            if key in line:
                return True
        return False

    def _setup(self):
        log.info("Creating oggm conda environment for {0}".format(self.name))

        try:
            conda = _find_conda()
        except IOError as e:
            raise util.UserError(str(e))

        env_file = tempfile.NamedTemporaryFile(mode="w", delete=False,
                                               suffix=".yml")
        try:
            pyver = str(self._python).replace(".", "")[:2]
            oggm_env = OGGM_CONDA_ENVS[pyver]
            req = requests.get(OGGM_CONDA_ENV_URL.format(oggm_env))
            req.raise_for_status()
            env_text = req.text

            for line in env_text.splitlines():
                if line.startswith("prefix:") or self._has_requirement(line):
                    continue
                elif line.startswith("name:"):
                    env_file.write("name: {0}\n".format(self.name))
                else:
                    env_file.write(line + "\n")

            conda_args, pip_args = self._get_requirements(conda)
            env_file.writelines(('  - %s\n' % s for s in conda_args))
            if pip_args:
                env_file.write('  - pip:\n')
                env_file.writelines(('    - %s\n' % s for s in pip_args))

            env_file.close()

            util.check_output([conda] + ['env', 'create', '-f', env_file.name,
                                         '-p', self._path, '--force'])
        except Exception as exc:
            if os.path.isfile(env_file.name):
                with open(env_file.name, "r") as f:
                    text = f.read()
                log.info("oggm conda env create failed: in {} with:\n{}"
                         .format(self._path, text))
            raise
        finally:
            os.unlink(env_file.name)
