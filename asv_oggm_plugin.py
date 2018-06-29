import subprocess
import glob
import sys
import os

from asv.plugins.virtualenv import Virtualenv


class OggmVirtualenv(Virtualenv):
    tool_name = "oggm_virtualenv"

    def _install_requirements(self):
        # pip>=10 is broken for some of our dependencies
        self.run_executable("pip", ["install", "-v", "--upgrade", "pip<10", "setuptools"])

        env_key = "OGGM_ASV_WHEEL_DIR_" + self._python.replace(".", "_")
        if env_key not in os.environ:
             env_key = "OGGM_ASV_WHEEL_DIR"

        # check if there are global wheels available somewhere
        if env_key in os.environ:
            wheel_dir = os.path.expanduser(os.environ[env_key])
            wheel_dir = os.path.join(wheel_dir, "*.whl")
            wheels = glob.glob(wheel_dir)
            self.run_executable("pip", ["install", "-v"] + wheels)
        else:
            # some packages(rasterio...) need numpy during setup.py. So install it and some other similar stuff individually.
            self.run_executable("pip", ["install", "-v", "numpy", "six", "cython"])

            # handle odd cases of GDAL/Fiona
            gdal_version = subprocess.check_output(["gdal-config", "--version"]).strip().decode("utf-8")
            gdal_flags = subprocess.check_output(["gdal-config", "--cflags"]).strip().decode("utf-8")
            gdal_flags = gdal_flags.replace("-I", "--include-dirs=")
            self.run_executable("pip", ["install", "-v", "--upgrade", "gdal==" + gdal_version, "--install-option=build_ext", "--install-option=" + gdal_flags])
            self.run_executable("pip", ["install", "-v", "--upgrade", "fiona", "--install-option=build_ext", "--install-option=" + gdal_flags])

        # install latest salem
        self.run_executable("pip", ["install", "-v", "--upgrade", "git+https://github.com/fmaussion/salem.git"])

        super(OggmVirtualenv, self)._install_requirements()

