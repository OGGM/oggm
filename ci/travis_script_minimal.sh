#!/bin/bash
set -x
set -e

apt-get -y update
apt-get -y install --no-install-recommends git

export PIP=pip3

$PIP install --upgrade pytest git+https://github.com/OGGM/pytest-mpl.git

exec "$(dirname "$0")"/travis_script.sh
