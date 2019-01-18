#!/bin/bash
set -x
set -e

$PIP install --upgrade git+https://github.com/fmaussion/salem.git
$PIP install --upgrade coveralls coverage

cd /root/oggm
$PIP install -e .

COV_OPTS=""
if [[ -n "$DO_COVERALLS" ]]; then
	coverage erase --rcfile=.coveragerc
	COV_OPTS="coverage run --rcfile=.coveragerc --source=./oggm --parallel-mode --module"
fi

$COV_OPTS pytest --verbose --mpl-upload $MPL --run-slow --run-test-env $OGGM_TEST_ENV oggm

if [[ -n "$DO_COVERALLS" ]]; then
	coverage combine --rcfile=.coveragerc
	coverage report --skip-covered --rcfile=.coveragerc
	coveralls || true
fi
