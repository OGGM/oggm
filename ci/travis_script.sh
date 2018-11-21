#!/bin/bash
set -x
set -e

pip3 install --upgrade git+https://github.com/fmaussion/salem.git
pip3 install --upgrade coveralls coverage pytest-cov

[[ -n "$DO_COVERALLS" ]] && COV_OPTS="--cov-config .coveragerc --cov=oggm --cov-report term-missing --cov-append" || COV_OPTS=""

cd /root/oggm
pip3 install -e .

[[ -n "$DO_COVERALLS" ]] && coverage erase

pytest --mpl-upload $MPL $COV_OPTS --run-slow --run-test-env $OGGM_TEST_ENV oggm

if [[ -n "$DO_COVERALLS" ]]; then
	coveralls || true
fi
