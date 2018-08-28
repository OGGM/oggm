#!/bin/bash
set -x
set -e

pip3 install --upgrade git+https://github.com/fmaussion/salem.git
pip3 install coveralls

cd /root/oggm
pip3 install -e .
pytest oggm $MPL --cov-config .coveragerc --cov=oggm --cov-report term-missing --run-slow --run-test-env $OGGM_TEST_ENV
coverage combine
coveralls || true
