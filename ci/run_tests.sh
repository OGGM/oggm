#!/bin/bash
set -e
set -x

chown -R "$(id -u):$(id -g)" "$HOME"

export OGGM_TEST_ENV="$1"
export OGGM_TEST_MODE=unset
export MPLBACKEND=agg

MPL_OUTPUT_OGGM_SUBDIR="$2"

if [[ $OGGM_TEST_ENV == *-* ]]; then
    export OGGM_TEST_MODE="${OGGM_TEST_ENV/*-/}"
    export OGGM_TEST_ENV="${OGGM_TEST_ENV/-*/}"
fi

[[ $OGGM_TEST_MODE == single ]] && export OGGM_TEST_MULTIPROC=False
[[ $OGGM_TEST_MODE == multi ]]  && export OGGM_TEST_MULTIPROC=True && export OGGM_MPL=--mpl
[[ $OGGM_TEST_MODE == mpl ]] && export OGGM_MPL=--mpl

if [[ $OGGM_TEST_ENV == minimal ]]; then
    # Special Mode for minimal tests on minimal Python-Image

    export PIP=pip3
    $PIP install --upgrade pytest git+https://github.com/OGGM/pytest-mpl.git
fi

[[ -d .git ]] || export SETUPTOOLS_SCM_PRETEND_VERSION="g$GITHUB_SHA"

$PIP install --upgrade coverage coveralls git+https://github.com/fmaussion/salem.git
$PIP install -e .

export COVERAGE_RCFILE="$PWD/.coveragerc"

coverage erase

coverage run --source=./oggm --parallel-mode --module \
    pytest --verbose --mpl-results-path="/tmp/oggm-mpl-results/${MPL_OUTPUT_OGGM_SUBDIR/:/_}/${OGGM_TEST_ENV/:/_}" $OGGM_MPL --run-slow --run-test-env $OGGM_TEST_ENV oggm

coverage combine
coverage xml
coverage report --skip-covered
