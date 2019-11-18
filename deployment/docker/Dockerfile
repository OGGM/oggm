###################################################
# Dockerfile to build a Python 3.6 environment
# with OGGM installed, based on Ubuntu 18.04.
###################################################

FROM oggm/base:latest
MAINTAINER Timo Rothenpieler

ARG GITHUB_SHA=master
ARG GITHUB_REPOSITORY=OGGM/oggm
RUN $PIP install "git+https://github.com/${GITHUB_REPOSITORY}@${GITHUB_SHA}"

ADD test.sh /root/test.sh
