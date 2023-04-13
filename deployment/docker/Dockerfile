###################################################
# Dockerfile to build a Python environment
# with OGGM installed, based on latest Ubuntu.
###################################################

FROM ghcr.io/oggm/base:latest

ARG GITHUB_SHA=master
ARG GITHUB_REPOSITORY=OGGM/oggm
RUN $PIP install "git+https://github.com/${GITHUB_REPOSITORY}@${GITHUB_SHA}"

COPY test.sh /root/test.sh
