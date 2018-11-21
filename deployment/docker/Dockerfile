###################################################
# Dockerfile to build a Python 3.6 environment
# with OGGM installed, based on Ubuntu 18.04.
###################################################

FROM oggm/base:latest
MAINTAINER Timo Rothenpieler

ARG SOURCE_COMMIT=master
RUN pip3 install "git+https://github.com/OGGM/oggm.git@$SOURCE_COMMIT"

ADD test.sh /root/test.sh
