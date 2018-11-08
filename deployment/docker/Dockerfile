###################################################
# Dockerfile to build a Python 3.6 environment
# with OGGM installed, based on Ubuntu 18.04.
###################################################

FROM oggm/base:latest
MAINTAINER Timo Rothenpieler

RUN if test -n "$SOURCE_COMMIT"; then pip3 install "git+https://github.com/OGGM/oggm.git@$SOURCE_COMMIT"; else pip3 install "git+https://github.com/OGGM/oggm.git"; fi

ADD test.sh /root/test.sh
