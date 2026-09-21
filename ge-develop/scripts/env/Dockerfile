# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

FROM ubuntu:18.04
ENV PROJECT_HOME=/code/Turing/graphEngine
ARG all_proxy=socks://127.0.0.1:8080
ARG http_proxy=http://127.0.0.1:8081
ARG https_proxy=http://127.0.0.1:8081
RUN printenv > /env_build.txt

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo 'deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\n\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse' > /etc/apt/sources.list

RUN apt-get update \
	&& apt-get install -y git g++ wget unzip clang-format-9 build-essential lcov vim graphviz doxygen openssh-server

RUN cpan install -y Graph::Easy

RUN wget https://cmake.org/files/v3.16/cmake-3.16.7-Linux-x86_64.tar.gz \
    && mkdir -p /opt/cmake-3.16.7 \
	&& tar -xvf cmake-3.16.7-Linux-x86_64.tar.gz -C /opt/cmake-3.16.7 --strip-components=1 \
	&& ln -sf  /opt/cmake-3.16.7/bin/*  /usr/bin/ \
  	&& mv /usr/bin/clang-format-9 /usr/bin/clang-format

RUN wget https://github.com/ccup/lcov/archive/refs/tags/add_lcov.tar.gz -O add_lcov.tar.gz \
	&& mkdir -p /opt/addlcov1.0.0 \
	&& tar -xvf add_lcov.tar.gz -C /opt/addlcov1.0.0 \
	&& mv /opt/addlcov1.0.0/lcov-add_lcov/bin/lcov /usr/bin/addlcov 

RUN mkdir /var/run/sshd \
    && echo "root:root" | chpasswd \
    && sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22 7777

RUN echo 'alias ge=/code/Turing/graphEngine/scripts/ge.sh'>>/etc/bash.bashrc

CMD ["/usr/sbin/sshd" "-D" "&"]
