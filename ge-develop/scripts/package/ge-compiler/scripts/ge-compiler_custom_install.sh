#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

sourcedir="$PWD/ge-compiler"
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
unset PYTHONPATH

. "${common_func_path}"

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
pylocal="n"
hetero_arch="n"

while true; do
    case "$1" in
    --install-path=*)
        pkg_install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        pkg_version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --pylocal=*)
        pylocal=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --hetero-arch=*)
        hetero_arch=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    shift
    if [ "$log_type" = "INFO" -o "$log_type" = "WARNING" -o "$log_type" = "ERROR" ]; then
        echo -e "[ge-compiler] [$cur_date] [$log_type]: $*"
    else
        echo "[ge-compiler] [$cur_date] [$log_type]: $*" 1> /dev/null
    fi
    echo "[ge-compiler] [$cur_date] [$log_type]: $*" >> "$logfile"
}

ge_compiler_install_package() {
    local _package="$1"
    local _pythonlocalpath="$2"
    log "INFO" "install python module package in ${_package}"
    if ! command -v pip3 >/dev/null 2>&1; then
        log "ERROR" "install ${_package} failed, pip3 is not installed."
        exit 1
    fi
    if [ -f "$_package" ]; then
        if [ "$pylocal" = "y" ]; then
            pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" -t "${_pythonlocalpath}" 1> /dev/null
        else
            if [ $(id -u) -ne 0 ]; then
                pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" --user 1> /dev/null
            else
                pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" 1> /dev/null
            fi
        fi
        local ret=$?
        if [ $ret -ne 0 ]; then
            log "WARNING" "install ${_package} failed, error code: $ret."
            exit 1
        else
            log "INFO" "install ${_package} successfully!"
        fi
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:install ${_package} faied, can not find the matched package for this platform."
        exit 1
    fi
}

WHL_INSTALL_DIR_PATH="${common_parse_dir}/python/site-packages"
PYTHON_LLM_DATADIST_WHL="${sourcedir}/lib64/llm_datadist_v1-0.0.1-py3-none-any.whl"
PYTHON_DATAFLOW_WHL="${sourcedir}/lib64/dataflow-0.0.1-py3-none-any.whl"
PYTHON_GE_WHL="${sourcedir}/lib64/ge_py-0.0.1-py3-none-any.whl"

custom_install() {
    if [ -z "$common_parse_dir/share/info/ge-compiler" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:ge-compiler directory is empty"
        exit 1
    fi

    if [ "$hetero_arch" != "y" ]; then
        log "INFO" "install ge-compiler extension module begin..."
        ge_compiler_install_package "${PYTHON_LLM_DATADIST_WHL}" "${WHL_INSTALL_DIR_PATH}"
        ge_compiler_install_package "${PYTHON_DATAFLOW_WHL}" "${WHL_INSTALL_DIR_PATH}"
        ge_compiler_install_package "${PYTHON_GE_WHL}" "${WHL_INSTALL_DIR_PATH}"
        log "INFO" "the ge-compiler extension module installed successfully!"

        if [ "${pylocal}" = "y" ]; then
            log "INFO" "please make sure PYTHONPATH include ${WHL_INSTALL_DIR_PATH}."
        else
            log "INFO" "The package te is already installed in python default path. It is recommended to install it using the '--pylocal' parameter, install the package ge-compiler in the ${WHL_INSTALL_DIR_PATH}."
        fi

        if [ "x$stage" = "xinstall" ]; then
            log "INFO" "ge-compiler do migrate user assets."
            if [ $? -ne 0 ]; then
                log "WARNING" "failed to copy custom directories."
                return 1
            fi
        fi
    fi

    return 0
}

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi

exit 0
