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

set -e

function help(){
    cat <<-EOF
Usage: ge docs [OPTIONS]

Options:
    -b, --brief  Build brief docs
    -a, --all    Build all docs
    -h, --help
EOF

}

PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)
BRIEF_DOXYFILE_PATH=${PROJECT_HOME}/scripts/docs/Doxyfile_brief
ALL_DOXYFILE_PATH=${PROJECT_HOME}/scripts/docs/Doxyfile_all


function build_brief_docs(){
    rm -rf "${PROJECT_HOME}/docs/doxygen"
    doxygen ${BRIEF_DOXYFILE_PATH}
}

function build_all_docs(){
    rm -rf "${PROJECT_HOME}/docs/doxygen"
    doxygen ${ALL_DOXYFILE_PATH}
}

function parse_args(){
    parsed_args=$(getopt -a -o bah --long brief,all,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        build_all_docs
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -b | --brief)
                build_brief_docs
                ;; 
            -a | --all)
                build_all_docs
                ;; 
            -h | --help)
                help;  exit 1;
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1;
                ;;
        esac
        shift
    done
}

function main(){
    parse_args "$@"
}

main "$@"

set +e