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

PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}

function help(){
    cat <<-EOF
Usage: ge clean [OPTIONS]

Options:
    -b, --build         Clean build dir
    -d, --docs          Clean generate docs
    -i, --install       Clean dependenices 
    -a, --all           Clean all
    -h, --help 
EOF

}

function clean_relative_dir(){
    rm -rf "${PROJECT_HOME}/${1:-output}"
}

function parse_args(){
    parsed_args=$(getopt -a -o bdiah --long build,docs,install,all,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        clean_relative_dir "build"
        clean_relative_dir "output"
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -b | --build)
                clean_relative_dir "build"
                clean_relative_dir "output"
                ;;
            -d | --docs)
                clean_relative_dir "docs/doxygen"
                ;;
            -i | --install)
                clean_relative_dir "deps"
                ;;
            -a | --all)
                clean_relative_dir "deps"
                clean_relative_dir "build"
                clean_relative_dir "output"
                clean_relative_dir "docs/doxygen"
                ;;
            -h | --help)
                help
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1
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