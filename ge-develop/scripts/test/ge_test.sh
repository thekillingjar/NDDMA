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
export ASCEND_INSTALL_PATH=${PROJECT_HOME}/deps/lib

function help(){
    cat <<-EOF
Usage: ge test [OPTIONS]

Options:
    -u, --unit          Run unit Test
    -b, --benchmark     Run benchmark Test
    -c, --component     Run component Test
    -h, --help 
EOF

}

function unit_test(){
    ${PROJECT_HOME}/build_fwk.sh -u
}

function component_test(){
    ${PROJECT_HOME}/build_fwk.sh -s -p all
}

function benchmark_test(){
    ${PROJECT_HOME}/build_fwk.sh -b
}

function parse_args(){
    parsed_args=$(getopt -a -o ubch --long unit,component,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        unit_test
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -u | --unit)
                unit_test
                ;; 
            -c | --component)
                component_test
                ;;

            -b | --benchmark)
                benchmark_test
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
