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

GE_BASH_HOME=$(dirname "$0")
export PROJECT_HOME=${PROJECT_HOME:-${GE_BASH_HOME}/../}
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)
CUSTOM_LIB_PATH="deps/lib"
export ASCEND_INSTALL_PATH=${PROJECT_HOME}/${CUSTOM_LIB_PATH}

function help(){
        cat <<-EOF
Usage: ge  COMMANDS

Run ge commands

Commands:
    env         Prepare docker env
    config      Config dependencies server
    update      Update dependencies
    format      Format code
    build       Build code
    test        Run test of UT/ST
    cov         Run Coverage  
    docs        Generate documents
    clean       Clean 
EOF

}

function ge_error() {
    echo "Error: $*" >&2
    help
    exit 1
}

function main(){
    if [ $# -eq 0 ]; then
        help; exit 0
    fi

    local cmd=$1
    local shell_cmd=""
    shift

    case "$cmd" in
        -h|--help)
            help; exit 0
            ;;
        cov)
            shell_cmd=$GE_BASH_HOME/coverage/ge_$cmd.sh
            ;;
        *)
            shell_cmd=$GE_BASH_HOME/$cmd/ge_$cmd.sh
            ;;

    esac

    [ -e "$shell_cmd" ] || {
        ge_error "ge $shell_cmd is not found"
    }

    $shell_cmd "$@"
}

main "$@"

set +e
