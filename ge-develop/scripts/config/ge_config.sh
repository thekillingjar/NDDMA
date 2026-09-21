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
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)

function help(){
    cat <<-EOF
Usage: ge config [OPTIONS]

update server config for ge, you need input all config info (ip, user, password)

Options:
    -i, --ip           Config ip config
    -u, --user         Config user name
    -p, --password     Config password
    -h, --help

Example: ge config -i=121.36.**.** -u=asc**, -p=Asc***\#@\!\$     (Need add escape character \ before special charactor $、#、!)

EOF

}

function write_config_file(){
    local ip=$1
    local user=$2
    local password=$3
    if [[ -z "$ip" ]] || [[ -z "$user" ]] || [[ -z "$user" ]]; then
        echo "You need input all info （ip, user,password）obout server config !!!"
        help
        exit 1
    fi

    local password=${password//!/\\!}
    local password=${password//#/\\#}
    local password=${password/\$/\\\$}
    local server_config_file=${PROJECT_HOME}/scripts/config/server_config.sh
    [ -n "${server_config_file}" ] && rm -rf "${server_config_file}"

    cat>${server_config_file}<<-EOF
SERVER_PATH=http://${ip}/package/etrans
DEP_USER=${user}
DEP_PASSWORD=${password}

EOF

}



function parse_args(){
    parsed_args=$(getopt -a -o i::u::p::h --long ip::,user::,password::,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        help
        exit 1
    fi
    local ip=""
    local user=""
    local password=""

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -i | --ip)
                ip=$2
                ;;
            -u | --user)
                user=$2
                ;; 
            -p | --password)
                password=$2
                ;;
            -h | --help)
                help; exit;
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1
                ;;
        esac
        shift 2
    done

    write_config_file $ip $user $password
}

function main(){
    parse_args "$@"
}

main "$@"

set +e