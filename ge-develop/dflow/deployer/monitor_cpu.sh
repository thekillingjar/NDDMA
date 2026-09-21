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

# Perform monitor for cann/air

current_path=$(dirname "$(readlink -f ${BASH_SOURCE[0]})")
DEPLOY_RES_PATH=""
function get_deploy_res_path() {
  if [[ -n "$RESOURCE_CONFIG_PATH" ]]; then
    resource_file=${RESOURCE_CONFIG_PATH}
    if [[ -f ${resource_file} ]]; then
      deploy_res_path=$(grep -Po 'deploy_res_path[" :]+\K[^"]+' ${resource_file} | head -n 1)
    fi
  fi

  if [[ -z ${deploy_res_path} ]]; then
    deploy_res_path=${HOME}/
  fi

  DEPLOY_RES_PATH=${deploy_res_path}
}

function get_client_connection() {
  get_deploy_res_path
  client_file=${DEPLOY_RES_PATH}/runtime/deploy_res/client.json
  if [[ -f "$client_file" ]]; then
    cat "$client_file"
  fi
  return 0
}

DEPLOYER_DAEMON_SUB_PROCESS_LIST=()
function get_deployer_daemon_sub_process() {
  DEPLOYER_DAEMON_SUB_PROCESS_LIST=()
  deployer_daemon_pid=$(pidof deployer_daemon)
  if [[ -z ${deployer_daemon_pid} ]]; then
    return;
  fi

  deployer_daemon_sub_pids=$(pgrep -P ${deployer_daemon_pid})
  DEPLOYER_DAEMON_SUB_PROCESS_LIST=(${deployer_daemon_sub_pids// /})
}

function check_proc_exist() {
  get_deployer_daemon_sub_process
  if [[ ${#DEPLOYER_DAEMON_SUB_PROCESS_LIST[*]} -gt 0 ]]; then
    proc_name=$(ps -p ${DEPLOYER_DAEMON_SUB_PROCESS_LIST[0]} -o comm=)
    echo "Process ${proc_name} exist."
    return 1
  fi

  count=$(ps -ef|grep -w deployer_daemon | grep -v grep | wc -l)
  if [[ $count -gt 0 ]]; then
    echo "Process deployer_daemon exist."
    return 1
  fi
  return 0
}

function stop() {
  echo "Stopping process."
  ps -ef | grep deployer_daemon | grep -v grep | awk '{print $2}' | xargs -r kill
  ps -ef | grep sub_deployer | grep -v grep | awk '{print $2}' | xargs -r kill
  ps -ef | grep udf_executor | grep -v grep | awk '{print $2}' | xargs -r kill
  get_deploy_res_path
  deploy_res_path=${DEPLOY_RES_PATH}/runtime/deploy_res/
  [ -d "$deploy_res_path" ] && rm -rf "$deploy_res_path"/*
  check_proc_exist
  if [[ "$?" -eq 0 ]]; then
    echo "Process stopped."
    return 0
  fi
  result=1;
  cnt=0
  while [[ $result -gt 0 && $cnt -lt 4 ]]
  do
    echo "Waiting for the process to end."
    sleep 10s
    check_proc_exist
    result=$?
    let cnt++
  done
  if [[ $result -eq 0 ]]; then
    echo "Process stopped."
    return 0;
  fi
  echo "Retry stop process."
  get_deployer_daemon_sub_process
  for proc in "${DEPLOYER_DAEMON_SUB_PROCESS_LIST[@]}"; do
    kill -9 ${proc}
  done
  ps -ef | grep -w deployer_daemon | grep -v grep | awk '{print $2}' | xargs kill -9
  sleep 10
  check_proc_exist
  if [[ "$?" -eq 0 ]]; then
    echo "Process stopped."
    return 0
  fi
  echo "Process stop failed."
  return 1
}

function start() {
  check_proc_exist
  if [[ "$?" -gt 0 ]]; then
    echo "Process exist."
    return 1
  fi

  get_deploy_res_path
  deploy_res_path="${DEPLOY_RES_PATH}/runtime/deploy_res/"
  if [ ! -x "${DEPLOY_RES_PATH}" ]; then
    echo "Does not have execute permission on dir ${DEPLOY_RES_PATH}."
    return 1;
  fi

  if [ ! -d "$deploy_res_path" ]; then
    if [ ! -d "${DEPLOY_RES_PATH}/runtime" ]; then
      mkdir "${DEPLOY_RES_PATH}/runtime"
      chmod 700 "${DEPLOY_RES_PATH}/runtime"
    fi
    mkdir "$deploy_res_path"
  fi
  chmod 700 "$deploy_res_path"

  $current_path/deployer_daemon
  return 0
}

function status() {
  proc_deployer_daemon_number=$(ps -ef | grep -w deployer_daemon | grep -v grep | wc -l)
  if [[ $proc_deployer_daemon_number -le 0 ]]
  then
    echo "Process deployer_daemon does not exist."
    return 1
  fi
  proc_deployer_daemon_status=$(ps -efl | grep deployer_daemon | grep -v grep | awk '{print $2}')
  if [[ "$proc_deployer_daemon_status" == "Z" ]] || [[ "$proc_deployer_daemon_status" == "D" ]] || [[ "$proc_deployer_daemon_status" == "T" ]]
  then
    echo "Process deployer_daemon is in abnormal state $proc_deployer_daemon_status."
    return 1
  fi
  get_client_connection

  deployer_daemon_pid=$(pidof deployer_daemon)
  ip_port=$(netstat -nlp | grep ${deployer_daemon_pid} | awk '{print $4}')
  timeout 10s curl ${ip_port} > /dev/null 2>&1
  curl_ret=$?
  RET_TIMEOUT=124
  if [[ "${curl_ret}" == "${RET_TIMEOUT}" ]]; then
    echo "${ip_port} no response."
    return 1
  fi
}

if [[ -n "$2" ]]
then
  export RESOURCE_CONFIG_PATH="$2"
fi

if [[ "$1" == "status" ]]
then
  status
  if [[ "$?" -eq 0 ]]; then
    exit 0
  fi
  exit 1
elif [[ "$1" == "start" ]]
then
  start
  if [[ "$?" -eq 0 ]]; then
    exit 0
  fi
  exit 1
elif [[ "$1" == "stop" ]]
then
  stop
  if [[ "$?" -eq 0 ]]; then
    exit 0
  fi
  exit 1
else
  echo "invalid input command."
fi