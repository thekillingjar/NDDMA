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

CLANG_FORMAT=$(which clang-format) || (echo "Please install 'clang-format' tool first"; exit 1)

version=$("${CLANG_FORMAT}" --version | sed -n "s/.*\ \([0-9]*\)\.[0-9]*\.[0-9]*.*/\1/p")
if [[ "${version}" -lt "8" ]]; then
  echo "clang-format's version must be at least 8.0.0"
  exit 1
fi


CURRENT_PATH=$(pwd)
PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}

echo "CURRENT_PATH=${CURRENT_PATH}"
echo "PROJECT_HOME=${PROJECT_HOME}"

# print usage message
function usage()
{
  echo "Format the specified source files to conform the code style."
  echo "Usage:"
  echo "bash $0 [-a] [-c] [-l] [-h]"
  echo "e.g. $0 -c"
  echo ""
  echo "Options:"
  echo "    -a format of all files"
  echo "    -c format of the files changed compared to last commit, default case"
  echo "    -l format of the files changed in last commit"
  echo "    -h Print usage"
}

# check and set options
function checkopts()
{
  # init variable
  mode="changed"    # default format changed files

  # Process the options
  while getopts 'aclh' opt
  do
    case "${opt}" in
      a)
        mode="all"
        ;;
      c)
        mode="changed"
        ;;
      l)
        mode="lastcommit"
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

# init variable
# check options
checkopts "$@"

# switch to project root path, which contains clang-format config file '.clang-format'

pushd "${CURRENT_PATH}"
    cd "${PROJECT_HOME}" || exit 1
    FMT_FILE_LIST='__format_files_list__'

    if [[ "X${mode}" == "Xall" ]]; then
      find src -type f -name "*" | grep "\.h$\|\.cc$" > "${FMT_FILE_LIST}" || true
      find inc -type f -name "*" | grep "\.h$\|\.cc$" >> "${FMT_FILE_LIST}" || true
    elif [[ "X${mode}" == "Xchanged" ]]; then
      # --diff-filter=ACMRTUXB will ignore deleted files in commit
      git diff --diff-filter=ACMRTUXB --name-only | grep "^inc\|^base|^compile|^executor|^runner" | grep "\.h$\|\.cc$" >> "${FMT_FILE_LIST}" || true
    else  # "X${mode}" == "Xlastcommit"
      git diff --diff-filter=ACMRTUXB --name-only HEAD~ HEAD | grep "^inc\|^base|^compile|^executor|^runner" | grep "\.h$\|\.cc$" > "${FMT_FILE_LIST}" || true
    fi

    while read line; do
      if [ -f "${line}" ]; then
        ${CLANG_FORMAT} -i "${line}"
      fi
    done < "${FMT_FILE_LIST}"

    rm "${FMT_FILE_LIST}"
popd 

echo "Specified cpp source files have been format successfully."
