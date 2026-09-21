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
BASEPATH=$(cd "$(dirname $0)/.."; pwd)
AIRDIR="$(basename $BASEPATH)"
OUTPUT_PATH="${BASEPATH}/output"
export BUILD_PATH="${BASEPATH}/build/"
export ASCEND_INSTALL_PATH="/mnt/d/Ascend"
export LD_LIBRARY_PATH=$ASCEND_INSTALL_PATH/compiler/lib64:$ASCEND_INSTALL_PATH/runtime/lib64:$ASCEND_INSTALL_PATH/runtime/lib64/stub:$LD_LIBRARY_PATH

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build_ffts.sh [-h] [-u] [-s] [-p] [-r] [-f] [-j[n]]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Build ut"
  echo "    -s Build st"
  echo "    -c Build ut with coverage tag"
  echo "    -r Run ut or st after compile ut or st"
  echo "    -f Filter parameter while runing ut or st"
  echo "    -p Build inference, train or all, default is train"
  echo "    -j[n] Set the number of threads used for building AIR, default is 8"
  echo "to be continued ..."
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  ENABLE_FFTS_UT="off"
  ENABLE_FFTS_ST="off"
  ENABLE_FFTS_LLT="off"
  ENABLE_LLT_COV="off"
  ENABLE_RUN_LLT="off"
  LLT_FILTER=""
  PLATFORM="train"
  # Process the options
  while getopts 'ushj:p:f:cr' opt
  do
    case "${opt}" in
      u)
        ENABLE_FFTS_UT="on"
        ENABLE_FFTS_LLT="on"
        ;;
      s)
        ENABLE_FFTS_ST="on"
        ENABLE_FFTS_LLT="on"
        ;;
      h)
        usage
        exit 0
        ;;
      j)
        OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
        THREAD_NUM=$OPTARG
        ;;
      p)
        OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
        PLATFORM=$OPTARG
        ;;
      f)
        LLT_FILTER=$OPTARG
        ;;
      c)
        ENABLE_LLT_COV="on"
        ;;
      r)
        ENABLE_RUN_LLT="on"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

build_ffts()
{
  echo "create build directory and build AIR";

  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"

  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH"
  if [[ "X$ENABLE_FFTS_LLT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_FFTS_LLT=ON"
  fi
  if [[ "X$ENABLE_LLT_COV" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_LLT_COV=ON"
  fi

  CMAKE_ARGS="${CMAKE_ARGS} -DBUILD_OPEN_PROJECT=True -DENABLE_OPEN_SRC=True"
  CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH} -DPLATFORM=${PLATFORM} -DPRODUCT=${PRODUCT}"
  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ..
  if [ $? -ne 0 ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi

  TARGET="ffts"
  if [ "X$ENABLE_FFTS_UT" = "Xon" ];then
    TARGET="ffts_ut"
    make ${VERBOSE} ${TARGET} -j${THREAD_NUM}
  elif [ "X$ENABLE_FFTS_ST" = "Xon" ];then
    TARGET="ffts_st"
    make ${VERBOSE} ${TARGET} -j${THREAD_NUM}
  else
    make ${VERBOSE} ${TARGET} -j${THREAD_NUM} && make install
  fi

  if [ $? -ne 0 ]
  then
    echo "execute command: make ${VERBOSE} ${TARGET} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "AIR build success!"
}

run_llt()
{
  if [ "X$ENABLE_RUN_LLT" != "Xon" ]
  then
    return 0
  fi

  cd "${BASEPATH}"
  if [ "X$ENABLE_FFTS_UT" = "Xon" ];then
    cd "${BASEPATH}/../"
    if [ "X$LLT_FILTER" = "X" ];then
      ./$AIRDIR/build/tests/engines/ffts_engine/ut/ffts_ut
    else
      ./$AIRDIR/build/tests/engines/ffts_engine/ut/ffts_ut --gtest_filter="$LLT_FILTER"
    fi
    if [ "X$ENABLE_LLT_COV" = "Xon" ]
    then
      cd "${BASEPATH}"
      mk_dir "${BASEPATH}/cov_ffts_ut/"
      lcov -c -d build/tests/engines/ffts_engine/ut -o cov_ffts_ut/tmp.info
      lcov -r cov_ffts_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_ffts_ut/coverage.info
      cd "${BASEPATH}/cov_ffts_ut/"
      genhtml coverage.info
      cd "${BASEPATH}"
    fi
  elif [ "X$ENABLE_FFTS_ST" = "Xon" ];then
    cd "${BASEPATH}/../"
    if [ "X$LLT_FILTER" = "X" ];then
      ./$AIRDIR/build/tests/engines/ffts_engine/st/ffts_st
    else
      ./$AIRDIR/build/tests/engines/ffts_engine/st/ffts_st --gtest_filter="$LLT_FILTER"
    fi
    if [ "X$ENABLE_LLT_COV" = "Xon" ]
    then
      cd "${BASEPATH}"
      mk_dir "${BASEPATH}/cov_ffts_st/"
      lcov -c -d build/tests/engines/ffts_engine/st -o cov_ffts_st/tmp.info
      lcov -r cov_ffts_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_ffts_st/coverage.info
      cd "${BASEPATH}/cov_ffts_st/"
      genhtml coverage.info
      cd "${BASEPATH}"
    fi
  else
    echo "There is no ut or st need to be run..."
  fi
  cd "${BASEPATH}"
}

main() {
  # AIR build start
  echo "---------------- AIR build start ----------------"
  checkopts "$@"
  g++ -v

  mk_dir ${OUTPUT_PATH}
  build_ffts || { echo "AIR build failed."; exit 1; }
  echo "---------------- AIR build finished ----------------"

  run_llt || { echo "Run llt failed."; exit 1; }
  echo "---------------- LLT run and generate coverage report finished ----------------"

  chmod -R 750 ${OUTPUT_PATH}
  find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 chmod 500

  echo "---------------- AIR output generated ----------------"
}

main "$@"
