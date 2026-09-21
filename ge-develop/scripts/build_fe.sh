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
BUILD_PATH="${BASEPATH}/build/"
ASCEND_INSTALL_PATH="/mnt/d/Ascend"
ASCEND_3RD_LIB_PATH="/mnt/3rd_lib_path"

export AIR_CODE_DIR=${AIRDIR}
echo "AIR_CODE_DIR=${AIR_CODE_DIR}"

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build_fe.sh [-h] [-b] [-u] [-s] [-r] [-f] [-d] [-j[n]]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -b Build benchmark"
  echo "    -u Build ut"
  echo "    -s Build st"
  echo "    -w Build whole process st"
  echo "    -c Build ut/st with coverage tag"
  echo "    -r Run ut or st after compile ut or st"
  echo "    -f Filter parameter while runing ut or st"
  echo "    -p Specify ascend install path"
  echo "    -d Download the baseline libs by specified date, such as -d20220426"
  echo "    -j[n] Set the number of threads used for building AIR, default is 8"
  echo "to be continued ..."
}

create_metadef_softlink()
{
  if [ ! -e "${BASEPATH}/base/metadef" ]
  then
    ln -sf ${BASEPATH}/../metadef ${BASEPATH}/base/metadef
  fi
  if [ ! -e "${BASEPATH}/base/parser" ]
  then
    ln -sf ${BASEPATH}/../parser ${BASEPATH}/base/parser
  fi
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  ENABLE_ENGINES_COMPILE="on"
  ENABLE_UT="off"
  ENABLE_ST="off"
  ENABLE_ST_WHOLE_PROCESS="off"
  ENABLE_FE_LLT="off"
  ENABLE_LLT_COV="off"
  ENABLE_RUN_LLT="off"
  ENABLE_FE_BENCHMARK="off"
  ENABLE_ASAN="false"
  LLT_FILTER=""
  BUILD_METADEF="on"
  CMAKE_BUILD_TYPE="Release"
  # Process the options
  while getopts 'bushj:p:f:d:crw' opt
  do
    case "${opt}" in
      b)
        ENABLE_ENGINES_COMPILE="off"
        ENABLE_FE_BENCHMARK="on"
        ;;
      u)
        ENABLE_ENGINES_COMPILE="off"
        ENABLE_UT="on"
        ENABLE_FE_LLT="on"
        ENABLE_ASAN="true"
        CMAKE_BUILD_TYPE="GCOV"
        ;;
      s)
        ENABLE_ENGINES_COMPILE="off"
        ENABLE_ST="on"
        ENABLE_FE_LLT="on"
        ENABLE_ASAN="true"
        CMAKE_BUILD_TYPE="GCOV"
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
        ASCEND_INSTALL_PATH=$OPTARG
        ;;
      f)
        LLT_FILTER=$OPTARG
        ;;
      d)
        OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
        PKG_DATE=$OPTARG
        #Download baseline pkg
        install_baseline_pkg
        exit 0
        ;;
      c)
        ENABLE_LLT_COV="on"
        ;;
      r)
        ENABLE_RUN_LLT="on"
        ;;
      w)
        ENABLE_ENGINES_COMPILE="off"
        ENABLE_ST_WHOLE_PROCESS="on"
        ENABLE_FE_LLT="on"
        CMAKE_BUILD_TYPE="GCOV"
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

TOPI_INIT=""
PYTHON_SITE_PKG_PATH=$(python3 -m site --user-site)
CANN_KB_INIT="$PYTHON_SITE_PKG_PATH/tbe/common/repository_manager/interface.py"
INIT_MULTI_PROCESS_ENV_PATH="$PYTHON_SITE_PKG_PATH/te_fusion/parallel_compilation.py"
insert_str="\ \ \ \ return 1, None, None, None\ #warning:\ fe_st\ substitute,\ need\ to\ restore"
install_python_stub_file()
{
  pip3 install sympy -i https://pypi.tuna.tsinghua.edu.cn/simple
  # 1. install dependencies
  echo "CANN_KB_INIT = ${CANN_KB_INIT}"
  echo "INIT_MULTI_PROCESS_ENV_PATH = ${INIT_MULTI_PROCESS_ENV_PATH}"
#  cann_kb_exist=$()
#  init_multi_process_env_exist=$(! -f "${INIT_MULTI_PROCESS_ENV_PATH}")
  if [ ! -f "${CANN_KB_INIT}" ] || [ ! -f "${INIT_MULTI_PROCESS_ENV_PATH}" ];
  then
    pip3 install --user "$ASCEND_INSTALL_PATH/compiler/lib64/te-0.4.0-py3-none-any.whl" --force-reinstall
  else
    echo "Both files exist."
  fi
}

restore_python_stub_file()
{
  echo "restore_python_stub_file"
  delete_line_nums=$(grep "${insert_str}" ${INIT_MULTI_PROCESS_ENV_PATH} -n |awk -F ":" '{print $1}')
  arr=(${delete_line_nums//,/})
  delete_size=${#arr[@]}
  echo "delete size ${delete_size}"
  for ((i=$delete_size-1; i>=0; i--))
  do
    sed -i "${arr[i]}d" ${INIT_MULTI_PROCESS_ENV_PATH}
  done
}

build_fe()
{
  echo "create build directory and build AIR";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  cmake -D BUILD_OPEN_PROJECT=True \
        -D ENABLE_OPEN_SRC=True \
        -D ENABLE_ENGINES_COMPILE=${ENABLE_ENGINES_COMPILE} \
        -D BUILD_METADEF=${BUILD_METADEF} \
        -D ENABLE_FE_LLT=${ENABLE_FE_LLT} \
        -D ENABLE_FFTS_LLT=${ENABLE_FFTS_LLT} \
        -D ENABLE_FE_BENCHMARK=${ENABLE_FE_BENCHMARK} \
        -D ENABLE_UT=${ENABLE_UT} \
        -D ENABLE_ST=${ENABLE_ST} \
        -D ENABLE_ST_WHOLE_PROCESS=${ENABLE_ST_WHOLE_PROCESS} \
        -D ENABLE_LLT_COV=${ENABLE_LLT_COV} \
        -D ENABLE_ASAN=${ENABLE_ASAN} \
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
        -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
        -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
        ..

  make ${VERBOSE} select_targets -j${THREAD_NUM} && make install
  if [ $? -ne 0 ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "AIR build success!"
}

run_llt()
{
  if [[ "X$ENABLE_RUN_LLT" != "Xon" && "X$ENABLE_FE_BENCHMARK" != "Xon" ]]
  then
    return 0
  fi

  export FE_ST_PATH=${ASCEND_INSTALL_PATH}
  export ASCEND_OPP_PATH=$ASCEND_INSTALL_PATH/opp
  export PATH=$ASCEND_INSTALL_PATH/compiler/ccec_compiler/bin:$PATH
  export TE_PARALLEL_COMPILER=1
  cd "${BASEPATH}/build/tests/engines/nn_engine/depends/te_fusion/"
  ln -sf libte_fusion_stub.so libte_fusion.so
  cd "${BASEPATH}"
  if [ "X$ENABLE_UT" = "Xon" ];then
    cd "${BASEPATH}/build/tests/engines/nn_engine/"
    mkdir -p ./ut/plugin/opskernel/fe_config/
    ln -sf ../depends/te_fusion/libte_fusion_stub.so ./ut/libte_fusion.so
    ln -sf ../../../depends/graph_tuner/libgraph_tuner_stub.so ./ut/plugin/opskernel/libgraph_tuner.so
    cp "${BASEPATH}/compiler/engines/nn_engine/optimizer/fe_config/fe.ini" "${BASEPATH}/build/tests/engines/nn_engine/ut/plugin/opskernel/fe_config"
    cd "${BASEPATH}/../"
    if [ "X$ENABLE_ASAN" = "Xtrue" ];then
      USE_ASAN=$(gcc -print-file-name=libasan.so)
      export LD_PRELOAD=${USE_ASAN}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
      export ASAN_OPTIONS=detect_leaks=0
    fi
    if [ "X$LLT_FILTER" = "X" ];then
      ./$AIRDIR/build/tests/engines/nn_engine/ut/fe_ut
    else
      ./$AIRDIR/build/tests/engines/nn_engine/ut/fe_ut --gtest_filter="$LLT_FILTER"
    fi
    if [ "X$ENABLE_ASAN" = "Xtrue" ];then
      unset LD_PRELOAD
      unset ASAN_OPTIONS
    fi
    if [ "X$ENABLE_LLT_COV" = "Xon" ]
    then
      cd "${BASEPATH}"
      mkdir "${BASEPATH}/cov_fe_ut/"
      lcov -c -d build/tests/engines/nn_engine/ut -o cov_fe_ut/tmp.info
      lcov -r cov_fe_ut/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_fe_ut/coverage.info
      cd "${BASEPATH}/cov_fe_ut/"
      genhtml coverage.info
      cd "${BASEPATH}"
    fi
  fi
  if [ "X$ENABLE_ST" = "Xon" ];then
    cd "${BASEPATH}/build/tests/engines/nn_engine/"
    mkdir -p ./st/plugin/opskernel/fe_config/
    ln -sf ../depends/te_fusion/libte_fusion_stub.so ./st/libte_fusion.so
    ln -sf ../../../depends/graph_tuner/libgraph_tuner_stub.so ./st/plugin/opskernel/libgraph_tuner.so
    cp "${BASEPATH}/compiler/engines/nn_engine/optimizer/fe_config/fe.ini" "${BASEPATH}/build/tests/engines/nn_engine/st/plugin/opskernel/fe_config"
    cd "${BASEPATH}/../"
    if [ "X$LLT_FILTER" = "X" ];then
      ./$AIRDIR/build/tests/engines/nn_engine/st/fe_st
    else
      ./$AIRDIR/build/tests/engines/nn_engine/st/fe_st --gtest_filter="$LLT_FILTER"
    fi
    if [ "X$ENABLE_LLT_COV" = "Xon" ]
    then
      cd "${BASEPATH}"
      mkdir "${BASEPATH}/cov_fe_st/"
      lcov -c -d build/tests/engines/nn_engine/st -o cov_fe_st/tmp.info
      lcov -r cov_fe_st/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_fe_st/coverage.info
      cd "${BASEPATH}/cov_fe_st/"
      genhtml coverage.info
      cd "${BASEPATH}"
    fi
  fi
  if [ "X$ENABLE_ST_WHOLE_PROCESS" = "Xon" ];then
    install_python_stub_file
    echo "---------------- Begin to run fe whole process st ----------------"
    cd "${BASEPATH}/../"
    ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess310P3.*
    ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess910A.*
    ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess910B.*
    ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeOmConsistencyCheck.*
    ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcess310B.*
    ./$AIRDIR/build/tests/engines/nn_engine/st_whole_process/fe_st_whole_process --gtest_filter=STestFeWholeProcessNano.*
    echo "---------------- Finish the fe whole process st ----------------"
    restore_python_stub_file
    if [ "X$ENABLE_LLT_COV" = "Xon" ]
    then
      cd "${BASEPATH}"
      mkdir "${BASEPATH}/cov_fe_st_whole_process/"
      lcov -c -d build/tests/engines/nn_engine/st -o cov_fe_st_whole_process/tmp.info
      lcov -r cov_fe_st_whole_process/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/test/*' '/usr/local/*' '/usr/include/*'  -o cov_fe_st_whole_process/coverage.info
      cd "${BASEPATH}/cov_fe_st_whole_process/"
      genhtml coverage.info
    fi
  fi
  if [ "X$ENABLE_FE_BENCHMARK" = "Xon" ];then
     cd "${BASEPATH}/../"
     ./$AIRDIR/build/tests/engines/nn_engine/benchmark/fe_benchmark
  fi
  cd "${BASEPATH}"
}

install_baseline_pkg()
{
  if [ "X$PKG_DATE" = "X" ]
  then
    return 0
  fi

  PKG_DIR="${BASEPATH}/Ascend"
  if [ "X$ASCEND_INSTALL_PATH" != "X" ]
  then
    PKG_DIR=$ASCEND_INSTALL_PATH
  fi

  echo "Begin to download baseline package[${PKG_DATE}] to dir[${PKG_DIR}]."
  mk_dir ${PKG_DIR}
  cd "${PKG_DIR}"
  rm -rf compiler_bak
  rm -rf runtime_bak
  rm -rf opp_bak
  rm -rf opensdk_bak
  if [ -e compiler ]
  then
    mv compiler compiler_bak
  fi
  if [ -e runtime ]
  then
    mv runtime runtime_bak
  fi
  if [ -e opp ]
  then
    mv opp opp_bak
  fi
  if [ -e opensdk ]
  then
    mv opensdk opensdk_bak
  fi

  rm -f ai_cann_x86.tar.gz
  rm -rf ai_cann_x86
  wget https://ascend-cann.obs.myhuaweicloud.com/CANN/"$PKG_DATE"_newest/ai_cann_x86.tar.gz
  tar xvf ai_cann_x86.tar.gz
  chmod +x ./ai_cann_x86/*.run
  ./ai_cann_x86/CANN-opensdk-*-linux.x86_64.run --noexec --extract=opensdk
  ./ai_cann_x86/CANN-compiler-*-linux.x86_64.run --noexec --extract=cm
  ./ai_cann_x86/CANN-runtime-*-linux.x86_64.run --noexec --extract=rt
  ./ai_cann_x86/CANN-opp-*-linux.x86_64.run --noexec --extract=op
  mv ./cm/compiler ./
  mv ./rt/runtime ./
  mv ./op/opp ./
  rm -rf cm
  rm -rf rt
  rm -rf op
  rm -rf ai_cann_x86
  rm -f ai_cann_x86.tar.gz
  cd "${BASEPATH}"
}

main() {
  # AIR build start
  echo "---------------- AIR build start ----------------"
  checkopts "$@"

  create_metadef_softlink
  #cd base;  git submodule update --init metadef

  env
  g++ -v
  export LD_LIBRARY_PATH=$ASCEND_INSTALL_PATH/compiler/lib64:$ASCEND_INSTALL_PATH/runtime/lib64:$ASCEND_INSTALL_PATH/runtime/lib64/stub:$LD_LIBRARY_PATH
  mk_dir ${OUTPUT_PATH}
  build_fe || { echo "AIR build failed."; exit 1; }
  echo "---------------- AIR build finished ----------------"

  run_llt || { echo "Run llt failed."; exit 1; }
  echo "---------------- LLT run and generate coverage report finished ----------------"
}

main "$@"
