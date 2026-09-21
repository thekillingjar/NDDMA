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

# print usage message
ATT_PROJECT_PATH=${HOME}
echo "Set ATT_PROJECT_PATH to $ATT_PROJECT_PATH"
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | help] [--build_type=<gen|clean>] [--test=<flash_attention>] [--target=<code|data>]"
  echo "              [--ascend_install_path=<PATH>] [--ascend_3rd_lib_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, help       Print usage"
  echo "    --example      test demo project(flash_attention), for examples: --example=flash_attention"
  echo "    --build_type   build type mode, support <gen|clean>, default gen mode,
                           gen mode: generator tiling func, tiling st project or generator tiling data result, for example: --build_type=gen
                           clean mode: clean st project build files, for example: --build_type=clean"
  echo "    --target       target type, support <code|data>, must set one target type in gen mode"
  echo "                   code: build user ascend ir and generator tiling_func.cc/tiling_data.h and other "
  echo "                   tiling st project files, for example: --target=code"
  echo "                   data:build st project and generator tiling data result, for example: --target=data"
  echo "    --ascend_install_path=<PATH>"
  echo "      Set ascend package install path, default /home/jenkins/Ascend/ascend-toolkit/latest"
  echo "    --ascend_3rd_lib_path=<PATH>"
  echo "      Set ascend third_party package install path, default ./output/third_party"
  echo ""
}

init_var() {
  # if used in yellow zone, should modify this path
  if [ -z "$ASCEND_3RD_LIB_PATH" ]; then
      ASCEND_3RD_LIB_PATH="$ATT_PROJECT_PATH/../output/third_party"
      echo "ASCEND_3RD_LIB_PATH is not set, set to $ASCEND_3RD_LIB_PATH"
  fi
  # sample的目录
  SAMPLE_PATH=$(dirname "$(realpath $0)")
  # demo(flash_attention)的目录
  DEMO_PATH=${SAMPLE_PATH}"/examples/"
  # 给用户构图/自定义算子的目录
  BUILD_RELATIVE_PATH="build/"
  CMAKE_BUILD_TYPE="" # gen | clean
  GEN_TARGET="" # code | data
  EXAMPLE="" # flash_attention or none
  SCENCE="tool"
  THREAD_NUM=8
}

cmake_command() {
  CMAKE_ARGS="-D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D GEN_TARGET=$GEN_TARGET \
              -D CMAKE_CXX_FLAGS="-I${ATT_PROJECT_PATH}/../tests/autofuse/ut/att/testcase/" \
              -D ATT_PROJECT_PATH=${ATT_PROJECT_PATH}"
  echo "CMAKE_ARGS is: $CMAKE_ARGS"
  mkdir ${BUILD_RELATIVE_PATH}
  echo "mkdir ${BUILD_RELATIVE_PATH}"
  cd ${BUILD_RELATIVE_PATH}
  echo "cd ${BUILD_RELATIVE_PATH}"
  cmake ${CMAKE_ARGS} ../
  echo "cmake ${CMAKE_ARGS} ../"
  if [ 0 -ne $? ]; then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    exit 1
  fi
}

set_env() {
  # 优先使用ATT工程生成的so
  so_directories=$(find "$ATT_PROJECT_PATH/../build/" -type f -name '*.so' -exec dirname {} \; | sort | uniq)
  all_directories="$so_directories
  for dir in $all_directories; do
    export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
    echo "Scans dir=$dir"
  done
  ARCH_NAME=$(uname -m)
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_INSTALL_PATH}/$ARCH_NAME-linux/lib64/
}

deploy_example() {
  if [ -n "$DEMO_PATH$EXAMPLE" ]; then
    # copy path
    echo "rm -rf ${SAMPLE_PATH}/user_src"
    rm -rf ${SAMPLE_PATH}/user_src
    echo "cp -rf ${DEMO_PATH}${EXAMPLE}"/user_src" ${SAMPLE_PATH}"
    cp -rf ${DEMO_PATH}${EXAMPLE}"/user_src" ${SAMPLE_PATH}
  fi
}

check_gen_code_success() {
  if [ $? -eq 0 ]; then
    # 检查是否存在匹配 *tiling*.cpp 的文件
    if [ -z "$(ls *tiling*.cpp 2> /dev/null)" ]; then
      echo "No matching .cpp file found."
      exit 1
    fi
    # 检查是否存在匹配 *tiling*.h 的文件
    if [ -z "$(ls *tiling*.h 2> /dev/null)" ]; then
      echo "No matching .h file found."
      exit 1
    fi
    echo "Gen tiling code successfully."
  else
    echo "Gen tiling code failed, please turn log on and check log."
    exit 1
  fi
}

execute_data_gen() {
  chmod 777 *
  ST_PROJECT_RELATIVE_PATH="gen_data_st"
  cp data_gen ${SAMPLE_PATH}"/"${ST_PROJECT_RELATIVE_PATH}
  echo "cd ${SAMPLE_PATH}/${ST_PROJECT_RELATIVE_PATH}"
  cd ${SAMPLE_PATH}"/"${ST_PROJECT_RELATIVE_PATH}
  export ASCEND_SLOG_PRINT_TO_STDOUT=1
  echo "ASCEND_SLOG_PRINT_TO_STDOUT=1"
  export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL}
  echo "ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL}"
  ./data_gen input_shapes.json
  if [ $? != 0 ];then
    echo "Execute data_gen failed, please check output log."
    exit 1
  fi
  echo "Execute data_gen successfully."
}

gen_target() {
  if [ "$CMAKE_BUILD_TYPE" != "gen" ]; then
    echo "please input -g or --gen if you want to gen code | data"
    exit 1
  fi
  if [ "$EXAMPLE" != "" ]; then
    deploy_example
  fi
  set_env
  cmake_command
  if [ "$GEN_TARGET" == "code" ] || [ "$GEN_TARGET" == "code_with_ctx" ]; then
    echo "Generating code..."
    # 生成ST代码
    # TTODO
    export ASAN_OPTIONS=verify_asan_link_order=0
    make code_gen -j${THREAD_NUM}
    echo "make code_gen -j${THREAD_NUM}"
    chmod 777 *
    echo "chmod 777 *"
    if [ "$GEN_TARGET" == "code" ]; then
      echo "./code_gen tiling_data ./ $EXAMPLE $SCENCE"
      ./code_gen tiling_data ./ $EXAMPLE $SCENCE
    else
      echo "./code_gen tiling_context ./ $EXAMPLE $SCENCE"
      ./code_gen tiling_context ./ $EXAMPLE $SCENCE
    fi
    if [ $? != 0 ];then
      echo "Execute code_gen failed, please check output log."
      exit 1
    fi
    echo "Execute code_gen successfully."
    # 校验文件是否成功生成
    check_gen_code_success
    # gen struct
    make gen_struct
    ./gen_struct
    if [ $? != 0 ];then
      echo "Execute gen struct failed, please check output log."
      exit 1
    fi
    echo "Execute gen struct successfully."
    cp *.cpp *.h input_shapes.json ${SAMPLE_PATH}/gen_data_st/
    echo "cp *.cpp *.h input_shapes.json ${SAMPLE_PATH}/gen_data_st/"
  elif [ "$GEN_TARGET" == "data" ]; then
    echo "Generating data..."
    # 生成ST数据
    make data_gen -j${THREAD_NUM}
    echo "make data_gen -j${THREAD_NUM}"
    chmod 777 *
    echo "chmod 777 *"
    execute_data_gen
  else
    echo "Invalid target $GEN_TARGET"
    exit 1
  fi
}

clean_project() {
  echo "Clean the project..."
  echo "rm -rf ${SAMPLE_PATH}/${BUILD_RELATIVE_PATH}"
  rm -rf ${SAMPLE_PATH}/${BUILD_RELATIVE_PATH}
}

# check value of example name
# usage: check_example example_name
check_example() {
  arg_value="$1"
  if [ "$arg_value" != "flash_attention" ] && [ "$arg_value" != "gelu" ] && [ "$arg_value" != "gelu_v2" ]; then
    echo "Invalid value $arg_value for option --$2"
    usage
    exit 1
  fi
}

check_sence() {
  arg_value="$1"
  if [ "$arg_value" != "tool" ] && [ "$arg_value" != "autofuse" ]; then
    echo "Invalid value $arg_value for option --$2"
    usage
    exit 1
  fi
}

# usage: check_build_type build_type
check_build_type() {
  arg_value="$1"
  if [ "$arg_value" != "gen" ] && [ "$arg_value" != "clean" ]; then
    echo "Invalid value $arg_value for option --$2"
    usage
    exit 1
  fi
}

# usage: check_target target
check_target() {
  arg_value="$1"
  if [ "$arg_value" != "code" ] && [ "$arg_value" != "data" ] && [ "$arg_value" != "code_with_ctx" ]; then
    echo "Invalid value $arg_value for option --$2"
    usage
    exit 1
  fi
}

# parse and set options
checkopts() {
  if [ -n "$ASCEND_INSTALL_PATH" ]; then
    ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
  else
    ASCEND_INSTALL_PATH="/home/jenkins/Ascend/ascend-toolkit/latest"
  fi
  # Process the options
  parsed_args=$(getopt -a -o j:hv -l help,ascend_install_path:,ascend_3rd_lib_path:,build_type:,target:,example:,scence:,log: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      --ascend_install_path)
        echo "ASCEND_INSTALL_PATH="$(realpath $2)""
        ASCEND_INSTALL_PATH="$(realpath $2)"
        shift 2
        ;;
      --ascend_3rd_lib_path)
        ASCEND_3RD_LIB_PATH="$(realpath $2)"
        echo "ASCEND_3RD_LIB_PATH="$(realpath $2)""
        shift 2
        ;;
      --log)
        ASCEND_GLOBAL_LOG_LEVEL="$(realpath $2)"
        shift 2
        ;;
      --example)
        echo "EXAMPLE="$2""
        check_example "$2" build_type
        EXAMPLE="$2"
        shift 2
        ;;
      --scence)
        echo "SCENCE="$2""
        check_sence "$2" build_type
        SCENCE="$2"
        shift 2
        ;;
      --build_type)
        check_build_type "$2" build_type
        echo "CMAKE_BUILD_TYPE="$2""
        CMAKE_BUILD_TYPE="$2"
        shift 2
        ;;
      --target)
        check_target "$2" build_type
        echo "GEN_TARGET="$2""
        GEN_TARGET="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

main() {
  init_var
  echo "cd $SAMPLE_PATH"
  cd $SAMPLE_PATH
  checkopts "$@"
  echo "---------------- Att sample build start ----------------"
  case "$CMAKE_BUILD_TYPE" in
    gen)
      clean_project
      gen_target
      ;;
    clean)
      clean_project
      ;;
    *)
      echo "Invalid build mode: $CMAKE_BUILD_TYPE"
      usage
      exit 1
      ;;
  esac
  echo "---------------- Att sample build finished ----------------"
}

main "$@"
