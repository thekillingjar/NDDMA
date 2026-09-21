# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

TEST_PROJECT_PATH=$(dirname "$(realpath $0)")
# 设置默认的ASCEND_INSTALL_PATH
default_ascend_path="/home/jenkins/Ascend/ascend-toolkit/latest"

# 如果ASCEND_INSTALL_PATH环境变量未设置，则检查命令行参数
if [ -z "$ASCEND_INSTALL_PATH" ]; then
  if [ -n "$1" ]; then
    ASCEND_INSTALL_PATH="$1"
  else
    ASCEND_INSTALL_PATH="$default_ascend_path"
  fi
fi

if [ -z "$TOP_DIR" ]; then
  if [ -n "$2" ]; then
    TOP_DIR="$2"
  else
    TOP_DIR="$TEST_PROJECT_PATH/../../"
  fi
fi

BUILD_RELATIVE_PATH="build/"
THREAD_NUM=8
ARCH_NAME=$(uname -m)
ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_PATH}/$ARCH_NAME-linux/lib64

CMAKE_ARGS="-D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
            -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
            -D TOP_DIR=${TOP_DIR}"
echo "CMAKE_ARGS is: $CMAKE_ARGS"

if [ -d "$BUILD_RELATIVE_PATH" ]
then
  rm -rf $BUILD_RELATIVE_PATH
  echo "rm -rf ${BUILD_RELATIVE_PATH}"
fi
mkdir ${BUILD_RELATIVE_PATH}
echo "mkdir ${BUILD_RELATIVE_PATH}"
cd ${BUILD_RELATIVE_PATH} || exit
echo "cd ${BUILD_RELATIVE_PATH}"

so_directories=$(find "${TOP_DIR}" -type f -name '*.so' -exec dirname {} \; | sort | uniq)
for dir in $so_directories; do
  export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
  echo "Scans dir=$dir"
done

cmake ${CMAKE_ARGS} ../
echo "cmake ${CMAKE_ARGS} ../"
if [ 0 -ne $? ]; then
  echo "execute command: cmake ${CMAKE_ARGS} .. failed."
  exit 1
fi
echo "Building code..."
# 联合编译
echo "make -j${THREAD_NUM}"
make -j${THREAD_NUM}
echo "chmod -R 777 ./"
chmod 777 *

./tiling_func
if [ 0 -ne $? ]; then
  exit 1
fi
exit 0