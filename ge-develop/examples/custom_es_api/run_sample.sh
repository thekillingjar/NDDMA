#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

echo "Start running custom es api sample"

set +u
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo -e "[ERROR] 环境变量ASCEND_HOME_PATH 未配置" >&2
  echo -e "[ERROR] 请先执行: source \${install_path}/cann/set_env.sh  " >&2
  exit 1
fi

BUILD_DIR="build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "[Info] 创建构建目录 ${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
fi

echo "[Info] 开始准备并编译目标: sample"
echo "[Info] 重新生成 CMake 构建文件并开始编译 sample"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target sample -j"$(nproc)"

echo "[Info] 运行 ${BUILD_DIR}/sample"
if [[ -x "${BUILD_DIR}/sample" ]]; then
  ./${BUILD_DIR}/sample
  echo "[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示"
else
  echo "ERROR: 找不到或不可执行 ${BUILD_DIR}/sample" >&2
  exit 1
fi