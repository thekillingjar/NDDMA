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


# 检查是否提供了输出路径
if [ -z "$1" ]; then
  echo "Usage: ./install.sh output_path"
  exit 1
fi

# 获取当前脚本的目录
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# 获取上级目录
PARENT_DIR=$(dirname "$SCRIPT_DIR")/../../..

# 修改同级目录下build.sh中的PROJECT_PATH环境变量
BUILD_SCRIPT_PATH="$SCRIPT_DIR/build.sh"

if [ -f "$BUILD_SCRIPT_PATH" ]; then
  # 目标输出路径
  OUTPUT_PATH=$1
  echo "Using OUTPUT_PATH=$OUTPUT_PATH"
  OUTPUT_SAMPLE_DIR=${OUTPUT_PATH}/sample/
  echo "Using OUTPUT_SAMPLE_DIR=$OUTPUT_SAMPLE_DIR"
  # 检查输出路径是否存在，如果不存在，则创建它
  if [ ! -d "$OUTPUT_SAMPLE_DIR" ]; then
    mkdir -p "$OUTPUT_SAMPLE_DIR"
  fi
  # 复制当前目录下的所有文件到输出路径，除了install.sh
  cp -r -- "$SCRIPT_DIR"/* "$OUTPUT_SAMPLE_DIR"
  # 修改同级目录下build.sh中的PROJECT_PATH环境变量
  BUILD_SCRIPT_PATH="$OUTPUT_SAMPLE_DIR/build.sh"
  sed -i "s|^ATT_PROJECT_PATH=.*|export ATT_PROJECT_PATH=$PARENT_DIR/../att|" "$BUILD_SCRIPT_PATH"
  echo "Updated build.sh file: sed -i \"s|^ATT_PROJECT_PATH=.*|export ATT_PROJECT_PATH=$PARENT_DIR/../att|\" \"$BUILD_SCRIPT_PATH\""
  # 排除install.sh
  rm -f "$OUTPUT_SAMPLE_DIR/install.sh"
  echo "Files have been copied to $OUTPUT_SAMPLE_DIR"
else
  echo "build.sh not found in the same directory as install.sh"
fi
