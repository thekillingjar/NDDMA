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
BASEPATH=$(cd "$(dirname $0)"; pwd)

BUILD_PATH=$BASEPATH/../../../build/

cd ${BUILD_PATH}
echo "current path: ${BUILD_PATH}"
./tests/engines/cpueng/ut/cpu_engine_utest &&
./tests/engines/cpueng/ut/tf_engine_utest &&
./tests/engines/cpueng/st/cpu_engine_stest &&
./tests/engines/cpueng/st/tf_engine_stest
