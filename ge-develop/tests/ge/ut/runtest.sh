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
BUILD_PATH=$BASEPATH/../../build/
OUTPUT_PATH=$BASEPATH/../../output/

echo $BUILD_PATH

export LD_LIBRARY_PATH=/usr/local/HiAI/driver/lib64:${BUILD_PATH}../third_party/prebuild/x86_64/:/usr/local/HiAI/runtime/lib64:${BUILD_PATH}/graphengine/:${D_LINK_PATH}/x86_64/:${LD_LIBRARY_PATH}
echo ${LD_LIBRARY_PATH}
${OUTPUT_PATH}/ut_libgraph &&
${OUTPUT_PATH}/ut_libge_multiparts_utest &&
${OUTPUT_PATH}/ut_libge_symbol_infer_utest &&
${OUTPUT_PATH}/ut_libge_distinct_load_utest &&
${OUTPUT_PATH}/ut_libge_others_utest &&
${OUTPUT_PATH}/ut_libge_kernel_utest &&
${OUTPUT_PATH}/ut_libge_common_utest &&
${OUTPUT_PATH}/ut_llm_engine_utest &&
${OUTPUT_PATH}/ut_llm_datadist_utest &&
${OUTPUT_PATH}/ut_sc_check
