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
#社区版
CHIP_NAME_C=A800-9010
DRIVER_VERSION_C=21.0.2
DRIVER_RUN_NAME_C=${CHIP_NAME_C}-npu-driver_${DRIVER_VERSION_C}_linux-x86_64.run
DRIVER_SERVER_PATH_C=https://obs-9be7.obs.cn-east-2.myhuaweicloud.com
export DRIVER_URL_C=${DRIVER_SERVER_PATH_C}/turing/resourcecenter/Software/AtlasI/A800-9010%201.0.11/${DRIVER_RUN_NAME_C}

PACKAGE_VERSION_C=5.0.5.alpha001
PACKAGE_NAME_C=Ascend-cann-toolkit_${PACKAGE_VERSION_C}_linux-x86_64.run
PACKAGE_SERVER_PATH_C=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com
export PACKAGE_URL_C=${PACKAGE_SERVER_PATH_C}/CANN/${PACKAGE_VERSION_C}/${PACKAGE_NAME_C}

DEV_TOOLS_VERSION_C=5.0.5.alpha001
CPU_ARCH_C=linux.x86_64
export COMPILER_RUN_NAME_C=CANN-compiler-${DEV_TOOLS_VERSION_C}-${CPU_ARCH_C}.run
export RUNTIME_RUN_NAME_C=CANN-runtime-${DEV_TOOLS_VERSION_C}-${CPU_ARCH_C}.run
export OPP_RUN_NAME_C=CANN-opp-${DEV_TOOLS_VERSION_C}-${CPU_ARCH_C}.run
set +e