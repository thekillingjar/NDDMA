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

SERVER_CONFIG_FILE=${PROJECT_HOME}/scripts/config/server_config.sh

[ -e "$SERVER_CONFIG_FILE" ] || {
    echo "You have not config dependencies account info first !!!!!"
    ${PROJECT_HOME}/scripts/config/ge_config.sh -h
    exit 1;
}

source scripts/config/server_config.sh

CPU_ARCH=ubuntu18.04.x86_64
DRIVER_VERSION=20.2.0
CHIP_NAME=A800-9010
PRODUCT_VERSION=driver_C76_TR5

DRIVER_NAME=npu-driver
DRIVER_RUN_NAME=${CHIP_NAME}-${DRIVER_NAME}_${DRIVER_VERSION}_ubuntu18.04-x86_64.run

DEV_TOOLS_VERSION=1.80.t22.0.b220
export COMPILER_RUN_NAME=CANN-compiler-${DEV_TOOLS_VERSION}-ubuntu18.04.x86_64.run
export RUNTIME_RUN_NAME=CANN-runtime-${DEV_TOOLS_VERSION}-ubuntu18.04.x86_64.run
export OPP_RUN_NAME=CANN-opp-${DEV_TOOLS_VERSION}-ubuntu18.04.x86_64.run

DEV_TOOLS_PACKAGE_NAME=ai_cann_x86
DEV_TOOLS_PACKAGE=${DEV_TOOLS_PACKAGE_NAME}.tar.gz

export DRIVER_URL=${SERVER_PATH}/${PRODUCT_VERSION}/${DRIVER_RUN_NAME}
export DEV_TOOLS_URL=${SERVER_PATH}/20211211/${DEV_TOOLS_PACKAGE}

set +e