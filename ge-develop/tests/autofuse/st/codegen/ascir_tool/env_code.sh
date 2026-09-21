# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 根据自己情况适配
export ASCEND_CUSTOM_PATH=/home/zgj/Ascend/latest/
# for model
export MIND_STUDIO_TOOLKIT_PATH=/home/zgj/Ascend/8.1.RC1.B010

export ASCEND_AICPU_PATH=$ASCEND_CUSTOM_PATH
export ASCEND_HOME_PATH=$ASCEND_CUSTOM_PATH
export ASCEND_OPP_PATH=$ASCEND_CUSTOM_PATH/opp/
source $ASCEND_HOME_PATH/bin/setenv.bash
export ASCEND_INSTALL_PATH=$ASCEND_CUSTOM_PATH

export LD_LIBRARY_PATH=$ASCEND_CUSTOM_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
export PATH=$MIND_STUDIO_TOOLKIT_PATH/tools/msopt/bin/:$PATH

export EXPERIMENTAL_LOWERING_REDUCE=1
export EXPERIMENTAL_LOWERING_CONCAT=1

#使用蓝区仓代码动态修改编译时放开下列环境变量 并根据实际代码仓配置路径
export CODE_PATH=/home/zgj/ascgen-dev/
export ASCEND_3RD_LIB_PATH=/home/zgj/Ascend/opensource_X86
export CODE_BUILD_PATH=$CODE_PATH/build
export PYTHONPATH=$CODE_PATH/build/compiler/py_module/:$PYTHONPATH
export LD_LIBRARY_PATH=$CODE_BUILD_PATH:$CODE_BUILD_PATH/att:$CODE_BUILD_PATH/autofuse:$CODE_BUILD_PATH/compiler/py_module:$CODE_BUILD_PATH/common:$CODE_BUILD_PATH/ascir/meta:$CODE_BUILD_PATH/ascir/generator:$CODE_BUILD_PATH/ascir/meta:$CODE_BUILD_PATH/base/metadef/graph/ascendc_ir/:$CODE_PATH/build/optimize:$CODE_BUILD_PATH/base/metadef/graph/:$CODE_BUILD_PATH/base/metadef/graph/ascendc_ir:$CODE_BUILD_PATH/base/metadef/graph/ascend_ir/generator:$CODE_BUILD_PATH/base/metadef/graph/expression:$CODE_BUILD_PATH/base/metadef/error_manager:$LD_LIBRARY_PATH