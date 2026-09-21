/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#define private public
#define protected public
#include "hostcpu_engine/engine/hostcpu_engine.h"
#undef protected
#undef private

#include "util/util.h"
#include "config/config_file.h"

#include "ge/ge_api_types.h"

using namespace aicpu;
using namespace ge;
using namespace std;

TEST(HostCpuEngine, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_HOST_CPUOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "HOSTCPUKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_HOST_CPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "HOSTCPUOptimizer");
    string builderConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_HOST_CPUKernelBuilder", builderConfig), true);
    ASSERT_EQ(builderConfig, "HOSTCPUBuilder");
}

TEST(HostCpuEngine, GetOpsKernelInfoStores_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"], nullptr);
    ASSERT_EQ(((AicpuOpsKernelInfoStore *)(opsKernelInfoStores["DNN_VM_HOST_CPU_OP_STORE"].get()))->engine_name_, "DNN_VM_HOST_CPU");
}

TEST(HostCpuEngine, GetGraphOptimizerObjs_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["DNN_VM_HOST_CPU_OPTIMIZER"], nullptr);
    ASSERT_EQ(((AicpuGraphOptimizer *)(graphOptimizers["DNN_VM_HOST_CPU_OPTIMIZER"].get()))->engine_name_, "DNN_VM_HOST_CPU");
}

TEST(HostCpuEngine, GetOpsKernelBuilderObjs_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["host_cpu_builder"], nullptr);
    ASSERT_EQ(((AicpuOpsKernelBuilder *)(opsKernelBuilders["host_cpu_builder"].get()))->engine_name_, "DNN_VM_HOST_CPU");
}

TEST(HostCpuEngine, Finalize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Finalize(), SUCCESS);
    ASSERT_EQ(HostCpuEngine::Instance()->Initialize(options), SUCCESS);
    ASSERT_EQ(HostCpuEngine::Instance()->Finalize(), SUCCESS);
}

TEST(HostCpuEngine, HostCpuEngine_Finalize_SUCCESS)
{
    HostCpuEngine aicpu_egine;
    ASSERT_EQ(aicpu_egine.Finalize(), SUCCESS);
}
