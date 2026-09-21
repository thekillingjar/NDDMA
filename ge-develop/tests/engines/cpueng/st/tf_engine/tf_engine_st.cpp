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
#include "tf_engine/engine/tf_engine.h"
#undef protected
#undef private

#include "util/tf_util.h"
#include "config/config_file.h"
#include "ops_kernel_info_store/ops_kernel_info_store_stub.h"
#include "ge/ge_api_types.h"
#include "base/err_msg.h"
#include "stub.h"

using namespace aicpu;
using namespace ge;

TEST(TfEngine, GetSoPath_SUCCESS)
{
    char *buffer;
    buffer = getcwd(NULL, 0);
    printf("pwd====%s\n", buffer);
    string path(buffer);
    size_t last = path.find_last_of('/');
    path = path.substr(0, last);
    path += "/stub/";
    free(buffer);
    ASSERT_EQ(GetSoPath(reinterpret_cast<void*>(&TestForGetSoPath::Instance())), path.c_str());
}

TEST(TfEngine, CheckUint32AddOverflow_SUCCESS)
{
    uint32_t a = 1;
    uint32_t b = 1;
    ASSERT_EQ(CheckUint32AddOverflow(a, b), false);
}

TEST(TfEngine, CheckUint32AddOverflow_FAIL)
{
    uint32_t a = UINT32_MAX;
    uint32_t b = 100;
    ASSERT_EQ(CheckUint32AddOverflow(a, b), true);
}


TEST(TfEngine, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "TFKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFOptimizer");
    string builderConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUKernelBuilder", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFBuilder");
}

TEST(TfEngine, GetOpsKernelInfoStores_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_tf_kernel"], nullptr);
    ASSERT_EQ(((AicpuOpsKernelInfoStore *)(opsKernelInfoStores["aicpu_tf_kernel"].get()))->engine_name_, "DNN_VM_AICPU");
}

TEST(TfEngine, GetGraphOptimizerObjs_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    ASSERT_EQ(((AicpuGraphOptimizer *)(graphOptimizers["aicpu_tf_optimizer"].get()))->engine_name_, "DNN_VM_AICPU");
}

TEST(TfEngine, GetOpsKernelBuilderObjs_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["aicpu_tf_builder"], nullptr);
    ASSERT_EQ(((AicpuOpsKernelBuilder *)(opsKernelBuilders["aicpu_tf_builder"].get()))->engine_name_, "DNN_VM_AICPU");
}

TEST(TfEngine, Finalize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    ASSERT_EQ(Finalize(), SUCCESS);
}
