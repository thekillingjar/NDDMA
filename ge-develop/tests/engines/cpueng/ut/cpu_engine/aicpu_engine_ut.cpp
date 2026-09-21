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
#include "aicpu_engine/engine/aicpu_engine.h"
#undef protected
#undef private

#include "util/util.h"
#include "config/config_file.h"
#include "ge/ge_api_types.h"

using namespace aicpu;
using namespace ge;

TEST(AicpuEngine, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "CUSTAICPUKernel,AICPUKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "AICPUOptimizer");
    string builderConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDKernelBuilder", builderConfig), true);
    ASSERT_EQ(builderConfig, "AICPUBuilder");
}

TEST(AicpuEngine, GetOpsKernelInfoStores_SUCCESS)
{
    map<string, OpsKernelInfoStorePtr> opsKernelInfoStores;
    GetOpsKernelInfoStores(opsKernelInfoStores);
    ASSERT_NE(opsKernelInfoStores["aicpu_ascend_kernel"], nullptr);
    ASSERT_EQ(((AicpuOpsKernelInfoStore *)(opsKernelInfoStores["aicpu_ascend_kernel"].get()))->engine_name_, "DNN_VM_AICPU_ASCEND");
}

TEST(AicpuEngine, GetGraphOptimizerObjs_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    AicpuEngine::Instance()->GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);
    ASSERT_EQ(((AicpuGraphOptimizer *)(graphOptimizers["aicpu_ascend_optimizer"].get()))->engine_name_, "DNN_VM_AICPU_ASCEND");
}

TEST(AicpuEngine, GetOpsKernelBuilderObjs_SUCCESS)
{
    map<string, OpsKernelBuilderPtr> opsKernelBuilders;
    AicpuEngine::Instance()->GetOpsKernelBuilderObjs(opsKernelBuilders);
    ASSERT_NE(opsKernelBuilders["aicpu_ascend_builder"], nullptr);
    ASSERT_EQ(((AicpuOpsKernelBuilder *)(opsKernelBuilders["aicpu_ascend_builder"].get()))->engine_name_, "DNN_VM_AICPU_ASCEND");
}

TEST(AicpuEngine, Finalize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Finalize(), SUCCESS);
    ASSERT_EQ(AicpuEngine::Instance()->Initialize(options), SUCCESS);
    ASSERT_EQ(AicpuEngine::Instance()->Finalize(), SUCCESS);
}

TEST(AicpuEngine, AicpuEngine_Finalize_SUCCESS)
{
    AicpuEngine aicpu_egine;
    ASSERT_EQ(aicpu_egine.Finalize(), SUCCESS);
}

TEST(AicpuEngine, TestBf16)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_BF16), 2);

    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_BF16), true);
    ASSERT_EQ(elem, "DT_BF16");

    std::vector<ge::DataType> data_type;
    GetDataType({{"DT_BF16", "DT_BF16"}}, "DT_BF16", data_type);
    ASSERT_EQ(data_type.size(), 1);
    ASSERT_EQ(data_type[0], DataType::DT_BF16);
}

TEST(AicpuEngine, TestUint1)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_UINT1), 1);
    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_UINT1), true);
    ASSERT_EQ(elem, "DT_UINT1");
}

TEST(AicpuEngine, TestHifloat8)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_HIFLOAT8), 1);
    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_HIFLOAT8), true);
    ASSERT_EQ(elem, "DT_HIFLOAT8");
}

TEST(AicpuEngine, Testfloat8_e4m3fn)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_FLOAT8_E4M3FN), 1);

    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_FLOAT8_E4M3FN), true);
    ASSERT_EQ(elem, "DT_FLOAT8_E4M3FN");

    std::vector<ge::DataType> data_type;
    GetDataType({{"DT_FLOAT8_E4M3FN", "DT_FLOAT8_E4M3FN"}}, "DT_FLOAT8_E4M3FN", data_type);
    ASSERT_EQ(data_type.size(), 1);
    ASSERT_EQ(data_type[0], DataType::DT_FLOAT8_E4M3FN);
}

TEST(AicpuEngine, Testfloat8_e5m2)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_FLOAT8_E5M2), 1);

    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_FLOAT8_E5M2), true);
    ASSERT_EQ(elem, "DT_FLOAT8_E5M2");

    std::vector<ge::DataType> data_type;
    GetDataType({{"DT_FLOAT8_E5M2", "DT_FLOAT8_E5M2"}}, "DT_FLOAT8_E5M2", data_type);
    ASSERT_EQ(data_type.size(), 1);
    ASSERT_EQ(data_type[0], DataType::DT_FLOAT8_E5M2);
}

TEST(AicpuEngine, Testfloat8_e8m0)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_FLOAT8_E8M0), 1);

    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_FLOAT8_E8M0), true);
    ASSERT_EQ(elem, "DT_FLOAT8_E8M0");

    std::vector<ge::DataType> data_type;
    GetDataType({{"DT_FLOAT8_E8M0", "DT_FLOAT8_E8M0"}}, "DT_FLOAT8_E8M0", data_type);
    ASSERT_EQ(data_type.size(), 1);
    ASSERT_EQ(data_type[0], DataType::DT_FLOAT8_E8M0);
}

TEST(AicpuEngine, Testfloat4_e2m1)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_FLOAT4_E2M1), 1);

    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_FLOAT4_E2M1), true);
    ASSERT_EQ(elem, "DT_FLOAT4_E2M1");

    std::vector<ge::DataType> data_type;
    GetDataType({{"DT_FLOAT4_E2M1", "DT_FLOAT4_E2M1"}}, "DT_FLOAT4_E2M1", data_type);
    ASSERT_EQ(data_type.size(), 1);
    ASSERT_EQ(data_type[0], DataType::DT_FLOAT4_E2M1);
}

TEST(AicpuEngine, Testfloat4_e1m2)
{
    ASSERT_EQ(GetDataTypeSize(DataType::DT_FLOAT4_E1M2), 1);

    string elem;
    ASSERT_EQ(ConvertDataType2String(elem, DataType::DT_FLOAT4_E1M2), true);
    ASSERT_EQ(elem, "DT_FLOAT4_E1M2");

    std::vector<ge::DataType> data_type;
    GetDataType({{"DT_FLOAT4_E1M2", "DT_FLOAT4_E1M2"}}, "DT_FLOAT4_E1M2", data_type);
    ASSERT_EQ(data_type.size(), 1);
    ASSERT_EQ(data_type[0], DataType::DT_FLOAT4_E1M2);
}