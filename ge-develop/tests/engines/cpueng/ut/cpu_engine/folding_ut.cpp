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
#include "aicpu_const_folding/folding.h"
#include "stub.h"

TEST(AicpuConstFoldingUT, InitCpuConstantFoldingNew_001)
{
    int32_t ret = InitCpuConstantFoldingNew([]() -> ge::HostCpuOp * {
        return new (std::nothrow) ge::HostCpuTestOp();
    });
    ASSERT_EQ(ret, 0);
}

TEST(AicpuConstFoldingUT, CpuConstantFoldingComputeNew_001)
{
    ge::Operator op("testop");
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc testTensorDesc;
    ge::GeShape testgeshape;
    testgeshape.SetDimNum(2);
    testTensorDesc.SetShape(testgeshape);
    op_desc->AddInputDesc("testop", testTensorDesc);
    op_desc->AddOutputDesc("testop", testTensorDesc);
    ge::AscendString attr1 = "attr1";
    op.SetAttr("testattr1", attr1);
    float32_t attr2 = 0;
    op.SetAttr("testattr2", attr2);
    int32_t attr3 = 0;
    op.SetAttr("attr3", attr3);
    bool attr4 = true;
    op.SetAttr("attr4", attr4);
    ge::DataType attr5 = ge::DT_FLOAT;
    op.SetAttr("attr5", attr5);
    ge::Tensor attr6;
    op.SetAttr("attr6", attr6);

    std::vector<ge::AscendString> attrList1;
    attrList1.push_back("attrList1");
    op.SetAttr("attrList1", attrList1);
    std::vector<float32_t> attrList2;
    attrList2.push_back(0);
    op.SetAttr("attrList2", attrList2);
    std::vector<int32_t> attrList3;
    attrList3.push_back(0);
    op.SetAttr("attrList3", attrList3);
    std::vector<bool> attrList4;
    attrList4.push_back(true);
    op.SetAttr("attrList4", attrList4);
    std::vector<ge::DataType> attrList5;
    attrList5.push_back(ge::DT_FLOAT);
    op.SetAttr("attrList5", attrList5);
    std::vector<ge::Tensor> attrList6;
    ge::Tensor testtensor;
    attrList6.push_back(testtensor);
    op.SetAttr("attrList6", attrList6);
    std::vector<std::vector<int64_t>> attrList7;
    std::vector<int64_t> testvec;
    testvec.push_back(0);
    attrList7.push_back(testvec);
    op.SetAttr("attrList7", attrList7);
    std::map<std::string, const ge::Tensor> inputs;
    std::map<std::string, ge::Tensor> outputs;
    ge::Tensor testTensor;
    inputs.emplace("testop", testTensor);
    outputs.emplace("testop", testTensor);
    int32_t ret = CpuConstantFoldingComputeNew(op, inputs, outputs);
    ASSERT_EQ(ret, 0);
}