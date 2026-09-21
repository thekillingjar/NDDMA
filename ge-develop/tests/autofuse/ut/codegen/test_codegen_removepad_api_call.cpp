/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "elewise/remove_pad_api_call.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;

class RemovePadApiCallTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(RemovePadApiCallTest, RemovePadApiCall_01) {
    std::string binaryname = "removepad";

    std::vector<ascir::AxisId> current_axis;
    std::vector<std::reference_wrapper<const Tensor>> inputs;
    std::vector<std::reference_wrapper<const Tensor>> outputs;
    ApiAttr api_attr;
    std::string result;

    ge::SizeVar s0(ge::Symbol("s0"));
    ge::SizeVar s1(ge::Symbol("s1"));
    ge::SizeVar s2(ge::Symbol("s2"));

    ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
    ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
    ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

    codegen::Tiler tiler;
    tiler.AddSizeVar(ge::SizeVar(s0));
    tiler.AddSizeVar(ge::SizeVar(s1));
    tiler.AddSizeVar(ge::SizeVar(s2));
    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddAxis(z2);

    codegen::TPipe tpipe("tpipe", tiler);
    ge::AscGraph graph("test");
    ge::ascir_op::Data x("x", graph);
    
    auto node = graph.FindNode("x");
    ge::AscTensor tensor = node->outputs[0];

    tensor.attr.axis = {z0.id, z1.id, z2.id};
    tensor.attr.vectorized_axis = {z1.id, z2.id};
    tensor.attr.repeats = {z0.size, z1.size, z2.size};
    tensor.attr.strides = {z1.size*z2.size, z2.size, One};
    tensor.attr.mem.tensor_id = 1;
    tensor.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
    tensor.attr.mem.position = ge::Position::kPositionVecIn;
    tensor.attr.opt.merge_scope = ge::kIdNone;
    tensor.attr.buf.id = 2;
    vector<ge::Expression> vectorized_strides{One, One};
    vectorized_strides[0] = z2.size;
    tensor.attr.vectorized_strides = vectorized_strides;

    std::string dtype_name;
    Tensor::DtypeName(tensor.attr.dtype, dtype_name);

    Tensor tensor1(tensor, dtype_name);
    tensor1.is_constant = true;
    tensor1.vectorized_axis_pos = {1, 2};
    tensor1.axis_size = {s0.expr, s1.expr, s2.expr};

    Tensor tensor2(tensor, dtype_name);
    tensor2.is_constant = true;
    tensor2.vectorized_axis_pos = {1, 2};
    tensor2.axis_size = {s0.expr, s1.expr, s2.expr};
    inputs.push_back(std::ref(tensor1));
    inputs.push_back(std::ref(tensor2));

    Tensor output_tensor(tensor, dtype_name);
    output_tensor.is_constant = true;
    output_tensor.vectorized_axis_pos = {1, 2};
    output_tensor.axis_size = {s0.expr, s1.expr, s2.expr};
    outputs.push_back(std::ref(output_tensor));
    codegen::RemovePadApiCall call(binaryname);
    Status status = call.Generate(tpipe, current_axis, inputs, outputs, result);;
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(RemovePadApiCallTest, RemovePadApiCall_02) {
    std::string binaryname = "removepad";

    std::vector<ascir::AxisId> current_axis;
    std::vector<std::reference_wrapper<const Tensor>> inputs;
    std::vector<std::reference_wrapper<const Tensor>> outputs;
    ApiAttr api_attr;
    std::string result;

    ge::SizeVar s0(ge::Symbol("s0"));
    ge::SizeVar s1(ge::Symbol("s1"));

    ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
    ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s1.expr};

    codegen::Tiler tiler;
    tiler.AddSizeVar(ge::SizeVar(s0));
    tiler.AddSizeVar(ge::SizeVar(s1));
    tiler.AddAxis(z0);
    tiler.AddAxis(z1);

    codegen::TPipe tpipe("tpipe", tiler);
    ge::AscGraph graph("test");
    ge::ascir_op::Data x("x", graph);
    
    auto node = graph.FindNode("x");
    ge::AscTensor tensor = node->outputs[0];

    tensor.attr.axis = {z0.id, z1.id};
    tensor.attr.vectorized_axis = {z1.id};
    tensor.attr.repeats = {z0.size, z1.size};
    tensor.attr.strides = {z1.size, One};
    tensor.attr.mem.tensor_id = 1;
    tensor.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
    tensor.attr.mem.position = ge::Position::kPositionVecIn;
    tensor.attr.opt.merge_scope = ge::kIdNone;
    tensor.attr.buf.id = 2;
    vector<ge::Expression> vectorized_strides{One, One};
    vectorized_strides[0] = z1.size;
    tensor.attr.vectorized_strides = vectorized_strides;

    std::string dtype_name;
    Tensor::DtypeName(tensor.attr.dtype, dtype_name);

    Tensor tensor1(tensor, dtype_name);
    tensor1.is_constant = true;
    tensor1.vectorized_axis_pos = {1};
    tensor1.axis_size = {s0.expr, s1.expr};

    Tensor tensor2(tensor, dtype_name);
    tensor2.is_constant = true;
    tensor2.vectorized_axis_pos = {1};
    tensor2.axis_size = {s0.expr, s1.expr};
    inputs.push_back(std::ref(tensor1));
    inputs.push_back(std::ref(tensor2));

    Tensor output_tensor(tensor, dtype_name);
    output_tensor.is_constant = true;
    output_tensor.vectorized_axis_pos = {1};
    output_tensor.axis_size = {s0.expr, s1.expr};
    outputs.push_back(std::ref(output_tensor));

    codegen::RemovePadApiCall call(binaryname);
    Status status = call.Generate(tpipe, current_axis, inputs, outputs, result);;
    EXPECT_EQ(status, ge::SUCCESS);
}