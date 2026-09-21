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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "transpose_base_type.h"
#include "transpose/transpose_api_call.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;
using namespace codegen;


class TransposeApiCallTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/* 补充Tranpose CodeGen UT测试用例 */
TEST(TransposeApiCallTest, CodeGenGetTransposeType_2D_10) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id};
    out_tensor.attr.vectorized_axis = {z1.id, z0.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_3D_102) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};
    out_tensor.attr.vectorized_axis = {z1.id, z0.id, z2.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_102);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_3D_021) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};
    out_tensor.attr.vectorized_axis = {z0.id, z2.id, z1.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_021);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_3D_210) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};
    out_tensor.attr.vectorized_axis = {z2.id, z1.id, z0.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_210);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_4D_0213) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto s3 = graph.CreateSizeVar("s3");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);
    auto z3 = graph.CreateAxis("z3", s3);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    out_tensor.attr.vectorized_axis = {z0.id, z2.id, z1.id, z3.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_0213);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_4D_2103) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto s3 = graph.CreateSizeVar("s3");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);
    auto z3 = graph.CreateAxis("z3", s3);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    out_tensor.attr.vectorized_axis = {z2.id, z1.id, z0.id, z3.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_2103);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_4D_0321) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto s3 = graph.CreateSizeVar("s3");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);
    auto z3 = graph.CreateAxis("z3", s3);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    out_tensor.attr.vectorized_axis = {z0.id, z3.id, z2.id, z1.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_ND2ND_0321);
}

TEST(TransposeApiCallTest, CodeGenGetTransposeType_Invalid) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    ge::ascir_op::Data x("x", graph);
    ge::ascir_op::Data y("y", graph);
    auto node0 = graph.FindNode("x");
    auto node1 = graph.FindNode("y");
    ge::AscTensor input_tensor = node0->outputs[0];
    ge::AscTensor out_tensor = node1->outputs[0];

    input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};
    out_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};

    std::string dtype_name;
    Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
    // Setup inputs
    Tensor input(input_tensor, dtype_name);
    Tensor output(out_tensor, dtype_name);

    codegen::TransposeApiCall call("Transpose");
    AutoFuseTransposeType transposeType;
    call.CodeGenGetTransposeType(input, output, transposeType);
    EXPECT_EQ(transposeType, AutoFuseTransposeType::TRANSPOSE_INVALID);
}


/* 补充Tranpose CodeGen UT测试用例 */
TEST(TransposeApiCallTest, TransposeApiCall) {
    ge::AscGraph graph("test_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    /* x是否需要指定位宽，调度API函数需要指定 */
    Data x_op("x", graph);
    Load load_op("load");
    ge::ascir_op::Transpose transpose_op("Transpose");
    graph.AddNode(load_op);
    graph.AddNode(transpose_op);

    //load_op.x.dtype = ge::DT_FLOAT;
    load_op.x = x_op.y;
    load_op.y.dtype = ge::DT_FLOAT;

    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id};
    *load_op.y.repeats = {s0,      s1, s2};
    *load_op.y.strides = {s1 * s2, s2, One};

    /* TODO：perm成员赋值方式 */
    transpose_op.x = load_op.y;
    //transpose_op.perm = std::vector<int32>{0, 2, 1};

    *transpose_op.y.axis = {z1.id, z0.id, z2.id};
    *transpose_op.y.repeats = {s1,      s0, s2};
    *transpose_op.y.strides = {s0 * s2, s2, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
    load->attr.api.type = ge::ApiType::kAPITypeCompute;
    load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;

    /*  TODO:无循环轴怎么配置*/
    load->attr.sched.loop_axis = z0.id;

    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
    load->outputs[0].attr.vectorized_strides = {s1 * s2, s2, One};
    load->outputs[0].attr.dtype = ge::DT_FLOAT;
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto transpose = graph.FindNode("Transpose");
    transpose->attr.api.compute_type = ge::ComputeType::kComputeTranspose;
    transpose->attr.api.type = ge::ApiType::kAPITypeCompute;
    transpose->attr.api.unit = ge::ComputeUnit::kUnitVector;
    transpose->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};

    /* TODO:不配置循环轴 */
    transpose->attr.sched.loop_axis = z0.id;

    /* TODO：新增perm节点 ？？ */
    transpose->outputs[0].attr.vectorized_axis = {z1.id, z0.id, z2.id};
    transpose->outputs[0].attr.vectorized_strides = {s0 * s2, s2, One};
    transpose->outputs[0].attr.dtype = ge::DT_FLOAT;
    transpose->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
    transpose->outputs[0].attr.mem.tensor_id = 1;
    transpose->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    transpose->outputs[0].attr.que.id = 2;
    transpose->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);

    /*  TODO：AddTensor只需要add输出吗？ Perm是否需要体现 */
    tpipe.AddTensor(load->outputs[0]);
    tpipe.AddTensor(transpose->outputs[0]);

    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddAxis(z2);
    tiler.AddSizeVar(ge::SizeVar(s0));
    tiler.AddSizeVar(ge::SizeVar(s1));
    tiler.AddSizeVar(ge::SizeVar(s2));
    tiler.SetTilingCaseId(0);

    codegen::ApiTensor x;
    x.id = load->outputs[0].attr.mem.tensor_id;
    codegen::TransposeApiCall call("Transpose");
    EXPECT_EQ(call.Init(transpose), 0);

    call.inputs.push_back(&x);

    std::string result;
    call.Generate(tpipe, vector<ge::AxisId>{}, result);

    std::cout << "TransposeApiCall" << result;
    printf("tets TransposeApiCall:%s\n", result.c_str());
    EXPECT_EQ(result, std::string{
            "AutoFuseTransposeType transposeType = AutoFuseTransposeType::TRANSPOSE_ND2ND_102;\nauto apiTilingData = t->Transpose_tilingData_0;\ncodegen::ConfusionTranspose<float>(local_1[0], local_0[0], tmp_buf_0, transposeType, apiTilingData);\nAscendC::PipeBarrier<PIPE_ALL>();\n"
    });
}



