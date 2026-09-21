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

#include "graph/operator_reg.h"

#include "ascir_utils.h"
#include "test_util.h"

namespace ge {
REG_OP(TestOp).INPUT(x, BasicType()).OP_END_FACTORY_REG(TestOp);
};

TEST(AscirOps_Register, RegOp_WillCreateAscirOpFactory) {
  ge::op::TestOp op("test_op");

  ascir::Graph graph("test");
  graph.SetInputs({op});
}

TEST(AscirCast, DataType_Ok) {
  ascir::Graph graph("test_graph");

  auto data = ascir::cg::ContiguousData("test_op1", graph, ge::DT_FLOAT, {});
  auto cast = ascir::cg::Cast("cast", data.y, ge::DT_FLOAT16);
  graph.SetInputs({data});


  auto cast_node = ge::GraphUtilsEx::GetComputeGraph(graph)->FindNode("cast");
  ASSERT_NE(cast_node, nullptr);
  ASSERT_NE(cast_node->GetInDataAnchor(0), nullptr);
  ASSERT_NE(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx(), 0);
  EXPECT_EQ(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "test_op1");

  auto op_desc = cast_node->GetOpDesc();
  // attr set ok
  AttrEq(op_desc, ascir::ops::Cast::ATTR_dst_type, ge::DT_FLOAT16);
  // output data type infer ok
  EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
}

TEST(AscirOps_StartNode, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data = ascir::cg::Data("test_op", graph, ge::DT_FLOAT16,
                              {a.id, b.id, c.id},
                              {A, B, C},
                              {B*C, C, 1});

  auto data_ret = graph.Find("test_op");

  EXPECT_EQ(data_ret.outputs[0].dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ret.outputs[0].axis[0], a.id);
  EXPECT_EQ(data_ret.outputs[0].axis[1], b.id);
  EXPECT_EQ(data_ret.outputs[0].axis[2], c.id);
  EXPECT_EQ(data_ret.outputs[0].repeats[0], A);
  EXPECT_EQ(data_ret.outputs[0].repeats[1], B);
  EXPECT_EQ(data_ret.outputs[0].repeats[2], C);
  EXPECT_EQ(data_ret.outputs[0].strides[0], B*C);
  EXPECT_EQ(data_ret.outputs[0].strides[1], C);
  EXPECT_EQ(data_ret.outputs[0].strides[2], 1);
}

TEST(AscirOps_StartNode, ConstantOk1) {
  ascir::Graph graph("test_graph");
  auto c1 = ascir::cg::Constant("c1", graph, 10, ge::DT_FLOAT);
  c1.y.dtype = ge::DT_INT64;
  graph.SetInputs({c1});
  graph.SetOutputs({c1});

  auto c1_ret = graph.Find("c1");
  EXPECT_EQ(c1_ret.outputs[0].dtype, ge::DT_INT64);
  EXPECT_EQ(c1_ret.outputs[0].axis().size(), 0);
  auto c1_op_desc = ge::OpDescUtils::GetOpDescFromOperator(c1);
  float value;
  ASSERT_TRUE(ge::AttrUtils::GetFloat(c1_op_desc, "value", value));
  EXPECT_FLOAT_EQ(value, 10);
  std::string expect_str = R"(Graph: test_graph
Sizes:
Axis:
Nodes:
  c1: Constant (0)
    .value = 10
    .dtype = Value of 11
    .axis = {}
    .hint:
      .compute_type = data
    .api:
      .type = Buffer
      .unit = None
    .y.dtype = int64_t
    .y.axis = {}
    .y.repeats = {}
    .y.strides = {}
    .y.vectorized_axis = {}
    .y.mem:
      .tensor_id = 0
      .alloc_type = Global
      .hardware = GM
      .position = TPosition::GM
      .buf_ids = {}
    .y.opt:
      .reuse_id = 0
      .ref_tensor = 0
      .merge_scope = 0
)";
  EXPECT_EQ(ascir::utils::DebugStr(graph, true), expect_str);
}

TEST(AscirOps_StartNode, ConstantOk2) {
  ascir::Graph graph("test_graph");
  auto c1 = ascir::cg::Constant("c1", graph, 11.1, ge::DT_FLOAT);
  c1.y.dtype = ge::DT_FLOAT;
  graph.SetInputs({c1});
  graph.SetOutputs({c1});

  auto c1_ret = graph.Find("c1");
  EXPECT_EQ(c1_ret.outputs[0].dtype, ge::DT_FLOAT);
  EXPECT_EQ(c1_ret.outputs[0].axis().size(), 0);
  auto c1_op_desc = ge::OpDescUtils::GetOpDescFromOperator(c1);
  float value;
  ASSERT_TRUE(ge::AttrUtils::GetFloat(c1_op_desc, "value", value));
  EXPECT_FLOAT_EQ(value, 11.1);
  std::string expect_str = R"(Graph: test_graph
Sizes:
Axis:
Nodes:
  c1: Constant (0)
    .value = 11.1
    .dtype = Value of 11
    .axis = {}
    .hint:
      .compute_type = data
    .api:
      .type = Buffer
      .unit = None
    .y.dtype = float32
    .y.axis = {}
    .y.repeats = {}
    .y.strides = {}
    .y.vectorized_axis = {}
    .y.mem:
      .tensor_id = 0
      .alloc_type = Global
      .hardware = GM
      .position = TPosition::GM
      .buf_ids = {}
    .y.opt:
      .reuse_id = 0
      .ref_tensor = 0
      .merge_scope = 0
)";
  EXPECT_EQ(ascir::utils::DebugStr(graph, true), expect_str);
}

TEST(AscirOps_ContiguousStartNode, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data = ascir::cg::ContiguousData("test_op", graph, ge::DT_FLOAT16, {a, b, c});

  auto data_ret = graph.Find("test_op");

  EXPECT_EQ(data_ret.outputs[0].dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ret.outputs[0].axis[0], a.id);
  EXPECT_EQ(data_ret.outputs[0].axis[1], b.id);
  EXPECT_EQ(data_ret.outputs[0].axis[2], c.id);
  EXPECT_EQ(data_ret.outputs[0].repeats[0], A);
  EXPECT_EQ(data_ret.outputs[0].repeats[1], B);
  EXPECT_EQ(data_ret.outputs[0].repeats[2], C);
  EXPECT_EQ(data_ret.outputs[0].strides[0], B*C);
  EXPECT_EQ(data_ret.outputs[0].strides[1], C);
  EXPECT_EQ(data_ret.outputs[0].strides[2], 1);
}

TEST(AscirOps_FlashSoftmaxInferDataType, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data0 = ascir::cg::ContiguousData("data0", graph, ge::DT_INT32, {a, b, c});
  auto data1 = ascir::cg::ContiguousData("data1", graph, ge::DT_FLOAT16, {a, b, c});
  auto data2 = ascir::cg::ContiguousData("data2", graph, ge::DT_FLOAT16, {a, b, c});

  ascir::cg::FlashSoftmax("fs", data0.y, data1.y, data2.y);

  auto fs = graph.Find("fs");
  EXPECT_EQ(fs.outputs[0].dtype, ge::DT_INT32);
  EXPECT_EQ(fs.outputs[1].dtype, ge::DT_INT32);
  EXPECT_EQ(fs.outputs[2].dtype, ge::DT_INT32);
}
TEST(AscirOps, ExecOrderIncreaseOk) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data0 = ascir::cg::ContiguousData("data0", graph, ge::DT_INT32, {a, b, c});
  auto data1 = ascir::cg::ContiguousData("data1", graph, ge::DT_FLOAT16, {a, b, c});
  auto data2 = ascir::cg::ContiguousData("data2", graph, ge::DT_FLOAT16, {a, b, c});

  ascir::cg::FlashSoftmax("fs", data0.y, data1.y, data2.y);

  auto data0_node = graph.Find("data0");
  auto data1_node = graph.Find("data1");
  auto data2_node = graph.Find("data2");
  auto fs_node = graph.Find("fs");
}

TEST(AscirOps_LoadInferView, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data0 = ascir::cg::ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, c});

  auto l = ascir::cg::Load("fs", data0.y);

  auto load = graph.Find("fs");
  ASSERT_EQ(load.outputs.GetAll().size(), 1);
  EXPECT_EQ(load.outputs[0].axis(), std::vector<int64_t>({a.id, b.id, c.id}));
  EXPECT_EQ(load.outputs[0].repeats(), std::vector<ascir::SizeExpr>({A, B, C}));
  EXPECT_EQ(load.outputs[0].strides(), std::vector<ascir::SizeExpr>({B*C, C, 1}));
  EXPECT_EQ(load.outputs[0].dtype, ge::DT_FLOAT16);
}
