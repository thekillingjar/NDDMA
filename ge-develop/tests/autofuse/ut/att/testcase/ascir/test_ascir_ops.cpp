/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "graph_utils_ex.h"
#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "graph/operator_reg.h"
#include "test_util.h"
#include "base/base_types.h"

using namespace att;
namespace ge {
REG_OP(TestOp).INPUT(x, BasicType()).__OP_END_IMPL_WITHOUT_REGISTER__(TestOp);

TEST(AscirOps_Register, RegOp_WillCreateAscirOpFactory) {
  ge::op::TestOp op("test_op");

  ge::AscGraph graph("test");
  EXPECT_EQ(graph.GetName(), "test");
}



TEST(AscirOps_StartNode, Ok) {
  ge::AscGraph graph("test_graph");
  auto A = ge::Symbol("A");
  auto B = ge::Symbol("B");
  auto C = ge::Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data = ascir::cg::Data("test_op", graph, ge::DT_FLOAT16,
                              {a.id, b.id, c.id},
                              {A, B, C},
                              {B*C, C, ge::sym::kSymbolOne});

  auto data_ret = graph.FindNode("test_op");

  EXPECT_EQ(data_ret->outputs[0].attr.dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ret->outputs[0].attr.axis[0], a.id);
  EXPECT_EQ(data_ret->outputs[0].attr.axis[1], b.id);
  EXPECT_EQ(data_ret->outputs[0].attr.axis[2], c.id);
  EXPECT_TRUE(data_ret->outputs[0].attr.repeats[0] == A);
  EXPECT_TRUE(data_ret->outputs[0].attr.repeats[1] == B);
  EXPECT_TRUE(data_ret->outputs[0].attr.repeats[2] == C);
  EXPECT_TRUE(data_ret->outputs[0].attr.strides[0] == (B * C));
  EXPECT_TRUE(data_ret->outputs[0].attr.strides[1] == C);
  EXPECT_TRUE(data_ret->outputs[0].attr.strides[2] == ge::Symbol(1));
}

TEST(AscirOps_StartNode, ConstantOk1) {
  ge::AscGraph graph("test_graph");
  auto c1 = ascir::cg::Data("c1", graph);
  c1.dtype = ge::DT_INT64;

  auto c1_ret = graph.FindNode("c1");
  EXPECT_EQ(c1_ret->outputs[0].attr.dtype, ge::DT_INT64);
  EXPECT_EQ(c1_ret->outputs[0].attr.axis.size(), 0);
  auto c1_op_desc = ge::OpDescUtils::GetOpDescFromOperator(c1.GetOwnerOp());
  float value;
  ASSERT_FALSE(ge::AttrUtils::GetFloat(c1_op_desc, "value", value));
//  EXPECT_FLOAT_EQ(value, 10);
  std::string expect_str = R"(Graph: test_graph
Sizes:
Axis:
Nodes:
  c1: Data (0)
    .value = 10
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
      .position = GM
      .buf_ids = {}
      .reuse_id = 0
    .y.opt:
      .ref_tensor = 0
      .merge_scope = 0
)";
//  EXPECT_EQ(ascir::utils::DebugStr(graph, true), expect_str);
}

TEST(AscirOps_StartNode, ConstantOk2) {
  ge::AscGraph graph("test_graph");
  auto c1 = ascir::cg::Data("c1", graph);
  c1.dtype = ge::DT_FLOAT;

  auto c1_ret = graph.FindNode("c1");
  EXPECT_EQ(c1_ret->outputs[0].attr.dtype, ge::DT_FLOAT);
  EXPECT_EQ(c1_ret->outputs[0].attr.axis.size(), 0);
  auto c1_op_desc = ge::OpDescUtils::GetOpDescFromOperator(c1.GetOwnerOp());
  float value;
  ASSERT_FALSE(ge::AttrUtils::GetFloat(c1_op_desc, "value", value));
//  EXPECT_FLOAT_EQ(value, 11.1);
  std::string expect_str = R"(Graph: test_graph
Sizes:
Axis:
Nodes:
  c1: Data (0)
    .value = 11.1
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
      .position = GM
      .buf_ids = {}
      .reuse_id = 0
    .y.opt:
      .ref_tensor = 0
      .merge_scope = 0
)";
//  EXPECT_EQ(ascir::utils::DebugStr(graph, true), expect_str);
}

TEST(AscirOps_ContiguousStartNode, Ok) {
  ge::AscGraph graph("test_graph");
  auto A = ge::Symbol("A");
  auto B = ge::Symbol("B");
  auto C = ge::Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data = ascir::cg::ContiguousData("test_op", graph, ge::DT_FLOAT16, {a, b, c});

  auto data_ret = graph.FindNode("test_op");

  EXPECT_EQ(data_ret->outputs[0].attr.dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ret->outputs[0].attr.axis[0], a.id);
  EXPECT_EQ(data_ret->outputs[0].attr.axis[1], b.id);
  EXPECT_EQ(data_ret->outputs[0].attr.axis[2], c.id);
  EXPECT_TRUE(data_ret->outputs[0].attr.repeats[0] == A);
  EXPECT_TRUE(data_ret->outputs[0].attr.repeats[1] == B);
  EXPECT_TRUE(data_ret->outputs[0].attr.repeats[2] == C);
  EXPECT_TRUE(data_ret->outputs[0].attr.strides[0] == (B * C));
  EXPECT_TRUE(data_ret->outputs[0].attr.strides[1] == C);
  EXPECT_TRUE(data_ret->outputs[0].attr.strides[2] == ge::Symbol(1));
}

TEST(AscirOps_FlashSoftmaxInferDataType, Ok) {
  ge::AscGraph graph("test_graph");
  auto A = ge::Symbol("A");
  auto B = ge::Symbol("B");
  auto C = ge::Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data0 = ascir::cg::ContiguousData("data0", graph, ge::DT_INT32, {a, b, c});
  auto data1 = ascir::cg::ContiguousData("data1", graph, ge::DT_FLOAT16, {a, b, c});
  auto data2 = ascir::cg::ContiguousData("data2", graph, ge::DT_FLOAT16, {a, b, c});

  ascir::cg::FlashSoftmax("fs", data0, data1, data2);

  auto fs = graph.FindNode("fs");
  EXPECT_EQ(fs->outputs[0].attr.dtype, ge::DT_INT32);
  EXPECT_EQ(fs->outputs[1].attr.dtype, ge::DT_INT32);
  EXPECT_EQ(fs->outputs[2].attr.dtype, ge::DT_INT32);
}
TEST(AscirOps, ExecOrderIncreaseOk) {
  ge::AscGraph graph("test_graph");
  auto A = ge::Symbol("A");
  auto B = ge::Symbol("B");
  auto C = ge::Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data0 = ascir::cg::ContiguousData("data0", graph, ge::DT_INT32, {a, b, c});
  auto data1 = ascir::cg::ContiguousData("data1", graph, ge::DT_FLOAT16, {a, b, c});
  auto data2 = ascir::cg::ContiguousData("data2", graph, ge::DT_FLOAT16, {a, b, c});

  ascir::cg::FlashSoftmax("fs", data0, data1, data2);

  auto data0_node = graph.FindNode("data0");
  auto data1_node = graph.FindNode("data1");
  auto data2_node = graph.FindNode("data2");
  auto fs_node = graph.FindNode("fs");
  EXPECT_EQ(static_cast<int64_t>(data0_node->attr.sched.exec_order), 0);
  EXPECT_EQ(static_cast<int64_t>(data1_node->attr.sched.exec_order), 1);
  EXPECT_EQ(static_cast<int64_t>(data2_node->attr.sched.exec_order), 2);
  EXPECT_EQ(static_cast<int64_t>(fs_node->attr.sched.exec_order), 3);
}
};
