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
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "cube_api_call/matmul/matmul_api_call.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;
using namespace codegen;
namespace ge{
namespace ascir{

class CubeApiCallTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/* cube CodeGen UT测试用例 */
TEST(CubeApiCallTest, CubeApiCall) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::MatMul mm("mm");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(mm);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  mm.x1 = load_op.y;
  mm.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto matmul = graph.FindNode("mm");
  matmul->attr.api.unit = ge::ComputeUnit::kUnitVector;
  matmul->outputs[0].attr.dtype = ge::DT_FLOAT16;
  matmul->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  matmul->outputs[0].attr.mem.tensor_id = 3;
  matmul->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  matmul->outputs[0].attr.que.id = 3;
  matmul->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  EXPECT_EQ(tpipe.AddTensor(matmul->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::MatmulApiCall call("mm");
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  std::stringstream ss;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), 0);
  EXPECT_EQ(call.PreProcess(tpipe, vector<ge::AxisId>{}, {}, result), 0);
  EXPECT_EQ(call.PostProcess(tpipe, vector<ge::AxisId>{}, {}, result), 0);
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, {}, {}, result), 0);
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, result), 0);
}
} // namespace ascir
} // namespace ge