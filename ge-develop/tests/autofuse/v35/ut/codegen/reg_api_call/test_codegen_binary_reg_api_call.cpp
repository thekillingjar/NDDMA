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
#include "elewise/binary_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, BinaryApiCallDivExtend) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Div div_op("div");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(div_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  div_op.x2 = load_op.y;
  div_op.x1 = constant_op.y;

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


  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;


  auto div = graph.FindNode("div");
  div->attr.api.unit = ge::ComputeUnit::kUnitVector;
  div->outputs[0].attr.dtype = ge::DT_FLOAT16;
  div->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  div->outputs[0].attr.mem.tensor_id = 2;
  div->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  div->outputs[0].attr.que.id = 2;
  div->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(div->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("DivExtend");
  EXPECT_EQ(call.Init(div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
    "DivExtends<half, true>(local_2[0], local_0[0], (half)1.0, local_0_actual_size);\n"
  });
}
