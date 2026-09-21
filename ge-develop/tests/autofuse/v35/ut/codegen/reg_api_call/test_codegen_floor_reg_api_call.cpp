/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "utils/api_call_factory.h"
#include "elewise/unary_api_tmp_v2_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, UnaryApiTmpV2FloorCall) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Floor floor_op("Floor");
  graph.AddNode(load_op);
  graph.AddNode(floor_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  floor_op.x = load_op.y;
  *floor_op.y.axis = {z0.id, z1.id};
  *floor_op.y.repeats = {s0, s1};
  *floor_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto floor = graph.FindNode("Floor");
  floor->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  floor->attr.api.type = ge::ApiType::kAPITypeCompute;
  floor->attr.api.unit = ge::ComputeUnit::kUnitVector;
  floor->attr.sched.loop_axis = z0.id;
  floor->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  floor->outputs[0].attr.vectorized_axis = {z1.id};
  floor->outputs[0].attr.vectorized_strides = {One};
  floor->outputs[0].attr.dtype = ge::DT_FLOAT;
  floor->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  floor->outputs[0].attr.mem.tensor_id = 1;
  floor->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  floor->outputs[0].attr.que.id = 2;
  floor->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(floor->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::UnaryApiTmpV2Call call("Floor");
  EXPECT_EQ(call.Init(floor), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "Floor(local_1[0], local_0[0], tmp_buf_0, local_0_actual_size);\n"
  });
}
