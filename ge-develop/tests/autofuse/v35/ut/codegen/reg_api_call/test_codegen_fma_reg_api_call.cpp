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
#include "elewise/ternary_api_tmp_v2_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, TernaryApiTmpV2FmaCall) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1_op("x1", graph);
  Data x2_op("x2", graph);
  Data x3_op("x3", graph);
  Load load1_op("load1");
  Load load2_op("load2");
  Load load3_op("load3");
  ge::ascir_op::Fma fma_op("Fma");
  graph.AddNode(load1_op);
  graph.AddNode(load2_op);
  graph.AddNode(load3_op);
  graph.AddNode(fma_op);

  load1_op.x = x1_op.y;
  load1_op.attr.sched.axis = {z0.id, z1.id};
  *load1_op.y.axis = {z0.id, z1.id};
  *load1_op.y.repeats = {s0, s1};
  *load1_op.y.strides = {s1, One};
  load2_op.x = x2_op.y;
  load2_op.attr.sched.axis = {z0.id, z1.id};
  *load2_op.y.axis = {z0.id, z1.id};
  *load2_op.y.repeats = {s0, s1};
  *load2_op.y.strides = {s1, One};
  load3_op.x = x3_op.y;
  load3_op.attr.sched.axis = {z0.id, z1.id};
  *load3_op.y.axis = {z0.id, z1.id};
  *load3_op.y.repeats = {s0, s1};
  *load3_op.y.strides = {s1, One};
  fma_op.x1 = load1_op.y;
  fma_op.x2 = load2_op.y;
  fma_op.x3 = load3_op.y;
  *fma_op.y.axis = {z0.id, z1.id};
  *fma_op.y.repeats = {s0, s1};
  *fma_op.y.strides = {s1, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1->attr.api.type = ge::ApiType::kAPITypeCompute;
  load1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.dtype = ge::DT_FLOAT;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load3->attr.api.type = ge::ApiType::kAPITypeCompute;
  load3->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id};
  load3->outputs[0].attr.vectorized_strides = {One};
  load3->outputs[0].attr.dtype = ge::DT_FLOAT;
  load3->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto fma = graph.FindNode("Fma");
  fma->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  fma->attr.api.type = ge::ApiType::kAPITypeCompute;
  fma->attr.api.unit = ge::ComputeUnit::kUnitVector;
  fma->attr.sched.loop_axis = z0.id;
  fma->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  fma->outputs[0].attr.vectorized_axis = {z1.id};
  fma->outputs[0].attr.vectorized_strides = {One};
  fma->outputs[0].attr.dtype = ge::DT_FLOAT;
  fma->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  fma->outputs[0].attr.mem.tensor_id = 3;
  fma->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  fma->outputs[0].attr.que.id = 2;
  fma->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(load3->outputs[0]);
  tpipe.AddTensor(fma->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  codegen::ApiTensor x3;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  x3.id = load3->outputs[0].attr.mem.tensor_id;

  codegen::TernaryApiTmpV2Call call("Fma");
  EXPECT_EQ(call.Init(fma), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "Fma(local_3[0], local_0[0], local_1[0], local_2[0], tmp_buf_0, local_0_actual_size);\n"
  });
}
