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
#include "elewise/leaky_relu_api_call.h"
#include "elewise/binary_tmp_api_call_v2.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, BinaryTmpApicallV2) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::Mod mod_op("mod");
  graph.AddNode(mod_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  mod_op.x1 = load_op.y;
  mod_op.x2 = load_op2.y;
  *mod_op.y.axis = {z0.id, z1.id};
  *mod_op.y.repeats = {s0, s1};
  *mod_op.y.strides = {s1, One};

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

  auto mod = graph.FindNode("mod");
  mod->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  mod->attr.api.type = ge::ApiType::kAPITypeCompute;
  mod->attr.api.unit = ge::ComputeUnit::kUnitVector;
  mod->attr.sched.loop_axis = z0.id;
  mod->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  mod->outputs[0].attr.vectorized_axis = {z1.id};
  mod->outputs[0].attr.vectorized_strides = {One};
  mod->outputs[0].attr.dtype = ge::DT_INT16;
  mod->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mod->outputs[0].attr.mem.tensor_id = 3;
  mod->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mod->outputs[0].attr.que.id = 2;
  mod->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(mod->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryTmpApiCallV2 call("Fmod");
  EXPECT_EQ(call.Init(mod), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{
    "local_0.SetSize(local_0_actual_size);\nlocal_1.SetSize(local_1_actual_size);\nFmod(local_3[0], local_0[0], local_1[0], tmp_buf_0, local_0_actual_size);\n"
  });
}