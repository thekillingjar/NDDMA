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
#include "elewise/binary_tmp_api_call.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, BinaryTmpApicall) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::LogicalOr logical_and_op("logical_and");
  graph.AddNode(logical_and_op);

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

  logical_and_op.x1 = load_op.y;
  logical_and_op.x2 = load_op2.y;
  *logical_and_op.y.axis = {z0.id, z1.id};
  *logical_and_op.y.repeats = {s0, s1};
  *logical_and_op.y.strides = {s1, One};

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

  auto logical_and = graph.FindNode("logical_and");
  logical_and->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  logical_and->attr.api.type = ge::ApiType::kAPITypeCompute;
  logical_and->attr.api.unit = ge::ComputeUnit::kUnitVector;
  logical_and->attr.sched.loop_axis = z0.id;
  logical_and->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  logical_and->outputs[0].attr.vectorized_axis = {z1.id};
  logical_and->outputs[0].attr.vectorized_strides = {One};
  logical_and->outputs[0].attr.dtype = ge::DT_INT16;
  logical_and->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  logical_and->outputs[0].attr.mem.tensor_id = 3;
  logical_and->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  logical_and->outputs[0].attr.que.id = 2;
  logical_and->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(logical_and->outputs[0]);

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

  codegen::BinaryTmpApiCall call("LogicalAnd");
  EXPECT_EQ(call.Init(logical_and), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{
    "LogicalAnd(local_3[0], local_0[0], local_1[0], local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, BinaryTmpApicallLogicalOrWithFloatUbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::LogicalOr logical_or_op("logical_or");
  graph.AddNode(logical_or_op);

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

  logical_or_op.x1 = load_op.y;
  logical_or_op.x2 = load_op2.y;
  *logical_or_op.y.axis = {z0.id, z1.id};
  *logical_or_op.y.repeats = {s0, s1};
  *logical_or_op.y.strides = {s1, One};

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

  auto logical_or = graph.FindNode("logical_or");
  logical_or->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  logical_or->attr.api.type = ge::ApiType::kAPITypeCompute;
  logical_or->attr.api.unit = ge::ComputeUnit::kUnitVector;
  logical_or->attr.sched.loop_axis = z0.id;
  logical_or->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  logical_or->outputs[0].attr.vectorized_axis = {z1.id};
  logical_or->outputs[0].attr.vectorized_strides = {One};
  logical_or->outputs[0].attr.dtype = ge::DT_INT16;
  logical_or->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  logical_or->outputs[0].attr.mem.tensor_id = 3;
  logical_or->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  logical_or->outputs[0].attr.que.id = 2;
  logical_or->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为LogicalOr的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(logical_or->outputs[0]);

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

  codegen::BinaryTmpApiCall call("LogicalOr");
  EXPECT_EQ(call.Init(logical_or), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{
    "LogicalOrScalarExtend(local_3[0], local_0[0], (float)local_1_ub_scalar, local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, BinaryTmpApicallLogicalOrWithUint8UbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::LogicalOr logical_or_op("logical_or");
  graph.AddNode(logical_or_op);

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

  logical_or_op.x1 = load_op.y;
  logical_or_op.x2 = load_op2.y;
  *logical_or_op.y.axis = {z0.id, z1.id};
  *logical_or_op.y.repeats = {s0, s1};
  *logical_or_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_UINT8;
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
  load2->outputs[0].attr.dtype = ge::DT_UINT8;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto logical_or = graph.FindNode("logical_or");
  logical_or->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  logical_or->attr.api.type = ge::ApiType::kAPITypeCompute;
  logical_or->attr.api.unit = ge::ComputeUnit::kUnitVector;
  logical_or->attr.sched.loop_axis = z0.id;
  logical_or->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  logical_or->outputs[0].attr.vectorized_axis = {z1.id};
  logical_or->outputs[0].attr.vectorized_strides = {One};
  logical_or->outputs[0].attr.dtype = ge::DT_INT16;
  logical_or->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  logical_or->outputs[0].attr.mem.tensor_id = 3;
  logical_or->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  logical_or->outputs[0].attr.que.id = 2;
  logical_or->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为LogicalOr的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(logical_or->outputs[0]);

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

  codegen::BinaryTmpApiCall call("LogicalOr");
  EXPECT_EQ(call.Init(logical_or), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{
    "LogicalOrScalarExtend(local_3[0], local_0[0], (uint8_t)local_1_ub_scalar, local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, BinaryTmpApicallLogicalOrWhenBothAreTensor) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::LogicalOr logical_or_op("logical_or");
  graph.AddNode(logical_or_op);

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

  logical_or_op.x1 = load_op.y;
  logical_or_op.x2 = load_op2.y;
  *logical_or_op.y.axis = {z0.id, z1.id};
  *logical_or_op.y.repeats = {s0, s1};
  *logical_or_op.y.strides = {s1, One};

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

  auto logical_or = graph.FindNode("logical_or");
  logical_or->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  logical_or->attr.api.type = ge::ApiType::kAPITypeCompute;
  logical_or->attr.api.unit = ge::ComputeUnit::kUnitVector;
  logical_or->attr.sched.loop_axis = z0.id;
  logical_or->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  logical_or->outputs[0].attr.vectorized_axis = {z1.id};
  logical_or->outputs[0].attr.vectorized_strides = {One};
  logical_or->outputs[0].attr.dtype = ge::DT_INT16;
  logical_or->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  logical_or->outputs[0].attr.mem.tensor_id = 3;
  logical_or->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  logical_or->outputs[0].attr.que.id = 2;
  logical_or->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(logical_or->outputs[0]);

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

  codegen::BinaryTmpApiCall call("LogicalOr");
  EXPECT_EQ(call.Init(logical_or), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{
    "LogicalOr(local_3[0], local_0[0], local_1[0], local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, BinaryTmpApicallBitwiseAnd) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::BitwiseAnd logical_and_op("bitwise_and");
  graph.AddNode(logical_and_op);

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

  logical_and_op.x1 = load_op.y;
  logical_and_op.x2 = load_op2.y;
  *logical_and_op.y.axis = {z0.id, z1.id};
  *logical_and_op.y.repeats = {s0, s1};
  *logical_and_op.y.strides = {s1, One};

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

  auto logical_and = graph.FindNode("bitwise_and");
  logical_and->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  logical_and->attr.api.type = ge::ApiType::kAPITypeCompute;
  logical_and->attr.api.unit = ge::ComputeUnit::kUnitVector;
  logical_and->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  logical_and->attr.sched.loop_axis = z0.id;
  logical_and->outputs[0].attr.vectorized_axis = {z1.id};
  logical_and->outputs[0].attr.vectorized_strides = {One};
  logical_and->outputs[0].attr.dtype = ge::DT_INT16;
  logical_and->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  logical_and->outputs[0].attr.mem.tensor_id = 3;
  logical_and->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  logical_and->outputs[0].attr.que.id = 2;
  logical_and->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(logical_and->outputs[0]);

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

  codegen::BinaryTmpApiCall call("BitwiseAnd");
  EXPECT_EQ(call.Init(logical_and), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{
    "BitwiseAnd(local_3[0], local_0[0], local_1[0], local_0_actual_size, tmp_buf_0);\n"
  });
}
