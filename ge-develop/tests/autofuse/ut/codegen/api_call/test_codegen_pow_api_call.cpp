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
#include "elewise/pow_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, PowWithSecondInputIsUbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::Pow pow_op("pow");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.2");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(pow_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {One, One};
  *load_op2.y.strides = {One, One};

  pow_op.x1 = load_op.y;
  pow_op.x2 = load_op2.y;

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

  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 3;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 3;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // pow load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // pow load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add pow tensor
  EXPECT_EQ(tpipe.AddTensor(pow->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{
      "Pow(local_3[0], local_0[0], (half)local_2_ub_scalar, local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, PowWithFirstInputIsUbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::Pow pow_op("pow");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.2");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(pow_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {One, One};
  *load_op.y.strides = {One, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  pow_op.x1 = load_op.y;
  pow_op.x2 = load_op2.y;

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

  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 3;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 3;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  // pow load1 tensor // 构造load1是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // pow load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);
  // add pow tensor
  EXPECT_EQ(tpipe.AddTensor(pow->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{
      "Pow(local_3[0], (half)local_0_ub_scalar, local_2[0], local_2_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, PowWithFirstInputIsScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Pow pow_op("pow");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(pow_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  pow_op.x2 = load_op.y;
  pow_op.x1 = constant_op.y;

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
  constant_node->outputs[0].attr.dtype = ge::DT_FLOAT16;


  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(pow->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;
  x2.id = load->outputs[0].attr.mem.tensor_id;

  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "Pow(local_2[0], scalar_1, local_0[0], local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, PowWithAllInputIsTensor) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load2_op("load2");
  ge::ascir_op::Pow pow_op("pow");

  graph.AddNode(load_op);
  graph.AddNode(pow_op);
  graph.AddNode(load2_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load2_op.x = x_op.y;
  load2_op.attr.sched.axis = {z0.id, z1.id};
  *load2_op.y.axis = {z0.id, z1.id};
  *load2_op.y.repeats = {s0, s1};
  *load2_op.y.strides = {s1, One};

  pow_op.x1 = load_op.y;
  pow_op.x2 = load2_op.y;

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
  load2->outputs[0].attr.mem.tensor_id = 0;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(pow->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;


  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "Pow(local_2[0], local_0[0], local_0[0], local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, PowWithAllInputIsScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Pow pow_op("pow");
  Scalar constant1_op("constant1");
  constant1_op.ir_attr.SetValue("1.0");
  Scalar constant2_op("constant2");
  constant2_op.ir_attr.SetValue("2.0");
  graph.AddNode(constant1_op);
  graph.AddNode(constant2_op);
  graph.AddNode(pow_op);

  pow_op.x1 = constant1_op.y;
  pow_op.x2 = constant2_op.y;

  auto constant1_node = graph.FindNode("constant1");
  constant1_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant1_node->outputs[0].attr.mem.tensor_id = 0;
  constant1_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;
  constant1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;

  auto constant2_node = graph.FindNode("constant2");
  constant2_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant2_node->outputs[0].attr.mem.tensor_id = 1;
  constant2_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;
  constant2_node->outputs[0].attr.dtype = ge::DT_FLOAT16;

  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor("1.0", constant1_node->outputs[0], "const_1");
  tpipe.AddTensor("2.0", constant2_node->outputs[0], "const_2");
  tpipe.AddTensor(pow->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant1_node->outputs[0].attr.mem.tensor_id;
  x2.id = constant2_node->outputs[0].attr.mem.tensor_id;

  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{
    "Pow(local_2[0], scalar_0, scalar_1, local_2_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, PowWitAllInputIsUbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1_op("x1", graph);
  Data x2_op("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  ge::ascir_op::Pow pow_op("pow");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(pow_op);

  load_op.x = x1_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {One, One};
  *load_op.y.strides = {One, One};

  load_op2.x = x2_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {One, One};
  *load_op2.y.strides = {One, One};

  pow_op.x1 = load_op.y;
  pow_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 0;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 0;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  // pow load1 tensor // 构造load1是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t1(load->outputs[0], dtype_name, "t1");
  EXPECT_EQ(t1.Init(), 0);
  t1.need_gen_get_value_of_ub_scalar = true;
  t1.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t1), 0);

  codegen::Tensor t2(load2->outputs[0], dtype_name, "t2");
  EXPECT_EQ(t2.Init(), 0);
  t2.need_gen_get_value_of_ub_scalar = true;
  t2.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t2), 0);

  // add pow tensor
  EXPECT_EQ(tpipe.AddTensor(pow->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{
      "Pow(local_2[0], (half)local_0_ub_scalar, (half)local_1_ub_scalar, local_2_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, PowWithInputIsScalarAndUbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  ge::ascir_op::Pow pow_op("pow");
  Scalar constant1_op("constant1");
  constant1_op.ir_attr.SetValue("1.0");
  Load load_op("load");
  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {One, One};
  *load_op.y.strides = {One, One};

  graph.AddNode(constant1_op);
  graph.AddNode(load_op);
  graph.AddNode(pow_op);

  pow_op.x1 = constant1_op.y;
  pow_op.x2 = load_op.y;

  auto constant1_node = graph.FindNode("constant1");
  constant1_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant1_node->outputs[0].attr.mem.tensor_id = 0;
  constant1_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;
  constant1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto pow = graph.FindNode("pow");
  pow->attr.api.unit = ge::ComputeUnit::kUnitVector;
  pow->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  pow->outputs[0].attr.dtype = ge::DT_FLOAT16;
  pow->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 0;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  tpipe.AddTensor("1.0", constant1_node->outputs[0], "const_1");
  std::string dtype_name;
  codegen::Tensor::DtypeName(load->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  tpipe.AddTensor(pow->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant1_node->outputs[0].attr.mem.tensor_id;
  x2.id = load->outputs[0].attr.mem.tensor_id;

  codegen::PowApiCall call("Pow");
  EXPECT_EQ(call.Init(pow), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<ge::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{
    "Pow(local_2[0], scalar_0, (half)local_1_ub_scalar, local_2_actual_size, tmp_buf_0);\n"
  });
}