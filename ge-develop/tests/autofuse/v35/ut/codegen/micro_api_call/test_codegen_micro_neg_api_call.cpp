/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#include "gtest/gtest.h"
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "micro_api_call_factory.h"
#include "micro_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

// Test Neg with INT8 type
TEST(CodegenKernel, MicroNegApiCall_Load_Neg_INT8_Store) {
  ge::AscGraph graph("test_neg_int8_graph");

  ge::Expression Two = ge::Symbol(2);
  ge::Expression Three = ge::Symbol(3);
  ge::Expression Four = ge::Symbol(4);

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(4);
  auto s3 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data x_op("x", graph);
  ge::ascir_op::Load load_op("load");
  ge::ascir_op::Neg neg_op("neg");
  ge::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(neg_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  neg_op.x = load_op.y;
  neg_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *neg_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *neg_op.y.repeats = {s0, s1, s2, s3};
  *neg_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = neg_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_INT8;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto neg_node = graph.FindNode("neg");
  neg_node->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  neg_node->attr.api.type = ge::ApiType::kAPITypeCompute;
  neg_node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  neg_node->attr.sched.loop_axis = z0.id;
  neg_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  neg_node->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  neg_node->outputs[0].attr.dtype = ge::DT_INT8;
  neg_node->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  neg_node->outputs[0].attr.mem.tensor_id = 1;
  neg_node->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  neg_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  neg_node->outputs[0].attr.que.id = 2;
  neg_node->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_INT8;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddSizeVar(ge::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroApiCall call("Neg");
  EXPECT_EQ(call.Init(neg_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Neg(vreg_1, vreg_0, p_reg);\n"});
}

// Test Neg with INT64 type
TEST(CodegenKernel, MicroNegApiCall_Load_Neg_INT64_Store) {
  ge::AscGraph graph("test_neg_int64_graph");

  ge::Expression Two = ge::Symbol(2);
  ge::Expression Three = ge::Symbol(3);
  ge::Expression Four = ge::Symbol(4);

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(4);
  auto s3 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data x_op("x", graph);
  ge::ascir_op::Load load_op("load");
  ge::ascir_op::Neg neg_op("neg");
  ge::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(neg_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  neg_op.x = load_op.y;
  neg_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *neg_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *neg_op.y.repeats = {s0, s1, s2, s3};
  *neg_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = neg_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto neg_node = graph.FindNode("neg");
  neg_node->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  neg_node->attr.api.type = ge::ApiType::kAPITypeCompute;
  neg_node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  neg_node->attr.sched.loop_axis = z0.id;
  neg_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  neg_node->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  neg_node->outputs[0].attr.dtype = ge::DT_INT64;
  neg_node->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  neg_node->outputs[0].attr.mem.tensor_id = 1;
  neg_node->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  neg_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  neg_node->outputs[0].attr.que.id = 2;
  neg_node->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_INT64;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddSizeVar(ge::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroApiCall call("Neg");
  EXPECT_EQ(call.Init(neg_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Neg(vreg_1, vreg_0, p_reg);\n"});
}
