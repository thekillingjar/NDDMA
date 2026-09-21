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
#include "micro_api_call_factory.h"
#include "micro_cast_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, CastMicroApiCall_float_2_int64) {
  ge::AscGraph graph("test_graph");

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

  Data x_op("x", graph);
  Load load_op("load");
  Cast cast_op("cast");
  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  cast_op.x = load_op.y;
  cast_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast_op.y.repeats = {s0, s1, s2, s3};
  *cast_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  cast->attr.api.type = ge::ApiType::kAPITypeCompute;
  cast->attr.api.unit = ge::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  cast->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  cast->outputs[0].attr.dtype = ge::DT_INT64;
  cast->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_INT64;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

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

  auto load_output_name = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_output_tensor(load->outputs[0], load_output_name);

  auto cast_output_name = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor cast_output_tensor(cast->outputs[0], cast_output_name);

  TensorManager tensor_mng;
  codegen::CallParam cp = {"p_reg", "offset"};
  tensor_mng.AddTensor(load_output_tensor);
  tensor_mng.AddTensor(cast_output_tensor);
  codegen::MicroCastApiCall vf_cast("Cast");
  EXPECT_EQ(vf_cast.Init(cast), 0);
  vf_cast.AddInput(load->outputs[0].attr.mem.tensor_id);
  vf_cast.AddOutput(cast->outputs[0].attr.mem.tensor_id);

  std::string result;
  vf_cast.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Cast<int64_t, float, cast_trait_float_2_int64>(vreg_1, vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, CastMicroApiCall_float_2_half) {
  ge::AscGraph graph("test_graph");

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

  Data x_op("x", graph);
  Load load_op("load");
  Cast cast_op("cast");
  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  cast_op.x = load_op.y;
  cast_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast_op.y.repeats = {s0, s1, s2, s3};
  *cast_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  cast->attr.api.type = ge::ApiType::kAPITypeCompute;
  cast->attr.api.unit = ge::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  cast->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  cast->outputs[0].attr.dtype = ge::DT_FLOAT16;
  cast->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

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

  auto load_output_name = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_output_tensor(load->outputs[0], load_output_name);

  auto cast_output_name = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor cast_output_tensor(cast->outputs[0], cast_output_name);

  TensorManager tensor_mng;
  codegen::CallParam cp = {"p_reg", "offset"};
  tensor_mng.AddTensor(load_output_tensor);
  tensor_mng.AddTensor(cast_output_tensor);
  codegen::MicroCastApiCall vf_cast("Cast");
  EXPECT_EQ(vf_cast.Init(cast), 0);
  vf_cast.AddInput(load->outputs[0].attr.mem.tensor_id);
  vf_cast.AddOutput(cast->outputs[0].attr.mem.tensor_id);

  std::string result;
  vf_cast.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Cast<half, float, cast_trait_float_2_half>(vreg_1, vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, CastMicroApiCall_half_2_float) {
  ge::AscGraph graph("test_graph");

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

  Data x_op("x", graph);
  Load load_op("load");
  Cast cast_op("cast");
  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  cast_op.x = load_op.y;
  cast_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *cast_op.y.repeats = {s0, s1, s2, s3};
  *cast_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  cast->attr.api.type = ge::ApiType::kAPITypeCompute;
  cast->attr.api.unit = ge::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  cast->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  cast->outputs[0].attr.dtype = ge::DT_FLOAT;
  cast->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

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

  auto load_output_name = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_output_tensor(load->outputs[0], load_output_name);

  auto cast_output_name = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor cast_output_tensor(cast->outputs[0], cast_output_name);

  TensorManager tensor_mng;
  codegen::CallParam cp = {"p_reg", "offset"};
  tensor_mng.AddTensor(load_output_tensor);
  tensor_mng.AddTensor(cast_output_tensor);
  codegen::MicroCastApiCall vf_cast("Cast");
  EXPECT_EQ(vf_cast.Init(cast), 0);
  vf_cast.AddInput(load->outputs[0].attr.mem.tensor_id);
  vf_cast.AddOutput(cast->outputs[0].attr.mem.tensor_id);

  std::string result;
  vf_cast.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Cast<float, half, cast_trait_bf16_2_float>(vreg_1, vreg_0, p_reg);\n"});
}