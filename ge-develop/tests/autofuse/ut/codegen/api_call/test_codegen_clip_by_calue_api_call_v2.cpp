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
#include "elewise/clip_by_value_api_call_v2.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

void ClipByValueV2Float_BeforeAutofuse(ge::AscGraph &graph, codegen::Tiler &tiler, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(ge::SizeVar(s0));

  int order = 0;

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id};

  Data x2("x2");
  graph.AddNode(x2);
  x2.attr.sched.axis = {z0.id};
  x2.y.dtype = data_type;
  *x2.y.axis = {z0.id};

  Data x3("x3");
  graph.AddNode(x3);
  x3.attr.sched.axis = {z0.id};
  x3.y.dtype = data_type;
  *x3.y.axis = {z0.id};

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id};
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {One};

  Load load2("load2");
  graph.AddNode(load2);
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id};
  *load2.y.axis = {z0.id};
  *load2.y.repeats = {s0};
  *load2.y.strides = {One};

  Load load3("load3");
  graph.AddNode(load3);
  load3.x = x3.y;
  load3.attr.sched.axis = {z0.id};
  *load3.y.axis = {z0.id};
  *load3.y.repeats = {s0};
  *load3.y.strides = {One};

  ClipByValue clipbyvalue("clipbyvalue");
  graph.AddNode(clipbyvalue);
  clipbyvalue.x1 = load1.y;
  clipbyvalue.x2 = load2.y;
  clipbyvalue.x3 = load3.y;
  clipbyvalue.attr.sched.axis = {z0.id};
  clipbyvalue.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = clipbyvalue.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
}

void ClipByValueV2Float_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x3 = graph.FindNode("x3");
  x3->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = data_type;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = data_type;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.dtype = data_type;
  load3->attr.api.compute_type = ComputeType::kComputeLoad;

  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->outputs[0].attr.dtype = data_type;
  clipbyvalue->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  clipbyvalue->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  clipbyvalue->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  clipbyvalue->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void ClipByValueV2Float_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto x3 = graph.FindNode("x3");
  x3->attr.api.type = ApiType::kAPITypeBuffer;
  x3->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.type = ApiType::kAPITypeCompute;
  load3->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->attr.api.type = ApiType::kAPITypeCompute;
  clipbyvalue->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void ClipByValueV2Float_AfterScheduler(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z0T->id, z0t->id);
  graph.ApplySplit(load1, z0TB->id, z0Tb->id);
  load1->attr.sched.loop_axis = z0Tb->id;
  load1->outputs[0].attr.vectorized_axis = {z0t->id,};
  load1->outputs[0].attr.vectorized_strides = {One,};

  auto load2 = graph.FindNode("load2");
  graph.ApplySplit(load2, z0T->id, z0t->id);
  graph.ApplySplit(load2, z0TB->id, z0Tb->id);
  load2->attr.sched.loop_axis = z0Tb->id;
  load2->outputs[0].attr.vectorized_axis = {z0t->id,};
  load2->outputs[0].attr.vectorized_strides = {One,};

  auto load3 = graph.FindNode("load3");
  graph.ApplySplit(load3, z0T->id, z0t->id);
  graph.ApplySplit(load3, z0TB->id, z0Tb->id);
  load3->attr.sched.loop_axis = z0Tb->id;
  load3->outputs[0].attr.vectorized_axis = {z0t->id,};
  load3->outputs[0].attr.vectorized_strides = {One,};

  auto clipbyvalue = graph.FindNode("clipbyvalue");
  graph.ApplySplit(clipbyvalue, z0T->id, z0t->id);
  graph.ApplySplit(clipbyvalue, z0TB->id, z0Tb->id);
  clipbyvalue->attr.sched.loop_axis = z0Tb->id;
  clipbyvalue->outputs[0].attr.vectorized_axis = {z0t->id,};
  clipbyvalue->outputs[0].attr.vectorized_strides = {One,};

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0T->id, z0t->id);
  graph.ApplySplit(store, z0TB->id, z0Tb->id);
  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0t->id,};
  store->outputs[0].attr.vectorized_strides = {One,};
}

void ClipByValueV2Float_AfterQueBufAlloc(ge::AscGraph &graph) {
  int tensor_id = 0;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = tensor_id++;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.tensor_id = tensor_id++;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;

  auto x3 = graph.FindNode("x3");
  x3->outputs[0].attr.mem.tensor_id = tensor_id++;
  x3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x3->outputs[0].attr.mem.position = Position::kPositionGM;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = tensor_id++;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 1;
  load1->outputs[0].attr.que.buf_num = 1;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.mem.tensor_id = tensor_id++;
  load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = ge::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 1;
  load2->outputs[0].attr.que.depth = 1;
  load2->outputs[0].attr.que.buf_num = 1;
  load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.mem.tensor_id = tensor_id++;
  load3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load3->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load3->outputs[0].attr.buf.id = ge::kIdNone;
  load3->outputs[0].attr.que.id = 2;
  load3->outputs[0].attr.mem.reuse_id = 3;
  load3->outputs[0].attr.que.depth = 1;
  load3->outputs[0].attr.que.buf_num = 1;
  load3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->outputs[0].attr.mem.tensor_id = tensor_id++;
  clipbyvalue->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  clipbyvalue->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  clipbyvalue->outputs[0].attr.mem.position = Position::kPositionVecOut;
  clipbyvalue->outputs[0].attr.buf.id = ge::kIdNone;
  clipbyvalue->outputs[0].attr.que.id = 2;
  clipbyvalue->outputs[0].attr.mem.reuse_id = 4;
  clipbyvalue->outputs[0].attr.que.depth = 1;
  clipbyvalue->outputs[0].attr.que.buf_num = 1;
  clipbyvalue->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  clipbyvalue->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensor_id++;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
}

void ClipByValueV2Float_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs, codegen::Tiler &tiler,
                                    ge::DataType data_type) {
  ClipByValueV2Float_BeforeAutofuse(graph, tiler, data_type);
  ClipByValueV2Float_AfterInferOutput(graph, data_type);

  impl_graphs.push_back(ge::AscGraph("clip_by_value_float_general_0_nil_0_nil"));
  impl_graphs[0].CopyFrom(graph);
  ClipByValueV2Float_AfterGetApiInfo(impl_graphs[0]);
  ClipByValueV2Float_AfterScheduler(impl_graphs[0]);
  ClipByValueV2Float_AfterQueBufAlloc(impl_graphs[0]);
}

TEST(CodegenKernel, ClipByValueApiCallV2) {
  ge::AscGraph graph("clip_by_value_float");
  std::vector<ge::AscGraph> test_impl_graphs;
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  ClipByValueV2Float_AfterAutofuse(graph, test_impl_graphs, tiler, ge::DT_FLOAT);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto load3 = graph.FindNode("load3");
  auto clipbyvalue = graph.FindNode("clipbyvalue");
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(load3->outputs[0]);
  tpipe.AddTensor(clipbyvalue->outputs[0]);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;

  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;

  codegen::ClipByValueApiCallV2 call("ClipByValue");
  EXPECT_EQ(call.Init(clipbyvalue), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "ClipByValue(global_-1[0], global_-1[0], global_-1[0], global_-1[0], global_-1_actual_size);\n"
  });
}

TEST(CodegenKernel, ClipByValueV2WithSecondInputAndThirdInputAreScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::ClipByValue clipbyvalue_op("clipbyvalue");
  Scalar constant1_op("constant1");
  constant1_op.ir_attr.SetValue("1.0");
  Scalar constant2_op("constant2");
  constant2_op.ir_attr.SetValue("2.0");
  graph.AddNode(load_op);
  graph.AddNode(clipbyvalue_op);
  graph.AddNode(constant1_op);
  graph.AddNode(constant2_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  clipbyvalue_op.x1 = load_op.y;
  clipbyvalue_op.x2 = constant1_op.y;
  clipbyvalue_op.x3 = constant2_op.y;

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


  auto constant1_node = graph.FindNode("constant1");
  constant1_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant1_node->outputs[0].attr.mem.tensor_id = 1;
  constant1_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;
  constant1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;

  auto constant2_node = graph.FindNode("constant2");
  constant2_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant2_node->outputs[0].attr.mem.tensor_id = 2;
  constant2_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;
  constant2_node->outputs[0].attr.dtype = ge::DT_FLOAT16;



  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->attr.api.unit = ge::ComputeUnit::kUnitVector;
  clipbyvalue->outputs[0].attr.dtype = ge::DT_FLOAT16;
  clipbyvalue->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  clipbyvalue->outputs[0].attr.mem.tensor_id = 3;
  clipbyvalue->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  clipbyvalue->outputs[0].attr.que.id = 2;
  clipbyvalue->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor("1.0", constant1_node->outputs[0], "const_x2");
  tpipe.AddTensor("2.0", constant2_node->outputs[0], "const_x3");
  tpipe.AddTensor(clipbyvalue->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2, x3;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = constant1_node->outputs[0].attr.mem.tensor_id;
  x3.id = constant2_node->outputs[0].attr.mem.tensor_id;


  codegen::ClipByValueApiCallV2 call("ClipByValue");
  EXPECT_EQ(call.Init(clipbyvalue), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "ClipByValue(local_3[0], local_0[0], scalar_1, scalar_2, local_0_actual_size);\n"
  });
}


TEST(CodegenKernel, ClipByValueV2WithAllInputAreScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  ge::ascir_op::ClipByValue clipbyvalue_op("clipbyvalue");
  Scalar constant1_op("constant1");
  constant1_op.ir_attr.SetValue("1.2");
  Scalar constant2_op("constant2");
  constant2_op.ir_attr.SetValue("1.0");
  Scalar constant3_op("constant3");
  constant3_op.ir_attr.SetValue("2.0");

  graph.AddNode(constant1_op);
  graph.AddNode(constant2_op);
  graph.AddNode(constant3_op);
  graph.AddNode(clipbyvalue_op);


  clipbyvalue_op.x1 = constant1_op.y;
  clipbyvalue_op.x2 = constant2_op.y;
  clipbyvalue_op.x3 = constant3_op.y;

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

  auto constant3_node = graph.FindNode("constant3");
  constant3_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant3_node->outputs[0].attr.mem.tensor_id = 2;
  constant3_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;
  constant3_node->outputs[0].attr.dtype = ge::DT_FLOAT16;



  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->attr.api.unit = ge::ComputeUnit::kUnitVector;
  clipbyvalue->outputs[0].attr.dtype = ge::DT_FLOAT16;
  clipbyvalue->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  clipbyvalue->outputs[0].attr.mem.tensor_id = 3;
  clipbyvalue->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  clipbyvalue->outputs[0].attr.que.id = 2;
  clipbyvalue->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor("1.2", constant1_node->outputs[0], "const_x1");
  tpipe.AddTensor("1.0", constant2_node->outputs[0], "const_x2");
  tpipe.AddTensor("2.0", constant3_node->outputs[0], "const_x3");
  tpipe.AddTensor(clipbyvalue->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1, x2, x3;
  x1.id = constant1_node->outputs[0].attr.mem.tensor_id;
  x2.id = constant2_node->outputs[0].attr.mem.tensor_id;
  x3.id = constant3_node->outputs[0].attr.mem.tensor_id;


  codegen::ClipByValueApiCallV2 call("ClipByValue");
  EXPECT_EQ(call.Init(clipbyvalue), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  call.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
      "ClipByValue(local_3[0], scalar_0, scalar_1, scalar_2);\n"
  });
}

TEST(WhereApiCallTest, ClipByValueV2WithAllInputAreUbScalar) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Data x_op3("x3", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");
  ge::ascir_op::ClipByValue clipbyvalue_op("clipbyvalue");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(load_op3);
  graph.AddNode(clipbyvalue_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {One, One, One};
  *load_op1.y.strides = {One, One, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {One, One, One};
  *load_op2.y.strides = {One, One, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {One, One, One};
  *load_op3.y.strides = {One, One, One};

  clipbyvalue_op.x1 = load_op1.y;
  clipbyvalue_op.x2 = load_op2.y;
  clipbyvalue_op.x3 = load_op3.y;
  *clipbyvalue_op.y.axis = {z0.id, z1.id, z2.id};
  *clipbyvalue_op.y.repeats = {s0, s1, s2};
  *clipbyvalue_op.y.strides = {s1*s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1->attr.api.type = ge::ApiType::kAPITypeCompute;
  load1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {One, One};
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
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {One, One};
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
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {One, One};
  load3->outputs[0].attr.dtype = ge::DT_FLOAT;
  load3->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  clipbyvalue->attr.api.type = ge::ApiType::kAPITypeCompute;
  clipbyvalue->attr.api.unit = ge::ComputeUnit::kUnitVector;
  clipbyvalue->attr.sched.loop_axis = z0.id;
  clipbyvalue->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  clipbyvalue->outputs[0].attr.vectorized_strides = {s2, One};
  clipbyvalue->outputs[0].attr.dtype = ge::DT_FLOAT;
  clipbyvalue->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  clipbyvalue->outputs[0].attr.mem.tensor_id = 3;
  clipbyvalue->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  clipbyvalue->outputs[0].attr.que.id = 2;
  clipbyvalue->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // begin:构造x1 x2 x3 是ub_scalar的tensor
  std::string dtype_name;
  codegen::Tensor::DtypeName(load1->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t_x1(load1->outputs[0], dtype_name, "t_x1");
  EXPECT_EQ(t_x1.Init(), 0);
  t_x1.need_gen_get_value_of_ub_scalar = true;
  t_x1.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x1), 0);

  codegen::Tensor t_x2(load2->outputs[0], dtype_name, "t_x2");
  EXPECT_EQ(t_x2.Init(), 0);
  t_x2.need_gen_get_value_of_ub_scalar = true;
  t_x2.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x2), 0);

  codegen::Tensor t_x3(load3->outputs[0], dtype_name, "t_x3");
  EXPECT_EQ(t_x3.Init(), 0);
  t_x3.need_gen_get_value_of_ub_scalar = true;
  t_x3.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x3), 0);
  // end
  tpipe.AddTensor(clipbyvalue->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::ClipByValueApiCallV2 call("ClipByValue");
  EXPECT_EQ(call.Init(clipbyvalue), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{
      "ClipByValue(local_3[0], (float)local_0_ub_scalar, (float)local_1_ub_scalar, (float)local_2_ub_scalar);\n"
  });
}