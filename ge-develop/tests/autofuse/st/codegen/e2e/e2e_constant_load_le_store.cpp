/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "e2e_constant_load_le_store.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void ConstantLoadLeStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  int order = 0;

  Scalar constant("constant");
  graph.AddNode(constant);
  constant.y.dtype = ge::DT_FLOAT16;
  constant.ir_attr.SetValue("0.5");

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id};
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id};
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  Le le("le");
  graph.AddNode(le);
  le.x1 = load.y;
  le.x2 = constant.y;
  le.attr.sched.axis = {z0.id};
  le.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = le.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
}

void ConstantLoadLeStore_AfterInferOutput(ge::AscGraph &graph) {
  auto constant = graph.FindNode("constant");
  constant->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->attr.api.compute_type = ComputeType::kComputeLoad;

  auto le = graph.FindNode("le");
  le->outputs[0].attr.dtype = ge::DT_UINT8;
  le->outputs[0].attr.axis = load->outputs[0].attr.axis;
  le->outputs[0].attr.repeats = load->outputs[0].attr.repeats;
  le->outputs[0].attr.strides = load->outputs[0].attr.strides;
  le->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = ge::DT_UINT8;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->outputs[0].attr.dtype = ge::DT_UINT8;
}

void ConstantLoadLeStore_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x = graph.FindNode("x");
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;

  auto constant = graph.FindNode("constant");
  constant->attr.api.type = ApiType::kAPITypeBuffer;
  constant->attr.api.unit = ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto le = graph.FindNode("le");
  le->attr.api.type = ApiType::kAPITypeCompute;
  le->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void ConstantLoadLeStore_AfterScheduler(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z0T->id, z0t->id);
  graph.ApplySplit(load, z0TB->id, z0Tb->id);
  load->attr.sched.loop_axis = z0Tb->id;
  load->outputs[0].attr.vectorized_axis = {z0t->id,};
  load->outputs[0].attr.vectorized_strides = {One,};

  auto le = graph.FindNode("le");
  graph.ApplySplit(le, z0T->id, z0t->id);
  graph.ApplySplit(le, z0TB->id, z0Tb->id);
  le->attr.sched.loop_axis = z0Tb->id;
  le->outputs[0].attr.vectorized_axis = {z0t->id,};
  le->outputs[0].attr.vectorized_strides = {One,};

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0T->id, z0t->id);
  graph.ApplySplit(store, z0TB->id, z0Tb->id);
  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0t->id,};
  store->outputs[0].attr.vectorized_strides = {One,};
  cout << utils::DebugHintGraphStr(graph) << endl;
}

void ConstantLoadLeStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  int tensor_id = 0;

  auto constant = graph.FindNode("constant");
  constant->outputs[0].attr.mem.tensor_id = tensor_id++;
  constant->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeInvalid;
  constant->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareInvalid;
  constant->outputs[0].attr.mem.position = Position::kPositionInvalid;

  auto x = graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = tensor_id++;
  x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = MemHardware:: kMemHardwareGM;
  x->outputs[0].attr.mem.position = Position::kPositionGM;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensor_id++;
  load->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = MemHardware:: kMemHardwareUB;
  load->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 1;
  load->outputs[0].attr.que.buf_num = 1;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto le = graph.FindNode("le");
  le->outputs[0].attr.mem.tensor_id = tensor_id++;
  le->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  le->outputs[0].attr.mem.hardware = MemHardware:: kMemHardwareUB;
  le->outputs[0].attr.mem.position = Position::kPositionVecOut;
  le->outputs[0].attr.buf.id = ge::kIdNone;
  le->outputs[0].attr.que.id = 1;
  le->outputs[0].attr.mem.reuse_id = 1;
  le->outputs[0].attr.que.depth = 1;
  le->outputs[0].attr.que.buf_num = 1;
  le->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  le->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensor_id++;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware:: kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
}

void ConstantLoadLeStore_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs) {
  ConstantLoadLeStore_BeforeAutofuse(graph);
  ConstantLoadLeStore_AfterInferOutput(graph);

  impl_graphs.push_back(ge::AscGraph("constant_load_le_store_general_0_nil_0_nil"));
  impl_graphs[0].CopyFrom(graph);
  ConstantLoadLeStore_AfterGetApiInfo(impl_graphs[0]);
  ConstantLoadLeStore_AfterScheduler(impl_graphs[0]);
  ConstantLoadLeStore_AfterQueBufAlloc(impl_graphs[0]);
}
