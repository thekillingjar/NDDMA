/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "e2e_clip_by_value_float.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void PowInt32_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

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

  Pow pow("pow");
  graph.AddNode(pow);
  pow.x1 = load1.y;
  pow.x2 = load2.y;
  pow.attr.sched.axis = {z0.id};
  pow.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = pow.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
}

void PowInt32_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = data_type;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = data_type;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;

  auto pow = graph.FindNode("pow");
  pow->outputs[0].attr.dtype = data_type;
  pow->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  pow->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  pow->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  pow->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->outputs[0].attr.dtype = data_type;
}

void PowInt32_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto pow = graph.FindNode("pow");
  pow->attr.api.type = ApiType::kAPITypeCompute;
  pow->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void PowInt32_AfterScheduler(ge::AscGraph &graph) {
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

  auto pow = graph.FindNode("pow");
  graph.ApplySplit(pow, z0T->id, z0t->id);
  graph.ApplySplit(pow, z0TB->id, z0Tb->id);
  pow->attr.sched.loop_axis = z0Tb->id;
  pow->outputs[0].attr.vectorized_axis = {z0t->id,};
  pow->outputs[0].attr.vectorized_strides = {One,};

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0T->id, z0t->id);
  graph.ApplySplit(store, z0TB->id, z0Tb->id);
  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0t->id,};
  store->outputs[0].attr.vectorized_strides = {One,};
  cout << utils::DebugHintGraphStr(graph) << endl;
}

void PowInt32_AfterQueBufAlloc(ge::AscGraph &graph) {
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

  auto pow = graph.FindNode("pow");
  pow->outputs[0].attr.mem.tensor_id = tensor_id++;
  pow->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  pow->outputs[0].attr.mem.position = Position::kPositionVecOut;
  pow->outputs[0].attr.buf.id = ge::kIdNone;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.mem.reuse_id = 2;
  pow->outputs[0].attr.que.depth = 1;
  pow->outputs[0].attr.que.buf_num = 1;
  pow->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  pow->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensor_id++;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
}

void PowInt32_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs,
                                            ge::DataType data_type) {
  PowInt32_BeforeAutofuse(graph, data_type);
  PowInt32_AfterInferOutput(graph, data_type);

  impl_graphs.push_back(ge::AscGraph("pow_int32_general_0_nil_0_nil"));
  impl_graphs[0].CopyFrom(graph);
  PowInt32_AfterGetApiInfo(impl_graphs[0]);
  PowInt32_AfterScheduler(impl_graphs[0]);
  PowInt32_AfterQueBufAlloc(impl_graphs[0]);
}