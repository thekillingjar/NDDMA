/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void BroadcastLoadStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x0("x0");
  graph.AddNode(x0);
  x0.attr.sched.axis = {z0.id, z1.id};
  x0.y.dtype = ge::DT_FLOAT;
  *(x0.y.axis) = {z0.id, z1.id};

  Load load0("load0");
  graph.AddNode(load0);
  load0.x = x0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {One, Zero};

  Broadcast broadcast("broadcast");
  graph.AddNode(broadcast);
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};
  broadcast.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *(x1.y.axis) = {z0.id, z1.id};

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};

  ge::ascir_op::Mul mul("mul");
  graph.AddNode(mul);
  mul.x1 = broadcast.y;
  mul.x2 = load1.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, One};

  Store store("store");
  graph.AddNode(store);
  store.x = mul.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
}

void BroadcastLoadStore_AfterInferOutput(ge::AscGraph &graph) {
  auto x0 = graph.FindNode("x0");
  x0->attr.api.compute_type = ComputeType::kComputeInvalid;  // ComputeType::COMPUTE_DATA;

  auto load0 = graph.FindNode("load0");
  load0->outputs[0].attr.dtype = ge::DT_FLOAT;
  load0->attr.api.compute_type = ComputeType::kComputeLoad;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast->outputs[0].attr.dtype = ge::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;  // ComputeType::COMPUTE_DATA;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = ge::DT_FLOAT;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto mul = graph.FindNode("mul");
  mul->outputs[0].attr.dtype = (ge::DataType)load1->outputs[0].attr.dtype;
  mul->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  mul->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  mul->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  mul->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = (ge::DataType)mul->outputs[0].attr.dtype;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void BroadcastLoadStore_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x0 = graph.FindNode("x0");
  x0->attr.api.type = ApiType::kAPITypeBuffer;
  x0->attr.api.unit = ComputeUnit::kUnitNone;

  auto load0 = graph.FindNode("load0");
  load0->attr.api.type = ApiType::kAPITypeCompute;
  load0->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.type = ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = ComputeUnit::kUnitVector;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto mul = graph.FindNode("mul");
  mul->attr.api.type = ApiType::kAPITypeCompute;
  mul->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void BroadcastLoadStore_AfterScheduler_z1_SplitTo_z0z1TBz0z1Tbz1t(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T->id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0z1B, z0z1b] = graph.BlockSplit(block_axis->id);
  vector<AxisId> vectorized_axis{z1t->id};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);

  auto load0 = graph.FindNode("load0");
  graph.ApplySplit(load0, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load0, block_axis->id);
  graph.ApplySplit(load0, z0z1B->id, z0z1b->id);
  load0->attr.sched.loop_axis = z0z1b->id;
  load0->outputs[0].attr.vectorized_axis = vectorized_axis;
  load0->outputs[0].attr.vectorized_strides = {Zero};

  auto broadcast = graph.FindNode("broadcast");
  graph.ApplySplit(broadcast, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(broadcast, block_axis->id);
  graph.ApplySplit(broadcast, z0z1B->id, z0z1b->id);
  broadcast->attr.sched.loop_axis = z0z1b->id;
  broadcast->outputs[0].attr.vectorized_axis = vectorized_axis;
  broadcast->outputs[0].attr.vectorized_strides = {One};
  broadcast->attr.api.unit = ComputeUnit::kUnitVector;

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load1, block_axis->id);
  graph.ApplySplit(load1, z0z1B->id, z0z1b->id);
  load1->attr.sched.loop_axis = z0z1b->id;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = {One};

  auto mul = graph.FindNode("mul");
  graph.ApplySplit(mul, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(mul, block_axis->id);
  graph.ApplySplit(mul, z0z1B->id, z0z1b->id);
  mul->attr.sched.loop_axis = z0z1b->id;
  mul->outputs[0].attr.vectorized_axis = vectorized_axis;
  mul->outputs[0].attr.vectorized_strides = {One};

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(store, block_axis->id);
  graph.ApplySplit(store, z0z1B->id, z0z1b->id);
  store->attr.sched.loop_axis = z0z1b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = {One};
}

void BroadcastLoadStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  auto x0 = graph.FindNode("x0");
  x0->outputs[0].attr.mem.tensor_id = 0;
  x0->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x0->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x0->outputs[0].attr.mem.position = Position::kPositionGM;
  x0->outputs[0].attr.buf.id = ge::kIdNone;
  x0->outputs[0].attr.que.id = ge::kIdNone;
  x0->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x0->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load0 = graph.FindNode("load0");
  load0->outputs[0].attr.mem.tensor_id = 1;
  load0->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load0->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load0->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load0->outputs[0].attr.buf.id = ge::kIdNone;
  load0->outputs[0].attr.que.id = 0;
  load0->outputs[0].attr.mem.reuse_id = 0;
  load0->outputs[0].attr.que.depth = 2;
  load0->outputs[0].attr.que.buf_num = 2;
  load0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load0->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->outputs[0].attr.mem.tensor_id = 2;
  broadcast->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  broadcast->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  broadcast->outputs[0].attr.buf.id = 1;
  broadcast->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  broadcast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 3;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = 4;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 1;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto mul = graph.FindNode("mul");
  mul->outputs[0].attr.mem.tensor_id = 5;
  mul->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  mul->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  mul->outputs[0].attr.mem.position = Position::kPositionVecOut;
  mul->outputs[0].attr.buf.id = ge::kIdNone;
  mul->outputs[0].attr.que.id = 1;
  mul->outputs[0].attr.mem.reuse_id = 2;
  mul->outputs[0].attr.que.depth = 2;
  mul->outputs[0].attr.que.buf_num = 2;
  mul->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  mul->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 6;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}