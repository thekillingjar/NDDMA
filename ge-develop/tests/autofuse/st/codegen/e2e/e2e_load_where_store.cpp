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

void LoadWhereStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = ge::DT_UINT8;
  *(x1.y.axis) = {z0.id, z1.id, z2.id};

  Data x2("x2");
  graph.AddNode(x2);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  x2.y.dtype = ge::DT_FLOAT;
  *(x2.y.axis) = {z0.id, z1.id, z2.id};

  Data x3("x3");
  graph.AddNode(x3);
  x3.attr.sched.axis = {z0.id, z1.id, z2.id};
  x3.y.dtype = ge::DT_FLOAT;
  *(x3.y.axis) = {z0.id, z1.id, z2.id};

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1*s2, s2, One};

  Load load2("load2");
  graph.AddNode(load2);
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {s0, s1, s2};
  *load2.y.strides = {s1*s2, s2, One};

  Load load3("load3");
  graph.AddNode(load3);
  load3.x = x3.y;
  load3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load3.y.axis = {z0.id, z1.id, z2.id};
  *load3.y.repeats = {s0, s1, s2};
  *load3.y.strides = {s1*s2, s2, One};

  ge::ascir_op::Where where("where");
  graph.AddNode(where);
  where.x1 = load1.y;
  where.x2 = load2.y;
  where.x3 = load3.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1*s2, s2, One};
  where.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = where.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1*s2, s2, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadWhereStore_AfterInferOutput(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto x3 = graph.FindNode("x3");
  x3->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = ge::DT_UINT8;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.dtype = ge::DT_FLOAT;
  load3->attr.api.compute_type = ComputeType::kComputeLoad;

  auto where = graph.FindNode("where");
  where->outputs[0].attr.dtype =(ge::DataType)load2->outputs[0].attr.dtype;
  where->outputs[0].attr.axis = load2->outputs[0].attr.axis;
  where->outputs[0].attr.repeats = load2->outputs[0].attr.repeats;
  where->outputs[0].attr.strides = load2->outputs[0].attr.strides;
  where->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = (ge::DataType)where->outputs[0].attr.dtype;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadWhereStore_AfterGetApiInfo(ge::AscGraph &graph) {
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

  auto where = graph.FindNode("where");
  where->attr.api.type = ApiType::kAPITypeCompute;
  where->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadWhereStore_AfterScheduler(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T->id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0B, z0b] = graph.BlockSplit(block_axis->id);
  vector<AxisId> vectorized_axis{z1t->id, z2};
  vector<ge::Expression> vectorized_strides{Zero, One};
  vector<ge::Expression> vectorized_strides1{Zero, One};
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 8);
  vectorized_strides1[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32);

  // ApplySplit on load, abs, store
  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load1, block_axis->id);
  graph.ApplySplit(load1, z0B->id, z0b->id);
  load1->attr.sched.loop_axis = z0b->id;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = vectorized_strides1;

  auto load2 = graph.FindNode("load2");
  graph.ApplySplit(load2, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load2, block_axis->id);
  graph.ApplySplit(load2, z0B->id, z0b->id);
  load2->attr.sched.loop_axis = z0b->id;
  load2->outputs[0].attr.vectorized_axis = vectorized_axis;
  load2->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto load3 = graph.FindNode("load3");
  graph.ApplySplit(load3, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load3, block_axis->id);
  graph.ApplySplit(load3, z0B->id, z0b->id);
  load3->attr.sched.loop_axis = z0b->id;
  load3->outputs[0].attr.vectorized_axis = vectorized_axis;
  load3->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto where = graph.FindNode("where");
  graph.ApplySplit(where, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(where, block_axis->id);
  graph.ApplySplit(where, z0B->id, z0b->id);
  where->attr.sched.loop_axis = z0b->id;
  where->outputs[0].attr.vectorized_axis = vectorized_axis;
  where->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(store, block_axis->id);
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadWhereStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x3 = graph.FindNode("x3");
  x3->outputs[0].attr.mem.tensor_id = 2;
  x3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x3->outputs[0].attr.mem.position = Position::kPositionGM;
  x3->outputs[0].attr.buf.id = ge::kIdNone;
  x3->outputs[0].attr.que.id = ge::kIdNone;
  x3->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = ge::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 1;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.mem.tensor_id = 5;
  load3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load3->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load3->outputs[0].attr.buf.id = ge::kIdNone;
  load3->outputs[0].attr.que.id = 2;
  load3->outputs[0].attr.mem.reuse_id = 2;
  load3->outputs[0].attr.que.depth = 2;
  load3->outputs[0].attr.que.buf_num = 2;
  load3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto where = graph.FindNode("where");
  where->outputs[0].attr.mem.tensor_id = 6;
  where->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  where->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  where->outputs[0].attr.mem.position = Position::kPositionVecOut;
  where->outputs[0].attr.buf.id = ge::kIdNone;
  where->outputs[0].attr.que.id = 3;
  where->outputs[0].attr.mem.reuse_id = 3;
  where->outputs[0].attr.que.depth = 2;
  where->outputs[0].attr.que.buf_num = 2;
  where->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  where->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 7;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}
