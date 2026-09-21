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

void LoadSwitchScalarSubStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = in_data_type;
  *(x1.y.axis) = {z0.id, z1.id, z2.id};

  Scalar constant("constant");
  graph.AddNode(constant);
  constant.y.dtype = in_data_type;
  constant.ir_attr.SetValue("2.0");

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = in_data_type;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1*s2, s2, One};

  ge::ascir_op::Sub sub("sub");
  graph.AddNode(sub);
  sub.x1 = constant.y;
  sub.x2 = load1.y;
  sub.y.dtype = out_data_type;
  sub.attr.sched.axis = {z0.id, z1.id, z2.id};
  *sub.y.repeats = {s0, s1, s2};
  *sub.y.strides = {s1*s2, s2, One};
  sub.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = sub.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = out_data_type;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1*s2, s2, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = out_data_type;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadSwitchScalarSubStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = in_data_type;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;


  auto sub = graph.FindNode("sub");
  sub->outputs[0].attr.dtype = out_data_type;
  sub->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  sub->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  sub->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  sub->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = out_data_type;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadSwitchScalarSubStore_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto constant = graph.FindNode("constant");
  constant->attr.api.type = ApiType::kAPITypeBuffer;
  constant->attr.api.unit = ComputeUnit::kUnitNone;
  constant->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto sub = graph.FindNode("sub");
  sub->attr.api.type = ApiType::kAPITypeCompute;
  sub->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadSwitchScalarSubStore_AfterScheduler(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T->id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0B, z0b] = graph.BlockSplit(block_axis->id);
  vector<AxisId> vectorized_axis{z1t->id, z2};
  vector<ge::Expression> vectorized_strides{One, One};
  vector<ge::Expression> load2_vectorized_strides{One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  if (size == 0) {
    return;
  }
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  all_axis = graph.GetAllAxis();
  auto m_axis = all_axis[block_axis->id];

  // ApplySplit on load, cast, store
  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load1, block_axis->id, m_axis->from);
  graph.ApplySplit(load1, z0B->id, z0b->id);
  load1->attr.sched.loop_axis = z0b->id;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto sub = graph.FindNode("sub");
  graph.ApplySplit(sub, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(sub, block_axis->id, m_axis->from);
  graph.ApplySplit(sub, z0B->id, z0b->id);
  sub->attr.sched.loop_axis = z0b->id;
  sub->outputs[0].attr.vectorized_axis = vectorized_axis;
  sub->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(store, block_axis->id, m_axis->from);
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadSwitchScalarSubStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto constant = graph.FindNode("constant");
  constant->outputs[0].attr.mem.tensor_id = 1;
  constant->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeInvalid;
  constant->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareInvalid;
  constant->outputs[0].attr.mem.position = Position::kPositionInvalid;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = 2;
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

  auto sub = graph.FindNode("sub");
  sub->outputs[0].attr.mem.tensor_id = 4;
  sub->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  sub->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  sub->outputs[0].attr.mem.position = Position::kPositionVecOut;
  sub->outputs[0].attr.buf.id = ge::kIdNone;
  sub->outputs[0].attr.que.id = 2;
  sub->outputs[0].attr.mem.reuse_id = 1;
  sub->outputs[0].attr.que.depth = 2;
  sub->outputs[0].attr.que.buf_num = 2;
  sub->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  sub->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 5;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}
