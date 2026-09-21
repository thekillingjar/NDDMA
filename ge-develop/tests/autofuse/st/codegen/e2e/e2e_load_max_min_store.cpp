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
#include "ascir_ops_utils.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadMaxMinStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", ge::sym::Max(s0, One));
  auto z1 = graph.CreateAxis("z1", s1);


  Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;

  Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = data_type;

  Data x3("x3");
  graph.AddNode(x3);
  x3.y.dtype = data_type;

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};

  Load load2("load2");
  graph.AddNode(load2);
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};

  Load load3("load3");
  graph.AddNode(load3);
  load3.x = x3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};

  ge::ascir_op::Maximum max("max");
  graph.AddNode(max);
  max.x1 = load1.y;
  max.x2 = load2.y;
  max.attr.sched.axis = {z0.id, z1.id};
  *max.y.axis = {z0.id, z1.id};
  *max.y.repeats = {s0, s1};
  *max.y.strides = {s1, One};
  max.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  ge::ascir_op::Minimum min("min");
  graph.AddNode(min);
  min.x1 = load3.y;
  min.x2 = max.y;
  min.attr.sched.axis = {z0.id, z1.id};
  *min.y.axis = {z0.id, z1.id};
  *min.y.repeats = {s0, s1};
  *min.y.strides = {s1, One};
  min.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = min.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
}

void LoadMaxMinStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto x3 = graph.FindNode("x3");
  x3->attr.api.compute_type = ComputeType::kComputeInvalid;
  x3->attr.api.type = ApiType::kAPITypeBuffer;
  x3->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = data_type;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = data_type;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.dtype = data_type;
  load3->attr.api.compute_type = ComputeType::kComputeLoad;
  load3->attr.api.type = ApiType::kAPITypeCompute;
  load3->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto max = graph.FindNode("max");
  max->outputs[0].attr.dtype = data_type;
  max->attr.api.compute_type = ComputeType::kComputeElewise;
  max->attr.api.type = ApiType::kAPITypeCompute;
  max->attr.api.unit = ComputeUnit::kUnitVector;

  auto min = graph.FindNode("min");
  min->outputs[0].attr.dtype = data_type;
  min->attr.api.compute_type = ComputeType::kComputeElewise;
  min->attr.api.type = ApiType::kAPITypeCompute;
  min->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto z0 = load1->attr.sched.axis[0];
  auto z1 = load1->attr.sched.axis[1];

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto [z1T, z1t] = graph.TileSplit(z1);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }

    graph.ApplySplit(node, z0T->id, z0t->id);
    graph.ApplySplit(node, z0TB->id, z0Tb->id);
    graph.ApplySplit(node, z1T->id, z1t->id);
    graph.ApplyReorder(node, {z0TB->id, z0Tb->id, z1T->id, z0t->id, z1t->id});
  }

  vector<ascir::AxisId> vectorized_axis = {z0t->id, z1t->id};
  vector<ge::Expression> vectorized_strides{graph.FindAxis(z1t->id)->size, One};
  // Vectorized/Loop axis
  load1->attr.sched.loop_axis = z1T->id;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = vectorized_strides;

  load2->attr.sched.loop_axis = z1T->id;
  load2->outputs[0].attr.vectorized_axis = vectorized_axis;
  load2->outputs[0].attr.vectorized_strides = vectorized_strides;

  load3->attr.sched.loop_axis = z1T->id;
  load3->outputs[0].attr.vectorized_axis = vectorized_axis;
  load3->outputs[0].attr.vectorized_strides = vectorized_strides;

  max->attr.sched.loop_axis = z1T->id;
  max->outputs[0].attr.vectorized_axis = vectorized_axis;
  max->outputs[0].attr.vectorized_strides = vectorized_strides;

  min->attr.sched.loop_axis = z1T->id;
  min->outputs[0].attr.vectorized_axis = vectorized_axis;
  min->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z1T->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x3->outputs[0].attr.mem.tensor_id = 2;
  x3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x3->outputs[0].attr.mem.position = Position::kPositionGM;
  x3->outputs[0].attr.buf.id = ge::kIdNone;
  x3->outputs[0].attr.que.id = ge::kIdNone;
  x3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

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

  max->outputs[0].attr.mem.tensor_id = 6;
  max->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  max->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  max->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  max->outputs[0].attr.buf.id = 0;
  max->outputs[0].attr.que.id = ge::kIdNone;
  max->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  max->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  min->outputs[0].attr.mem.tensor_id = 7;
  min->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  min->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  min->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  min->outputs[0].attr.buf.id = 1;
  min->outputs[0].attr.que.id = ge::kIdNone;
  min->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  min->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 8;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void Construct_Enable_Cache_Max_Struct(ge::AscGraph &graph) {
  static ge::Expression Zero = ge::Symbol(0);
  static ge::Expression One = ge::Symbol(1);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto axis = {z0.id, z1.id, z2.id, z3.id};

  Data data_0("data_0", graph);
  data_0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data_0.y.dtype = ge::DT_FLOAT16;
  data_0.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = data_0.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Broadcast b0_broadcast("b0_broadcast");
  b0_broadcast.x = b0_load.y;
  b0_broadcast.attr.sched.axis = axis;
  b0_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b0_broadcast.y.dtype = ge::DT_FLOAT16;
  *b0_broadcast.y.axis = axis;
  *b0_broadcast.y.repeats = {s0, s1, s2, s3};
  *b0_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data_1("data_1", graph);
  data_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data_1.y.dtype = ge::DT_FLOAT16;
  data_1.ir_attr.SetIndex(1);

  Load b1_load("b1_load");
  b1_load.x = data_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0, s1, One, s3};
  *b1_load.y.strides = {s1 * s3, s3, Zero, One};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b1_load.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0, s1, s2, s3};
  *b1_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add b0_add("b0_add");
  b0_add.x1 = b0_broadcast.y;
  b0_add.x2 = b1_broadcast.y;
  b0_add.attr.sched.axis = axis;
  b0_add.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_add.y.dtype = ge::DT_FLOAT16;
  *b0_add.y.axis = axis;
  *b0_add.y.repeats = {s0, s1, s2, s3};
  *b0_add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_add.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0, One, One, s3};
  *b0_max.y.strides = {s3, Zero, Zero, One};

  Store b0_store("b0_store");
  b0_store.x = b0_max.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0, One, One, s3};
  *b0_store.y.strides = {s3, Zero, Zero, One};

  Output output_0("output_0");
  output_0.x = b0_store.y;
  output_0.attr.api.compute_type = ComputeType::kComputeInvalid;
  output_0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output_0.y.dtype = ge::DT_FLOAT;
  output_0.ir_attr.SetIndex(0);
}

void Construct_All_Reduce_Max_Struct(ge::AscGraph &graph) {
  static ge::Expression Zero = ge::Symbol(0);
  static ge::Expression One = ge::Symbol(1);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto axis = {z0.id, z1.id, z2.id, z3.id};

  Data data_0("data_0", graph);
  data_0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data_0.y.dtype = ge::DT_FLOAT16;
  data_0.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = data_0.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Broadcast b0_broadcast("b0_broadcast");
  b0_broadcast.x = b0_load.y;
  b0_broadcast.attr.sched.axis = axis;
  b0_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b0_broadcast.y.dtype = ge::DT_FLOAT16;
  *b0_broadcast.y.axis = axis;
  *b0_broadcast.y.repeats = {s0, s1, s2, s3};
  *b0_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data_1("data_1", graph);
  data_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data_1.y.dtype = ge::DT_FLOAT16;
  data_1.ir_attr.SetIndex(1);

  Load b1_load("b1_load");
  b1_load.x = data_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0, s1, One, s3};
  *b1_load.y.strides = {s1 * s3, s3, Zero, One};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b1_load.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0, s1, s2, s3};
  *b1_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add b0_add("b0_add");
  b0_add.x1 = b0_broadcast.y;
  b0_add.x2 = b1_broadcast.y;
  b0_add.attr.sched.axis = axis;
  b0_add.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_add.y.dtype = ge::DT_FLOAT16;
  *b0_add.y.axis = axis;
  *b0_add.y.repeats = {s0, s1, s2, s3};
  *b0_add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_add.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {One, One, One, One};
  *b0_max.y.strides = {Zero, Zero, Zero, Zero};

  Store b0_store("b0_store");
  b0_store.x = b0_max.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {One, One, One, One};
  *b0_store.y.strides = {Zero, Zero, Zero, Zero};

  Output output_0("output_0");
  output_0.x = b0_store.y;
  output_0.attr.api.compute_type = ComputeType::kComputeInvalid;
  output_0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output_0.y.dtype = ge::DT_FLOAT;
  output_0.ir_attr.SetIndex(0);
}

void Construct_Enable_Cache_Sum_Struct(ge::AscGraph &graph) {
  static ge::Expression Zero = ge::Symbol(0);
  static ge::Expression One = ge::Symbol(1);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto axis = {z0.id, z1.id, z2.id, z3.id};

  Data data_0("data_0", graph);
  data_0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data_0.y.dtype = ge::DT_FLOAT16;
  data_0.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = data_0.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Broadcast b0_broadcast("b0_broadcast");
  b0_broadcast.x = b0_load.y;
  b0_broadcast.attr.sched.axis = axis;
  b0_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b0_broadcast.y.dtype = ge::DT_FLOAT16;
  *b0_broadcast.y.axis = axis;
  *b0_broadcast.y.repeats = {s0, s1, s2, s3};
  *b0_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data_1("data_1", graph);
  data_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data_1.y.dtype = ge::DT_FLOAT16;
  data_1.ir_attr.SetIndex(1);

  Load b1_load("b1_load");
  b1_load.x = data_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0, s1, One, s3};
  *b1_load.y.strides = {s1 * s3, s3, Zero, One};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b1_load.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0, s1, s2, s3};
  *b1_broadcast.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add b0_add("b0_add");
  b0_add.x1 = b0_broadcast.y;
  b0_add.x2 = b1_broadcast.y;
  b0_add.attr.sched.axis = axis;
  b0_add.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_add.y.dtype = ge::DT_FLOAT16;
  *b0_add.y.axis = axis;
  *b0_add.y.repeats = {s0, s1, s2, s3};
  *b0_add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Sum b0_sum("b0_sum");
  b0_sum.x = b0_add.y;
  b0_sum.attr.sched.axis = axis;
  b0_sum.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_sum.y.dtype = ge::DT_FLOAT16;
  *b0_sum.y.axis = axis;
  *b0_sum.y.repeats = {s0, One, One, s3};
  *b0_sum.y.strides = {s3, Zero, Zero, One};

  Store b0_store("b0_store");
  b0_store.x = b0_sum.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0, One, One, s3};
  *b0_store.y.strides = {s3, Zero, Zero, One};

  Output output_0("output_0");
  output_0.x = b0_store.y;
  output_0.attr.api.compute_type = ComputeType::kComputeInvalid;
  output_0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output_0.y.dtype = ge::DT_FLOAT;
  output_0.ir_attr.SetIndex(0);
}
