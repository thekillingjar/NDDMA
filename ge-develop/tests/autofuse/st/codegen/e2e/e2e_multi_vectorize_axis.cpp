/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "e2e_broadcast.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadMultiVectorizeAxisStore_BeforeAutofuse(ge::AscGraph &graph, bool is_f16) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);


  Data x("x");
  graph.AddNode(x);
  if (is_f16) {
    x.y.dtype = ge::DT_FLOAT16;
  } else {
    x.y.dtype = ge::DT_FLOAT;
  }

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1*s2, s2, One};

  Store store("store");
  graph.AddNode(store);
  store.x = load.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1*s2, s2, One};
  store.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  if (is_f16) {
    y.y.dtype = ge::DT_FLOAT16;
  } else {
    y.y.dtype = ge::DT_FLOAT;
  }

  //graph.SetInputs({x});
  //graph.SetOutputs({y});
}

void LoadMultiVectorizeAxisStore_AfterAutofuse(ge::AscGraph& graph, bool is_f16 = true) {
  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid;
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  if (is_f16) {
    load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  } else {
    load->outputs[0].attr.dtype = ge::DT_FLOAT;
  }

  load->attr.api.compute_type = ComputeType::kComputeLoad;
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  if (is_f16) {
    store->outputs[0].attr.dtype = ge::DT_FLOAT16;
  } else {
    store->outputs[0].attr.dtype = ge::DT_FLOAT;
  }

  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto z0 = load->attr.sched.axis[0];
  auto z1 = load->attr.sched.axis[1];
  auto z2 = load->attr.sched.axis[2];

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }

    graph.ApplySplit(node, z0T->id, z0t->id);
    graph.ApplySplit(node, z0TB->id, z0Tb->id);
  }

  std::vector<AxisId> vectorized_axis{z0t->id, z1, z2};
  vector<ge::Expression> vectorized_strides{One, One, One};
  vectorized_strides[0] = graph.FindAxis(vectorized_axis[vectorized_axis.size() - 0x2])->size *
    graph.FindAxis(vectorized_axis[vectorized_axis.size() - 1])->size;
  vectorized_strides[1] = graph.FindAxis(vectorized_axis[vectorized_axis.size() - 1])->size;
  // Vectorized/Loop axis
  load->attr.sched.loop_axis = z0Tb->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = Position::kPositionGM;
  x->outputs[0].attr.buf.id = ge::kIdNone;
  x->outputs[0].attr.que.id = ge::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

