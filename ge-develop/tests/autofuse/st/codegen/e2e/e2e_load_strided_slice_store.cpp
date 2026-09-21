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

#include "e2e_load_strided_slice_store.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadStridedSliceStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1+s2);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x0("x0");
  graph.AddNode(x0);
  x0.attr.sched.axis = {z0.id, z1.id};
  x0.y.dtype = ge::DT_FLOAT;
  *x0.y.axis = {z0.id, z1.id};
  *x0.y.repeats = {s0, s1+s2};
  *x0.y.strides = {s1+s2, One};

  Load load0("load0");
  graph.AddNode(load0);
  load0.x = x0.y;
  load0.attr.sched.axis = {z0.id, z2.id};
  load0.ir_attr.SetOffset(s1);
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z2.id};
  *load0.y.repeats = {s0, s2};
  *load0.y.strides = {s1+s2, One};

  Store store("store");
  graph.AddNode(store);
  store.x = load0.y;
  store.attr.sched.axis = {z0.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z2.id};
  *store.y.repeats = {s0, s2};
  *store.y.strides = {s2, One};
  store.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z2.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, z2.id};
}

void LoadStridedSliceStore_AfterInferOutput(ge::AscGraph &graph) {
  auto x0 = graph.FindNode("x0");
  x0->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load0 = graph.FindNode("load0");
  load0->outputs[0].attr.dtype = ge::DT_FLOAT;
  load0->attr.api.compute_type = ComputeType::kComputeLoad;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = (ge::DataType)load0->outputs[0].attr.dtype;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadStridedSliceStore_AfterGetApiInfo(ge::AscGraph &graph) {
    auto x0 = graph.FindNode("x0");
    x0->attr.api.type = ApiType::kAPITypeBuffer;
    x0->attr.api.unit = ComputeUnit::kUnitNone;

    auto load0 = graph.FindNode("load0");
    load0->attr.api.type = ApiType::kAPITypeCompute;
    load0->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto store = graph.FindNode("store");
    store->attr.api.type = ApiType::kAPITypeCompute;
    store->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto y = graph.FindNode("y");
    y->attr.api.type = ApiType::kAPITypeBuffer;
    y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadStridedSliceStore_AfterScheduler(ge::AscGraph &graph) {
    auto all_axis = graph.GetAllAxis();
    auto z0 = all_axis[0]->id;
    auto z1 = all_axis[1]->id;
    auto z2 = all_axis[2]->id;

    auto [z0T, z0t] = graph.TileSplit(z0);
    auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
    vector<AxisId> vectorized_axis{z0t->id, z2};
    vector<ge::Expression> vectorized_strides{One, One};
    uint32_t align_size = 32 / sizeof(float);
    vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 8);
    vector<ge::Expression> vectorized_strides_x1{ge::sym::Align(graph.FindAxis(z1)->size, 8), One};

    // ApplySplit on x, load, abs, store
    auto x0 = graph.FindNode("x0");
    graph.ApplySplit(x0, z0T->id, z0t->id);
    graph.ApplySplit(x0, z0TB->id, z0Tb->id);
    x0->attr.sched.loop_axis = z0Tb->id;
    x0->outputs[0].attr.vectorized_axis = {z0t->id, z1};
    x0->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto load0 = graph.FindNode("load0");
    graph.ApplySplit(load0, z0T->id, z0t->id);
    graph.ApplySplit(load0, z0TB->id, z0Tb->id);
    load0->attr.sched.loop_axis = z0Tb->id;
    load0->outputs[0].attr.vectorized_axis = vectorized_axis;
    load0->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto store = graph.FindNode("store");
    graph.ApplySplit(store, z0T->id, z0t->id);
    graph.ApplySplit(store, z0TB->id, z0Tb->id);
    store->attr.sched.loop_axis = z0Tb->id;
    store->outputs[0].attr.vectorized_axis = vectorized_axis;
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadStridedSliceStore_AfterQueBufAlloc(ge::AscGraph &graph) {
    auto x0 = graph.FindNode("x0");
    x0->outputs[0].attr.mem.tensor_id = 0;
    x0->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
    x0->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
    x0->outputs[0].attr.mem.position = Position::kPositionGM;
    x0->outputs[0].attr.buf.id = ge::kIdNone;
    x0->outputs[0].attr.que.id = ge::kIdNone;
    x0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    x0->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto load0 = graph.FindNode("load0");
    load0->outputs[0].attr.mem.tensor_id = 2;
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