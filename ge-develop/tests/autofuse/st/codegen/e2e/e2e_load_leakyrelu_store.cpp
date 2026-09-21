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

void LoadLeakyreluStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = in_data_type;
  *(x.y.axis) = {z0.id, z1.id, z2.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = in_data_type;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1*s2, s2, One};

  ge::ascir_op::LeakyRelu leakyrelu("leakyrelu");
  graph.AddNode(leakyrelu);
  leakyrelu.x = load.y;
  leakyrelu.y.dtype = out_data_type;
  leakyrelu.attr.sched.axis = {z0.id, z1.id, z2.id};

  leakyrelu.ir_attr.SetNegative_slope(0.8); // 测试的时候按照0.8去测试

  *leakyrelu.y.repeats = {s0, s1, s2};
  *leakyrelu.y.strides = {s1*s2, s2, One};
  leakyrelu.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = leakyrelu.y;
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

void LoadLeakyreluStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
    auto x = graph.FindNode("x");
    x->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

    auto load = graph.FindNode("load");
    load->outputs[0].attr.dtype = in_data_type;
    load->attr.api.compute_type = ComputeType::kComputeLoad;

    auto leakyrelu = graph.FindNode("leakyrelu");
    leakyrelu->outputs[0].attr.dtype = out_data_type;
    leakyrelu->outputs[0].attr.axis = load->outputs[0].attr.axis;
    leakyrelu->outputs[0].attr.repeats = load->outputs[0].attr.repeats;
    leakyrelu->outputs[0].attr.strides = load->outputs[0].attr.strides;
    leakyrelu->attr.api.compute_type = ComputeType::kComputeElewise;

    auto store = graph.FindNode("store");
    store->outputs[0].attr.dtype =  out_data_type;
    store->attr.api.compute_type = ComputeType::kComputeStore;

    auto y = graph.FindNode("y");
    y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadLeakyreluStore_AfterGetApiInfo(ge::AscGraph &graph) {
    auto x = graph.FindNode("x");
    x->attr.api.type = ApiType::kAPITypeBuffer;
    x->attr.api.unit = ComputeUnit::kUnitNone;

    auto load = graph.FindNode("load");
    load->attr.api.type = ApiType::kAPITypeCompute;
    load->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto leakyrelu = graph.FindNode("leakyrelu");
    leakyrelu->attr.api.type = ApiType::kAPITypeCompute;
    leakyrelu->attr.api.unit = ComputeUnit::kUnitVector;

    auto store = graph.FindNode("store");
    store->attr.api.type = ApiType::kAPITypeCompute;
    store->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto y = graph.FindNode("y");
    y->attr.api.type = ApiType::kAPITypeBuffer;
    y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadLeakyreluStore_AfterScheduler(ge::AscGraph &graph) {
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
    auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
    vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

    all_axis = graph.GetAllAxis();
    auto m_axis = all_axis[block_axis->id];

    // ApplySplit on load, leakyrelu, store
    auto load = graph.FindNode("load");
    graph.ApplySplit(load, z1T->id, z1t->id);
    graph.ApplySchedAxisMerge(load, block_axis->id, m_axis->from);
    graph.ApplySplit(load, z0B->id, z0b->id);
    load->attr.sched.loop_axis = z0b->id;
    load->outputs[0].attr.vectorized_axis = vectorized_axis;
    load->outputs[0].attr.vectorized_strides = vectorized_strides;

    auto leakyrelu = graph.FindNode("leakyrelu");
    graph.ApplySplit(leakyrelu, z1T->id, z1t->id);
    graph.ApplySchedAxisMerge(leakyrelu, block_axis->id, m_axis->from);
    graph.ApplySplit(leakyrelu, z0B->id, z0b->id);
    leakyrelu->attr.sched.loop_axis = z0b->id;
    leakyrelu->outputs[0].attr.vectorized_axis = vectorized_axis;
    leakyrelu->outputs[0].attr.vectorized_strides = vectorized_strides;

    auto store = graph.FindNode("store");
    graph.ApplySplit(store, z1T->id, z1t->id);
    graph.ApplySchedAxisMerge(store, block_axis->id, m_axis->from);
    graph.ApplySplit(store, z0B->id, z0b->id);
    store->attr.sched.loop_axis = z0b->id;
    store->outputs[0].attr.vectorized_axis = vectorized_axis;
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadLeakyreluStore_AfterQueBufAlloc(ge::AscGraph &graph) {
    auto x = graph.FindNode("x");
    x->outputs[0].attr.mem.tensor_id = 0;
    x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
    x->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
    x->outputs[0].attr.mem.position = Position::kPositionGM;
    x->outputs[0].attr.buf.id = ge::kIdNone;
    x->outputs[0].attr.que.id = ge::kIdNone;
    x->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    x->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto load = graph.FindNode("load");
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

    auto leakyrelu = graph.FindNode("leakyrelu");
    leakyrelu->outputs[0].attr.mem.tensor_id = 2;
    leakyrelu->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
    leakyrelu->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
    leakyrelu->outputs[0].attr.mem.position = Position::kPositionVecOut;
    leakyrelu->outputs[0].attr.buf.id = ge::kIdNone;
    leakyrelu->outputs[0].attr.que.id = 1;
    leakyrelu->outputs[0].attr.mem.reuse_id = 1;
    leakyrelu->outputs[0].attr.que.depth = 2;
    leakyrelu->outputs[0].attr.que.buf_num = 2;
    leakyrelu->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    leakyrelu->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto store = graph.FindNode("store");
    store->outputs[0].attr.mem.tensor_id = 3;
    store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
    store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
    store->outputs[0].attr.mem.position = Position::kPositionGM;
    store->outputs[0].attr.buf.id = ge::kIdNone;
    store->outputs[0].attr.que.id = ge::kIdNone;
    store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

