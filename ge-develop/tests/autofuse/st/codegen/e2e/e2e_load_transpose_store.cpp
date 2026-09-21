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


/* Transpose[0,1,2] -> [0,2,1] */
void LoadTransposeStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
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
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1 * s2, s2, One};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = in_data_type;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Transpose transpose("transpose");
  graph.AddNode(transpose);
  transpose.x = load.y;
  transpose.y.dtype = out_data_type;
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};  /* TODO sched.axis */
  transpose.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};
  *transpose.y.axis = {z0.id, z2.id, z1.id};
  *transpose.y.repeats = {s0, s2, s1};
  *transpose.y.strides = {s1 * s2, s1, One};

  Store store("store");
  graph.AddNode(store);
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z2.id, z1.id};
  store.y.dtype = out_data_type;
  *store.y.axis = {z0.id, z2.id, z1.id};
  *store.y.repeats = {s0, s2, s1};
  *store.y.strides = {s1 * s2, s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z2.id, z1.id};
  y.y.dtype = out_data_type;
  *y.y.axis = {z0.id, z2.id, z1.id};
}

void LoadTransposeStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
    auto x = graph.FindNode("x");
    x->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

    auto load = graph.FindNode("load");
    load->outputs[0].attr.dtype = in_data_type;
    load->attr.api.compute_type = ComputeType::kComputeLoad;

    /* 为啥不在BeforeAutoFuse上处理 */
    auto transpose = graph.FindNode("transpose");
    transpose->outputs[0].attr.dtype = out_data_type;
/*    transpose->outputs[0].attr.axis = load->outputs[0].attr.axis;
    transpose->outputs[0].attr.repeats = load->outputs[0].attr.repeats;
    transpose->outputs[0].attr.strides = load->outputs[0].attr.strides;*/
    transpose->attr.api.compute_type = ComputeType::kComputeTranspose;

    auto store = graph.FindNode("store");
    store->outputs[0].attr.dtype =  out_data_type;
    store->attr.api.compute_type = ComputeType::kComputeStore;

    auto y = graph.FindNode("y");
    y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadTransposeStore_AfterGetApiInfo(ge::AscGraph &graph) {
    auto x = graph.FindNode("x");
    x->attr.api.type = ApiType::kAPITypeBuffer;
    x->attr.api.unit = ComputeUnit::kUnitNone;

    auto load = graph.FindNode("load");
    load->attr.api.type = ApiType::kAPITypeCompute;
    load->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto transpose = graph.FindNode("transpose");
    transpose->attr.api.type = ApiType::kAPITypeCompute;
    transpose->attr.api.unit = ComputeUnit::kUnitVector;

    auto store = graph.FindNode("store");
    store->attr.api.type = ApiType::kAPITypeCompute;
    store->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto y = graph.FindNode("y");
    y->attr.api.type = ApiType::kAPITypeBuffer;
    y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadTransposeStore_AfterScheduler(ge::AscGraph &graph) {
    auto all_axis = graph.GetAllAxis();
    auto z0 = all_axis[0]->id;
    auto z1 = all_axis[1]->id;
    auto z2 = all_axis[2]->id;

    auto [z0T, z0t] = graph.TileSplit(z0);
    auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

    uint32_t align_size = 64 / sizeof(float);

    /* 输入轴信息 */
    vector<AxisId> vectorized_axis_input{z0t->id, z1, z2};
    vector<ge::Expression> vectorized_strides_input{One, One, One};
    vectorized_strides_input[1] = ge::sym::Align(graph.FindAxis(z2)->size, align_size);
    vectorized_strides_input[0] = graph.FindAxis(z1)->size * ge::sym::Align(graph.FindAxis(z2)->size, align_size);

    /* 输出轴信息 */
    vector<AxisId> vectorized_axis_output{z0t->id, z2, z1};
    vector<ge::Expression> vectorized_strides_output{One, One, One};
    vectorized_strides_output[1] = ge::sym::Align(graph.FindAxis(z1)->size, align_size);
    vectorized_strides_output[0] = ge::sym::Align(graph.FindAxis(z1)->size, align_size) * graph.FindAxis(z2)->size;

    auto x0 = graph.FindNode("x");
    graph.ApplySplit(x0, z0T->id, z0t->id);
    graph.ApplySplit(x0, z0TB->id, z0Tb->id);
    x0->attr.sched.loop_axis = z0Tb->id;
    x0->outputs[0].attr.vectorized_axis = vectorized_axis_input;
    x0->outputs[0].attr.vectorized_strides = vectorized_strides_input;

    auto load0 = graph.FindNode("load");
    graph.ApplySplit(load0, z0T->id, z0t->id);
    graph.ApplySplit(load0, z0TB->id, z0Tb->id);
    load0->attr.sched.loop_axis = z0Tb->id;
    load0->outputs[0].attr.vectorized_axis = vectorized_axis_input;
    load0->outputs[0].attr.vectorized_strides = vectorized_strides_input;

    auto transpose = graph.FindNode("transpose");
    graph.ApplySplit(transpose, z0T->id, z0t->id);
    graph.ApplySplit(transpose, z0TB->id, z0Tb->id);
    transpose->attr.sched.loop_axis = z0Tb->id;
    transpose->outputs[0].attr.vectorized_axis = vectorized_axis_output;
    transpose->outputs[0].attr.vectorized_strides = vectorized_strides_output;

    auto store = graph.FindNode("store");
    graph.ApplySplit(store, z0T->id, z0t->id);
    graph.ApplySplit(store, z0TB->id, z0Tb->id);
    store->attr.sched.loop_axis = z0Tb->id;
    store->attr.sched.loop_axis = z0Tb->id;
    store->outputs[0].attr.vectorized_axis = vectorized_axis_output;
    store->outputs[0].attr.vectorized_strides = vectorized_strides_output;
}

void LoadTransposeStore_AfterQueBufAlloc(ge::AscGraph &graph) {
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

    auto transpose = graph.FindNode("transpose");
    transpose->outputs[0].attr.mem.tensor_id = 2;
    transpose->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
    transpose->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
    transpose->outputs[0].attr.mem.position = Position::kPositionVecOut;
    transpose->outputs[0].attr.buf.id = ge::kIdNone;
    transpose->outputs[0].attr.que.id = 1;
    transpose->outputs[0].attr.mem.reuse_id = 1;
    transpose->outputs[0].attr.que.depth = 2;
    transpose->outputs[0].attr.que.buf_num = 2;
    transpose->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    transpose->outputs[0].attr.opt.merge_scope = ge::kIdNone;

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

