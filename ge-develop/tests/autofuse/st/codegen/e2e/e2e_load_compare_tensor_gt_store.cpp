/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "e2e_load_compare_tensor_gt_store.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadCompareTensorGtStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type) {
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

  Gt gt("gt");
  graph.AddNode(gt);
  gt.x1 = load1.y;
  gt.x2 = load2.y;
  gt.attr.sched.axis = {z0.id};
  gt.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = gt.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
}

void LoadCompareTensorGtStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type) {
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

  auto gt = graph.FindNode("gt");
  gt->outputs[0].attr.dtype = ge::DT_INT8;
  gt->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  gt->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  gt->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  gt->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = ge::DT_INT8;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->outputs[0].attr.dtype = ge::DT_INT8;
}

void LoadCompareTensorGtStore_AfterGetApiInfo(ge::AscGraph &graph) {
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

  auto gt = graph.FindNode("gt");
  gt->attr.api.type = ApiType::kAPITypeCompute;
  gt->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadCompareTensorGtStore_AfterScheduler(ge::AscGraph &graph) {
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

  auto gt = graph.FindNode("gt");
  graph.ApplySplit(gt, z0T->id, z0t->id);
  graph.ApplySplit(gt, z0TB->id, z0Tb->id);
  gt->attr.sched.loop_axis = z0Tb->id;
  gt->outputs[0].attr.vectorized_axis = {z0t->id,};
  gt->outputs[0].attr.vectorized_strides = {One,};

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0T->id, z0t->id);
  graph.ApplySplit(store, z0TB->id, z0Tb->id);
  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0t->id,};
  store->outputs[0].attr.vectorized_strides = {One,};
  cout << utils::DebugHintGraphStr(graph) << endl;
}

void LoadCompareTensorGtStore_AfterQueBufAlloc(ge::AscGraph &graph) {
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

  auto gt = graph.FindNode("gt");
  gt->outputs[0].attr.mem.tensor_id = tensor_id++;
  gt->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gt->outputs[0].attr.mem.position = Position::kPositionVecOut;
  gt->outputs[0].attr.buf.id = ge::kIdNone;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.mem.reuse_id = 2;
  gt->outputs[0].attr.que.depth = 1;
  gt->outputs[0].attr.que.buf_num = 1;
  gt->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gt->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensor_id++;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
}

void LoadCompareTensorGtStore_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs,
                                            ge::DataType data_type) {
  LoadCompareTensorGtStore_BeforeAutofuse(graph, data_type);
  LoadCompareTensorGtStore_AfterInferOutput(graph, data_type);

  impl_graphs.push_back(ge::AscGraph("load_compare_half_tensor_gt_store_general_0_nil_0_nil"));
  impl_graphs[0].CopyFrom(graph);
  LoadCompareTensorGtStore_AfterGetApiInfo(impl_graphs[0]);
  LoadCompareTensorGtStore_AfterScheduler(impl_graphs[0]);
  LoadCompareTensorGtStore_AfterQueBufAlloc(impl_graphs[0]);
}