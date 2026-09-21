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
#include "ascgraph_info_complete.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#define private public
#include "optimize.h"
#undef private
#include "e2e_common.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadBroadcastStore_BeforeAutofuse(ge::AscGraph &graph, int broad_axis, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);


  Data x("x");
  graph.AddNode(x);
  x.y.dtype = data_type;
  x.ir_attr.SetIndex(0);

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  if (broad_axis == 0) {
    *load.y.repeats = {s0, One};
    *load.y.strides = {One, Zero};
  } else if (broad_axis == 1) {
    *load.y.repeats = {One, s1};
    *load.y.strides = {Zero, One};
  }

  Broadcast broadcast("broadcast");
  graph.AddNode(broadcast);
  broadcast.x = load.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};
  broadcast.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = data_type;

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
  y.ir_attr.SetIndex(0);

  //graph.SetInputs({x});
  //graph.SetOutputs({y});
}

void LoadBroadcastStore_AfterAutofuse(ge::AscGraph& graph, int broad_axis, ge::DataType data_type) {
  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid;
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = data_type;

  load->attr.api.compute_type = ComputeType::kComputeLoad;
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast->outputs[0].attr.dtype = data_type;

  broadcast->attr.api.type = ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = data_type;

  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto z0 = load->attr.sched.axis[0];
  auto z1 = load->attr.sched.axis[1];
  vector<ge::Expression> vectorized_strides{One, One};

  if (broad_axis == 0) {
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

    // Vectorized/Loop axis
    load->attr.sched.loop_axis = z1T->id;
    load->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
    load->outputs[0].attr.vectorized_strides = {One, Zero};

    broadcast->attr.sched.loop_axis = z1T->id;
    broadcast->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
    auto size = ge::GetSizeByDataType(data_type);
    vectorized_strides[0] = ge::sym::Align(graph.FindAxis(z1t->id)->size, 32 / size);
    broadcast->outputs[0].attr.vectorized_strides = vectorized_strides;

    store->attr.sched.loop_axis = z1T->id;
    store->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
  } else if (broad_axis == 1) {
    auto [z0T, z0t] = graph.TileSplit(z0);
    for (auto node : graph.GetAllNodes()) {
      if (IsOps<Data>(node) || IsOps<Output>(node)) {
        continue;
      }
      graph.ApplySplit(node, z0T->id, z0t->id);
    }
    // Vectorized/Loop axis
    load->attr.sched.loop_axis = z0T->id;
    load->outputs[0].attr.vectorized_axis = {z0t->id, z1};
    load->outputs[0].attr.vectorized_strides = {Zero, One};

    broadcast->attr.sched.loop_axis = z0T->id;
    broadcast->outputs[0].attr.vectorized_axis = {z0t->id, z1};
    auto size = ge::GetSizeByDataType(data_type);
    vectorized_strides[0] = ge::sym::Align(graph.FindAxis(z1)->size, 32 / size);
    broadcast->outputs[0].attr.vectorized_strides = vectorized_strides;

    store->attr.sched.loop_axis = z0T->id;
    store->outputs[0].attr.vectorized_axis = {z0t->id, z1};
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

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

  broadcast->outputs[0].attr.mem.tensor_id = 2;
  broadcast->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = Position::kPositionVecOut;
  broadcast->outputs[0].attr.buf.id = ge::kIdNone;
  broadcast->outputs[0].attr.que.id = 1;
  broadcast->outputs[0].attr.mem.reuse_id = 1;
  broadcast->outputs[0].attr.que.depth = 2;
  broadcast->outputs[0].attr.que.buf_num = 2;
  broadcast->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  broadcast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware =  MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void ConstructGraph(ge::AscGraph& graph, std::vector<ge::AscGraph> &impl_graphs) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0).id;
  auto z1 = graph.CreateAxis("z1", s1).id;
  auto z2 = graph.CreateAxis("z2", s2).id;

  int order = 0;

  Data x_op("x");
  graph.AddNode(x_op);
  x_op.y.dtype = ge::DT_FLOAT16;

  Load load_op("load");
  graph.AddNode(load_op);
  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0, z1, z2};
  *load_op.y.axis = {z0, z1, z2};
  *load_op.y.repeats = {One, s1, s2};
  *load_op.y.strides = {Zero, s2, One};

  Broadcast broadcast_op("broadcast");
  graph.AddNode(broadcast_op);
  broadcast_op.x = load_op.y;
  broadcast_op.attr.sched.axis = {z0, z1, z2};
  *broadcast_op.y.axis = {z0, z1, z2};
  *broadcast_op.y.repeats = {s0, s1, s2};
  *broadcast_op.y.strides = {s1*s2, s2, One};

  Store store_op("store");
  graph.AddNode(store_op);
  store_op.x = broadcast_op.y;
  store_op.attr.sched.axis = {z0, z1, z2};
  *store_op.y.axis = {z0, z1, z2};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1*s2, s2, One};

  Output y_op("y");
  graph.AddNode(y_op);
  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;

  //graph.SetInputs({x});
  //graph.SetOutputs({y});
  AssignDefaultIoIndex(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kAscGraph});
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  impl_graphs.push_back(ge::AscGraph("broadcast_merge_axis_general_0_nil_0_nil"));
  impl_graphs[0].CopyFrom(graph);

  optimize::AscGraphInfoComplete::CompleteApiInfo(impl_graphs[0]);

  auto z0z1 = impl_graphs[0].MergeAxis({z0, z1});
  auto [z0z1B, z0z1b] = impl_graphs[0].BlockSplit(z0z1->id);

  auto all_axis = impl_graphs[0].GetAllAxis();
  auto m_axis = all_axis[z0z1->id];


  auto data = impl_graphs[0].FindNode("x");
  data->attr.api.unit = ComputeUnit::kUnitNone;

  auto load = impl_graphs[0].FindNode("load");
  impl_graphs[0].ApplySchedAxisMerge(load, z0z1->id, m_axis->from);
  impl_graphs[0].ApplySplit(load, z0z1B->id, z0z1b->id);
  load->attr.sched.loop_axis = z0z1b->id;
  load->outputs[0].attr.vectorized_axis = {z2};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.opt.reuse_id = 0;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto broadcast = impl_graphs[0].FindNode("broadcast");
  impl_graphs[0].ApplySchedAxisMerge(broadcast, z0z1->id, m_axis->from);
  impl_graphs[0].ApplySplit(broadcast, z0z1B->id, z0z1b->id);
  broadcast->attr.sched.loop_axis = z0z1b->id;
  broadcast->outputs[0].attr.vectorized_axis = {z2};
  broadcast->outputs[0].attr.vectorized_strides = {One};
  broadcast->outputs[0].attr.opt.reuse_id = 0;
  broadcast->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = impl_graphs[0].FindNode("store");
  impl_graphs[0].ApplySchedAxisMerge(store, z0z1->id, m_axis->from);
  impl_graphs[0].ApplySplit(store, z0z1B->id, z0z1b->id);
  store->attr.sched.loop_axis = z0z1b->id;
  store->outputs[0].attr.vectorized_axis = {z2};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.opt.reuse_id = 0;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  optimizer.BufQueAlloc(graph, impl_graphs);

  load = impl_graphs[0].FindNode("load");
  broadcast = impl_graphs[0].FindNode("broadcast");
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  broadcast->outputs[0].attr.que.id = 1;
  broadcast->outputs[0].attr.mem.reuse_id = 1;
}

void ConstructMultiAxisGraph(ge::AscGraph& graph, std::vector<ge::AscGraph> &impl_graphs, std::vector<bool> is_broadcast_axis,
                            std::string kernel_name) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  std::vector<ge::Expression> all_size_var = {s0, s1, s2, s3, s4};

  auto z0 = graph.CreateAxis("z0", s0).id;
  auto z1 = graph.CreateAxis("z1", s1).id;
  auto z2 = graph.CreateAxis("z2", s2).id;
  auto z3 = graph.CreateAxis("z3", s3).id;
  auto z4 = graph.CreateAxis("z4", s4).id;

  int order = 0;

  Data x_op("x");
  graph.AddNode(x_op);
  x_op.y.dtype = ge::DT_FLOAT;

  Load load_op("load");
  graph.AddNode(load_op);
  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0, z1, z2, z3, z4};
  *load_op.y.axis = {z0, z1, z2, z3, z4};
  /*
  *根据is_broadcast_axis生成广播前load的轴信息
  * 如 [z0 z1 z2 z4] broadcast to [z0 z1 z2 z3 z4] (s0*s1*s2*s4 -> s0*s1*s2*s3*s4)
  *load_op.y.repeats ={s0, s1, s2, One, s4};
  *load_op.y.strides = {s1*s2*s4, s2*s4, s4, Zero, One};
  * 如 [z0 z3 z4] broadcast to [z0 z1 z2 z3 z4] (s0*s3*s4 -> s0*s1*s2*s3*s4)
  *load_op.y.repeats ={s0, One, One, s3, s4};
  *load_op.y.strides = {s3*s4, Zero, Zero, s4, One};
  * 如 [z0 z1 z2] broadcast to [z0 z1 z2 z3 z4] (s0*s1*s2 -> s0*s1*s2*s3*s4)
  *load_op.y.repeats ={s0, s1, s2, One, One};
  *load_op.y.strides = {s1*s2, s2, One, Zero, Zero};
  */
  ge::Expression load_stride = One;
  for (int i = is_broadcast_axis.size() - 1; i >= 0; --i) {
    if (is_broadcast_axis[i]) {
      load_op.y.repeats->insert(load_op.y.repeats->begin(), One);
      load_op.y.strides->insert(load_op.y.strides->begin(), Zero);
    } else {
      load_op.y.repeats->insert(load_op.y.repeats->begin(), all_size_var[i]);
      load_op.y.strides->insert(load_op.y.strides->begin(), load_stride);
      load_stride = load_stride * all_size_var[i];
    }
  }

  Broadcast broadcast_op("broadcast");
  graph.AddNode(broadcast_op);
  broadcast_op.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};
  broadcast_op.x = load_op.y;
  broadcast_op.attr.sched.axis = {z0, z1, z2, z3, z4};
  *broadcast_op.y.axis = {z0, z1, z2, z3, z4};
  *broadcast_op.y.repeats = {s0, s1, s2, s3, s4};
  *broadcast_op.y.strides = {s1*s2*s3*s4, s2*s3*s4, s3*s4, s4, One};

  Store store_op("store");
  graph.AddNode(store_op);
  store_op.x = broadcast_op.y;
  store_op.attr.sched.axis = {z0, z1, z2, z3, z4};
  *store_op.y.axis = {z0, z1, z2, z3, z4};
  *store_op.y.repeats = {s0, s1, s2, s3, s4};
  *store_op.y.strides = {s1*s2*s3*s4, s2*s3*s4, s3*s4, s4, One};

  Output y_op("y");
  graph.AddNode(y_op);
  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT;
  AssignDefaultIoIndex(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kAscGraph});
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  impl_graphs.push_back(ge::AscGraph(kernel_name.c_str()));
  impl_graphs[0].CopyFrom(graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(impl_graphs[0]);

  ge::Expression align_s4 = ge::sym::Align(s4, (32/sizeof(float32_t)));

  // broadcast的vectorized_strides
  vector<ge::Expression> not_align_vectorized_strides{s2*s3*s4, s3*s4, s4, One};
  vector<ge::Expression> vectorized_strides{s2*s3*align_s4, s3*align_s4, align_s4, One};

  /*
  * 向量化后四根轴
  * 对应上述load示例分别为
  * load_vectorized_strides{s2*s4, s4, Zero, One};
  * load_vectorized_strides{Zero, Zero, s4, One};
  * load_vectorized_strides{s2, One, Zero, Zero};
  */
  vector<ge::Expression> load_vectorized_strides;
  load_stride = One;
  bool first = true;
  for (int i = is_broadcast_axis.size() - 1; i > 0; --i) {
    // is_broadcast_axis包含全部轴，向量化只做后四根轴，此处只循环到第二根轴，即不遍历第0轴
    if (is_broadcast_axis[i]) {
      load_vectorized_strides.insert(load_vectorized_strides.begin(), Zero);
    } else {
      // 只对尾轴做对齐, 如果尾轴是broadcast，则不做对齐
      if (first && (i == (is_broadcast_axis.size() - 1))) {
        first = false;
        load_vectorized_strides.insert(load_vectorized_strides.begin(), load_stride);
        ge::Expression align_size = ge::sym::Align(all_size_var[i], (32/sizeof(float32_t)));
        load_stride = load_stride * align_size;
      } else {
        load_vectorized_strides.insert(load_vectorized_strides.begin(), load_stride);
        load_stride = load_stride * all_size_var[i];
      }
    }
  }

  for (auto node : impl_graphs[0].GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    node->attr.sched.loop_axis = z0;
    node->outputs[0].attr.vectorized_axis = {z1, z2, z3, z4};

    if (IsOps<Load>(node)) {
      node->outputs[0].attr.vectorized_strides = load_vectorized_strides;
    } else {
      node->outputs[0].attr.vectorized_strides = vectorized_strides;
    } 
  }
  optimizer.BufQueAlloc(graph, impl_graphs);

  auto load = impl_graphs[0].FindNode("load");
  auto broadcast = impl_graphs[0].FindNode("broadcast");
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  broadcast->outputs[0].attr.que.id = 1;
  broadcast->outputs[0].attr.mem.reuse_id = 1;
}