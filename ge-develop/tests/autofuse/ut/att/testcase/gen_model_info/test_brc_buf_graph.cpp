/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test_fa_ascir_graph.h"
#include "base/base_types.h"

namespace att {
using namespace att;
using namespace ge::ascir_op;
void BrcBufBeforeAutoFuse1(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto Z0 = ge::Symbol("Z0");
  auto Z1 = ge::Symbol("Z1");
  auto Z2 = ge::Symbol("Z2");

  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto normalAxis = {z0.id, z1.id, z2.id};
  std::initializer_list<Expr> normalRepeat = {Z0, Z1, Z2};
  std::initializer_list<Expr> normalStride = {Z1 * Z2, Z2, ONE};

  std::initializer_list<Expr> beforeBrcRepeat = {Z0, ONE, Z2};
  std::initializer_list<Expr> beforeBrcStride = {Z2, ZERO, ONE};

  std::initializer_list<Expr> reduceResRepeat = {Z0, ONE, Z2};
  std::initializer_list<Expr> reduceResStride = {Z2, ZERO, ONE};

  int32_t exec_order = 0;
  Data input_data("input_data", graph);
  input_data.attr.sched.exec_order = exec_order++;
  input_data.attr.sched.axis = normalAxis;
  input_data.y.dtype = ge::DT_FLOAT16;
  *input_data.y.axis = normalAxis;
  *input_data.y.repeats = beforeBrcRepeat;
  *input_data.y.strides = beforeBrcStride;

  Load load("load");
  load.x = input_data.y;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = normalAxis;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = normalAxis;
  *load.y.repeats = beforeBrcRepeat;
  *load.y.strides = beforeBrcStride;

  Cast cast0("cast0");
  cast0.x = load.y;
  cast0.attr.sched.exec_order = exec_order++;
  cast0.attr.sched.axis = normalAxis;
  cast0.y.dtype = ge::DT_FLOAT;
  *cast0.y.axis = normalAxis;
  *cast0.y.repeats = beforeBrcRepeat;
  *cast0.y.strides = beforeBrcStride;

  Broadcast broadcast("broadcast");
  broadcast.x = cast0.y;
  broadcast.attr.sched.exec_order = exec_order++;
  broadcast.attr.sched.axis = normalAxis;
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.axis = normalAxis;
  *broadcast.y.repeats = normalRepeat;
  *broadcast.y.strides = normalStride;

  Sum sum("sum");
  sum.x = broadcast.y;
  sum.attr.sched.exec_order = exec_order++;
  sum.attr.sched.axis = normalAxis;
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.axis = normalAxis;
  *sum.y.repeats = reduceResRepeat;
  *sum.y.strides = reduceResStride;

  Cast cast1("cast1");
  cast1.x = sum.y;
  cast1.attr.sched.exec_order = exec_order++;
  cast1.attr.sched.axis = normalAxis;
  cast1.y.dtype = ge::DT_FLOAT16;
  *cast1.y.axis = normalAxis;
  *cast1.y.repeats = reduceResRepeat;
  *cast1.y.strides = reduceResStride;

  Store store("store");
  store.x = cast1.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = normalAxis;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = normalAxis;
  *store.y.repeats = reduceResRepeat;
  *store.y.strides = reduceResStride;

  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufAfterScheduler1(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));

  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  vector<int64_t> VectorizedAxis{z1t.id, z2t.id};

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T.id, z1t.id);
  graph.ApplySplit(load, z2T.id, z2t.id);
  graph.ApplyMerge(load, z0z2T.id);
  graph.ApplySplit(load, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(load, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  load->attr.sched.loop_axis = z1T.id;
  load->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto cast0 = graph.FindNode("cast0");
  graph.ApplySplit(cast0, z1T.id, z1t.id);
  graph.ApplySplit(cast0, z2T.id, z2t.id);
  graph.ApplyMerge(cast0, z0z2T.id);
  graph.ApplySplit(cast0, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(cast0, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  cast0->attr.sched.loop_axis = z1T.id;
  cast0->outputs[0].attr.vectorized_axis = VectorizedAxis;
  
  auto broadcast = graph.FindNode("broadcast");
  graph.ApplySplit(broadcast, z1T.id, z1t.id);
  graph.ApplySplit(broadcast, z2T.id, z2t.id);
  graph.ApplyMerge(broadcast, z0z2T.id);
  graph.ApplySplit(broadcast, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(broadcast, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  broadcast->attr.sched.loop_axis = z1T.id;
  broadcast->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto sum = graph.FindNode("sum");
  graph.ApplySplit(sum, z1T.id, z1t.id);
  graph.ApplySplit(sum, z2T.id, z2t.id);
  graph.ApplyMerge(sum, z0z2T.id);
  graph.ApplySplit(sum, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(sum, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  sum->attr.sched.loop_axis = z1T.id;
  sum->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto cast1 = graph.FindNode("cast1");
  graph.ApplySplit(cast1, z1T.id, z1t.id);
  graph.ApplySplit(cast1, z2T.id, z2t.id);
  graph.ApplyMerge(cast1, z0z2T.id);
  graph.ApplySplit(cast1, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(cast1, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  cast1->attr.sched.loop_axis = z1T.id;
  cast1->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T.id, z1t.id);
  graph.ApplySplit(store, z2T.id, z2t.id);
  graph.ApplyMerge(store, z0z2T.id);
  graph.ApplySplit(store, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(store, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  store->attr.sched.loop_axis = z1T.id;
  store->outputs[0].attr.vectorized_axis = VectorizedAxis;
}

void BrcBufAfterQueBufAlloc1(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  auto input_data = graph.FindNode("input_data");
  input_data->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  input_data->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensorID++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = loadQue;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto cast0 = graph.FindNode("cast0");
  cast0->outputs[0].attr.mem.tensor_id = tensorID++;
  cast0->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  cast0->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  cast0->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast0->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  cast0->outputs[0].attr.buf.id = cast0Buf;
  cast0->outputs[0].attr.que.id = ge::kIdNone;
  cast0->outputs[0].attr.que.depth = ge::kIdNone;
  cast0->outputs[0].attr.que.buf_num = ge::kIdNone;
  cast0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto broadcast = graph.FindNode("broadcast");
  broadcast->outputs[0].attr.mem.tensor_id = tensorID++;
  broadcast->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  broadcast->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  broadcast->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  broadcast->outputs[0].attr.buf.id = broadcastBuf;
  broadcast->outputs[0].attr.que.id = ge::kIdNone;
  broadcast->outputs[0].attr.que.depth = ge::kIdNone;
  broadcast->outputs[0].attr.que.buf_num = ge::kIdNone;
  broadcast->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto sum = graph.FindNode("sum");
  sum->outputs[0].attr.mem.tensor_id = tensorID++;
  sum->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  sum->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  sum->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  sum->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  sum->outputs[0].attr.buf.id = sumBuf;
  sum->outputs[0].attr.que.id = ge::kIdNone;
  sum->outputs[0].attr.que.depth = ge::kIdNone;
  sum->outputs[0].attr.que.buf_num = ge::kIdNone;
  sum->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto cast1 = graph.FindNode("cast1");
  cast1->outputs[0].attr.mem.tensor_id = tensorID++;
  cast1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  cast1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  cast1->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  cast1->outputs[0].attr.buf.id = cast1Buf;
  cast1->outputs[0].attr.que.id = ge::kIdNone;
  cast1->outputs[0].attr.que.depth = ge::kIdNone;
  cast1->outputs[0].attr.que.buf_num = ge::kIdNone;
  cast1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensorID++;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store->outputs[0].attr.mem.reuse_id = 0;
  store->outputs[0].attr.opt.ref_tensor = 0;
}

void BrcBufBeforeAutoFuse2(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto Z0 = ge::Symbol("Z0");
  auto Z1 = ge::Symbol("Z1");
  auto Z2 = ge::Symbol("Z2");

  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto normalAxis = {z0.id, z1.id, z2.id};
  std::initializer_list<Expr> normalRepeat = {Z0, Z1, Z2};
  std::initializer_list<Expr> normalStride = {Z1 * Z2, Z2, ONE};

  std::initializer_list<Expr> beforeBrcRepeat = {Z0, ONE, Z2};
  std::initializer_list<Expr> beforeBrcStride = {Z2, ZERO, ONE};

  std::initializer_list<Expr> reduceResRepeat = {Z0, Z1, ONE};
  std::initializer_list<Expr> reduceResStride = {Z1, ONE, ZERO};

  int32_t exec_order = 0;
  Data input_data("input_data", graph);
  input_data.attr.sched.exec_order = exec_order++;
  input_data.attr.sched.axis = normalAxis;
  input_data.y.dtype = ge::DT_FLOAT16;
  *input_data.y.axis = normalAxis;
  *input_data.y.repeats = beforeBrcRepeat;
  *input_data.y.strides = beforeBrcStride;

  Load load("load");
  load.x = input_data.y;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = normalAxis;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = normalAxis;
  *load.y.repeats = beforeBrcRepeat;
  *load.y.strides = beforeBrcStride;

  Cast cast0("cast0");
  cast0.x = load.y;
  cast0.attr.sched.exec_order = exec_order++;
  cast0.attr.sched.axis = normalAxis;
  cast0.y.dtype = ge::DT_FLOAT;
  *cast0.y.axis = normalAxis;
  *cast0.y.repeats = beforeBrcRepeat;
  *cast0.y.strides = beforeBrcStride;

  Broadcast broadcast("broadcast");
  broadcast.x = cast0.y;
  broadcast.attr.sched.exec_order = exec_order++;
  broadcast.attr.sched.axis = normalAxis;
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.axis = normalAxis;
  *broadcast.y.repeats = normalRepeat;
  *broadcast.y.strides = normalStride;

  Sum sum("sum");
  sum.x = broadcast.y;
  sum.attr.sched.exec_order = exec_order++;
  sum.attr.sched.axis = normalAxis;
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.axis = normalAxis;
  *sum.y.repeats = reduceResRepeat;
  *sum.y.strides = reduceResStride;

  Cast cast1("cast1");
  cast1.x = sum.y;
  cast1.attr.sched.exec_order = exec_order++;
  cast1.attr.sched.axis = normalAxis;
  cast1.y.dtype = ge::DT_FLOAT16;
  *cast1.y.axis = normalAxis;
  *cast1.y.repeats = reduceResRepeat;
  *cast1.y.strides = reduceResStride;

  Store store("store");
  store.x = cast1.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = normalAxis;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = normalAxis;
  *store.y.repeats = reduceResRepeat;
  *store.y.strides = reduceResStride;

  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufBeforeAutoFuse3(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto Z0 = ge::Symbol("Z0");
  auto Z1 = ge::Symbol("Z1");
  auto Z2 = ge::Symbol("Z2");

  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto normalAxis = {z0.id, z1.id, z2.id};
  std::initializer_list<Expr> normalRepeat = {Z0, Z1, Z2};
  std::initializer_list<Expr> normalStride = {Z1 * Z2, Z2, ONE};

  std::initializer_list<Expr> beforeBrcRepeat = {Z0, ONE, Z2};
  std::initializer_list<Expr> beforeBrcStride = {Z2, ZERO, ONE};

  int32_t exec_order = 0;
  Data input_data("input_data", graph);
  input_data.attr.sched.exec_order = exec_order++;
  input_data.attr.sched.axis = normalAxis;
  input_data.y.dtype = ge::DT_FLOAT16;
  *input_data.y.axis = normalAxis;
  *input_data.y.repeats = beforeBrcRepeat;
  *input_data.y.strides = beforeBrcStride;

  Load load("load");
  load.x = input_data.y;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = normalAxis;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = normalAxis;
  *load.y.repeats = beforeBrcRepeat;
  *load.y.strides = beforeBrcStride;

  Cast cast0("cast0");
  cast0.x = load.y;
  cast0.attr.sched.exec_order = exec_order++;
  cast0.attr.sched.axis = normalAxis;
  cast0.y.dtype = ge::DT_FLOAT;
  *cast0.y.axis = normalAxis;
  *cast0.y.repeats = beforeBrcRepeat;
  *cast0.y.strides = beforeBrcStride;

  Broadcast broadcast("broadcast");
  broadcast.x = cast0.y;
  broadcast.attr.sched.exec_order = exec_order++;
  broadcast.attr.sched.axis = normalAxis;
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.axis = normalAxis;
  *broadcast.y.repeats = normalRepeat;
  *broadcast.y.strides = normalStride;

  Cast cast1("cast1");
  cast1.x = broadcast.y;
  cast1.attr.sched.exec_order = exec_order++;
  cast1.attr.sched.axis = normalAxis;
  cast1.y.dtype = ge::DT_FLOAT16;
  *cast1.y.axis = normalAxis;
  *cast1.y.repeats = normalRepeat;
  *cast1.y.strides = normalStride;

  Store store("store");
  store.x = cast1.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = normalAxis;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = normalAxis;
  *store.y.repeats = normalRepeat;
  *store.y.strides = normalStride;

  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufAfterScheduler3(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));

  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  vector<int64_t> VectorizedAxis{z1t.id, z2t.id};

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T.id, z1t.id);
  graph.ApplySplit(load, z2T.id, z2t.id);
  graph.ApplyMerge(load, z0z2T.id);
  graph.ApplySplit(load, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(load, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  load->attr.sched.loop_axis = z1T.id;
  load->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto cast0 = graph.FindNode("cast0");
  graph.ApplySplit(cast0, z1T.id, z1t.id);
  graph.ApplySplit(cast0, z2T.id, z2t.id);
  graph.ApplyMerge(cast0, z0z2T.id);
  graph.ApplySplit(cast0, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(cast0, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  cast0->attr.sched.loop_axis = z1T.id;
  cast0->outputs[0].attr.vectorized_axis = VectorizedAxis;
  
  auto broadcast = graph.FindNode("broadcast");
  graph.ApplySplit(broadcast, z1T.id, z1t.id);
  graph.ApplySplit(broadcast, z2T.id, z2t.id);
  graph.ApplyMerge(broadcast, z0z2T.id);
  graph.ApplySplit(broadcast, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(broadcast, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  broadcast->attr.sched.loop_axis = z1T.id;
  broadcast->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto cast1 = graph.FindNode("cast1");
  graph.ApplySplit(cast1, z1T.id, z1t.id);
  graph.ApplySplit(cast1, z2T.id, z2t.id);
  graph.ApplyMerge(cast1, z0z2T.id);
  graph.ApplySplit(cast1, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(cast1, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  cast1->attr.sched.loop_axis = z1T.id;
  cast1->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T.id, z1t.id);
  graph.ApplySplit(store, z2T.id, z2t.id);
  graph.ApplyMerge(store, z0z2T.id);
  graph.ApplySplit(store, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(store, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  store->attr.sched.loop_axis = z1T.id;
  store->outputs[0].attr.vectorized_axis = VectorizedAxis;
}

void BrcBufAfterQueBufAlloc3(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  auto input_data = graph.FindNode("input_data");
  input_data->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  input_data->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensorID++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = loadQue;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto cast0 = graph.FindNode("cast0");
  cast0->outputs[0].attr.mem.tensor_id = tensorID++;
  cast0->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  cast0->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  cast0->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast0->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  cast0->outputs[0].attr.buf.id = cast0Buf;
  cast0->outputs[0].attr.que.id = ge::kIdNone;
  cast0->outputs[0].attr.que.depth = ge::kIdNone;
  cast0->outputs[0].attr.que.buf_num = ge::kIdNone;
  cast0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto broadcast = graph.FindNode("broadcast");
  broadcast->outputs[0].attr.mem.tensor_id = tensorID++;
  broadcast->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  broadcast->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  broadcast->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  broadcast->outputs[0].attr.buf.id = broadcastBuf;
  broadcast->outputs[0].attr.que.id = ge::kIdNone;
  broadcast->outputs[0].attr.que.depth = ge::kIdNone;
  broadcast->outputs[0].attr.que.buf_num = ge::kIdNone;
  broadcast->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto cast1 = graph.FindNode("cast1");
  cast1->outputs[0].attr.mem.tensor_id = tensorID++;
  cast1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  cast1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  cast1->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  cast1->outputs[0].attr.buf.id = cast1Buf;
  cast1->outputs[0].attr.que.id = ge::kIdNone;
  cast1->outputs[0].attr.que.depth = ge::kIdNone;
  cast1->outputs[0].attr.que.buf_num = ge::kIdNone;
  cast1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensorID++;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store->outputs[0].attr.mem.reuse_id = 0;
  store->outputs[0].attr.opt.ref_tensor = 0;
}

void BrcBufBeforeAutoFuse4(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto Z0 = ge::Symbol("Z0");
  auto Z1 = ge::Symbol("Z1");
  auto Z2 = ge::Symbol("Z2");

  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto normalAxis = {z0.id, z1.id, z2.id};
  std::initializer_list<Expr> normalRepeat = {Z0, Z1, Z2};
  std::initializer_list<Expr> normalStride = {Z1 * Z2, Z2, ONE};

  std::initializer_list<Expr> beforeBrcRepeat = {Z0, ONE, Z2};
  std::initializer_list<Expr> beforeBrcStride = {Z2, ZERO, ONE};

  std::initializer_list<Expr> reduceResRepeat = {Z0, ONE, Z2};
  std::initializer_list<Expr> reduceResStride = {Z2, ZERO, ONE};

  int32_t exec_order = 0;
  Data input_data("input_data", graph);
  input_data.attr.sched.exec_order = exec_order++;
  input_data.attr.sched.axis = normalAxis;
  input_data.y.dtype = ge::DT_FLOAT16;
  *input_data.y.axis = normalAxis;
  *input_data.y.repeats = beforeBrcRepeat;
  *input_data.y.strides = beforeBrcStride;

  Load load("load");
  load.x = input_data.y;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = normalAxis;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = normalAxis;
  *load.y.repeats = beforeBrcRepeat;
  *load.y.strides = beforeBrcStride;

  Cast cast0("cast0");
  cast0.x = load.y;
  cast0.attr.sched.exec_order = exec_order++;
  cast0.attr.sched.axis = normalAxis;
  cast0.y.dtype = ge::DT_FLOAT;
  *cast0.y.axis = normalAxis;
  *cast0.y.repeats = beforeBrcRepeat;
  *cast0.y.strides = beforeBrcStride;

  Sum sum("sum");
  sum.x = cast0.y;
  sum.attr.sched.exec_order = exec_order++;
  sum.attr.sched.axis = normalAxis;
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.axis = normalAxis;
  *sum.y.repeats = reduceResRepeat;
  *sum.y.strides = reduceResStride;

  Cast cast1("cast1");
  cast1.x = sum.y;
  cast1.attr.sched.exec_order = exec_order++;
  cast1.attr.sched.axis = normalAxis;
  cast1.y.dtype = ge::DT_FLOAT16;
  *cast1.y.axis = normalAxis;
  *cast1.y.repeats = reduceResRepeat;
  *cast1.y.strides = reduceResStride;

  Store store("store");
  store.x = cast1.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = normalAxis;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = normalAxis;
  *store.y.repeats = reduceResRepeat;
  *store.y.strides = reduceResStride;

  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufAfterScheduler4(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));

  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  vector<int64_t> VectorizedAxis{z1t.id, z2t.id};

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T.id, z1t.id);
  graph.ApplySplit(load, z2T.id, z2t.id);
  graph.ApplyMerge(load, z0z2T.id);
  graph.ApplySplit(load, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(load, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  load->attr.sched.loop_axis = z1T.id;
  load->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto cast0 = graph.FindNode("cast0");
  graph.ApplySplit(cast0, z1T.id, z1t.id);
  graph.ApplySplit(cast0, z2T.id, z2t.id);
  graph.ApplyMerge(cast0, z0z2T.id);
  graph.ApplySplit(cast0, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(cast0, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  cast0->attr.sched.loop_axis = z1T.id;
  cast0->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto sum = graph.FindNode("sum");
  graph.ApplySplit(sum, z1T.id, z1t.id);
  graph.ApplySplit(sum, z2T.id, z2t.id);
  graph.ApplyMerge(sum, z0z2T.id);
  graph.ApplySplit(sum, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(sum, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  sum->attr.sched.loop_axis = z1T.id;
  sum->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto cast1 = graph.FindNode("cast1");
  graph.ApplySplit(cast1, z1T.id, z1t.id);
  graph.ApplySplit(cast1, z2T.id, z2t.id);
  graph.ApplyMerge(cast1, z0z2T.id);
  graph.ApplySplit(cast1, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(cast1, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  cast1->attr.sched.loop_axis = z1T.id;
  cast1->outputs[0].attr.vectorized_axis = VectorizedAxis;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T.id, z1t.id);
  graph.ApplySplit(store, z2T.id, z2t.id);
  graph.ApplyMerge(store, z0z2T.id);
  graph.ApplySplit(store, z0z2TB.id, z0z2Tb.id);
  graph.ApplyReorder(store, {z0z2TB.id, z0z2Tb.id, z1T.id, z1t.id, z2t.id});
  store->attr.sched.loop_axis = z1T.id;
  store->outputs[0].attr.vectorized_axis = VectorizedAxis;
}

void BrcBufAfterQueBufAlloc4(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  auto input_data = graph.FindNode("input_data");
  input_data->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  input_data->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensorID++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = loadQue;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto cast0 = graph.FindNode("cast0");
  cast0->outputs[0].attr.mem.tensor_id = tensorID++;
  cast0->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  cast0->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  cast0->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast0->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  cast0->outputs[0].attr.buf.id = cast0Buf;
  cast0->outputs[0].attr.que.id = ge::kIdNone;
  cast0->outputs[0].attr.que.depth = ge::kIdNone;
  cast0->outputs[0].attr.que.buf_num = ge::kIdNone;
  cast0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto sum = graph.FindNode("sum");
  sum->outputs[0].attr.mem.tensor_id = tensorID++;
  sum->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  sum->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  sum->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  sum->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  sum->outputs[0].attr.buf.id = sumBuf;
  sum->outputs[0].attr.que.id = ge::kIdNone;
  sum->outputs[0].attr.que.depth = ge::kIdNone;
  sum->outputs[0].attr.que.buf_num = ge::kIdNone;
  sum->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  
  auto cast1 = graph.FindNode("cast1");
  cast1->outputs[0].attr.mem.tensor_id = tensorID++;
  cast1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  cast1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  cast1->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  cast1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  cast1->outputs[0].attr.buf.id = cast1Buf;
  cast1->outputs[0].attr.que.id = ge::kIdNone;
  cast1->outputs[0].attr.que.depth = ge::kIdNone;
  cast1->outputs[0].attr.que.buf_num = ge::kIdNone;
  cast1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensorID++;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store->outputs[0].attr.mem.reuse_id = 0;
  store->outputs[0].attr.opt.ref_tensor = 0;
}
}
