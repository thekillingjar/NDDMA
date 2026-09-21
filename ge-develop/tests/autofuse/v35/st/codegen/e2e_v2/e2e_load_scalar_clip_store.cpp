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

void LoadScalarClipStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *(x1.y.axis) = {z0.id, z1.id};

  Data x2("x2");
  graph.AddNode(x2);
  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_FLOAT;
  *(x2.y.axis) = {z0.id, z1.id};

  Data x3("x3");
  graph.AddNode(x3);
  x3.attr.sched.axis = {z0.id, z1.id};
  x3.y.dtype = ge::DT_FLOAT;
  *(x3.y.axis) = {z0.id, z1.id};

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};

  Load load2("load2");
  graph.AddNode(load2);
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {One, One};
  *load2.y.strides = {Zero, Zero};

  Load load3("load3");
  graph.AddNode(load3);
  load3.x = x3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {One, One};
  *load3.y.strides = {Zero, Zero};

  ge::ascir_op::ClipByValue clipbyvalue("clipbyvalue");
  graph.AddNode(clipbyvalue);
  clipbyvalue.x1 = load1.y;
  clipbyvalue.x2 = load2.y;
  clipbyvalue.x3 = load3.y;
  clipbyvalue.attr.sched.axis = {z0.id, z1.id};
  clipbyvalue.y.dtype = ge::DT_FLOAT;
  *clipbyvalue.y.axis = {z0.id, z1.id};
  *clipbyvalue.y.repeats = {s0, s1};
  *clipbyvalue.y.strides = {s1, One};

  Store store("store");
  graph.AddNode(store);
  store.x = clipbyvalue.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
}

void LoadScalarClipStore_AfterAutofuse(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;
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
  x2->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;
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
  x3->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;
  x3->attr.api.type = ApiType::kAPITypeBuffer;
  x3->attr.api.unit = ComputeUnit::kUnitNone;
  x3->outputs[0].attr.mem.tensor_id = 2;
  x3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x3->outputs[0].attr.mem.position = Position::kPositionGM;
  x3->outputs[0].attr.buf.id = ge::kIdNone;
  x3->outputs[0].attr.que.id = ge::kIdNone;
  x3->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  vector<AxisId> vectorized_axis{z1};
  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = ge::DT_FLOAT;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = {Zero};
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
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0;
  load2->outputs[0].attr.vectorized_axis = vectorized_axis;
  load2->outputs[0].attr.vectorized_strides = {Zero};
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = ge::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.dtype = ge::DT_FLOAT;
  load3->attr.api.compute_type = ComputeType::kComputeLoad;
  load3->attr.api.type = ApiType::kAPITypeCompute;
  load3->attr.api.unit = ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0;
  load3->outputs[0].attr.vectorized_axis = vectorized_axis;
  load3->outputs[0].attr.vectorized_strides = {Zero};
  load3->outputs[0].attr.mem.tensor_id = 5;
  load3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load3->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load3->outputs[0].attr.buf.id = ge::kIdNone;
  load3->outputs[0].attr.que.id = 2;
  load3->outputs[0].attr.mem.reuse_id = 0;
  load3->outputs[0].attr.que.depth = 2;
  load3->outputs[0].attr.que.buf_num = 2;
  load3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto clipbyvalue = graph.FindNode("clipbyvalue");
  clipbyvalue->outputs[0].attr.dtype = ge::DT_FLOAT;
  clipbyvalue->attr.api.compute_type = ComputeType::kComputeElewise;
  clipbyvalue->attr.api.type = ApiType::kAPITypeCompute;
  clipbyvalue->attr.api.unit = ComputeUnit::kUnitVector;
  clipbyvalue->attr.sched.loop_axis = z0;
  clipbyvalue->outputs[0].attr.vectorized_axis = vectorized_axis;
  clipbyvalue->outputs[0].attr.vectorized_strides = {One};
  clipbyvalue->outputs[0].attr.mem.tensor_id = 6;
  clipbyvalue->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  clipbyvalue->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  clipbyvalue->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  clipbyvalue->outputs[0].attr.buf.id = 0;
  clipbyvalue->outputs[0].attr.que.id = ge::kIdNone;
  clipbyvalue->outputs[0].attr.mem.reuse_id = 0;
  clipbyvalue->outputs[0].attr.que.depth = 2;
  clipbyvalue->outputs[0].attr.que.buf_num = 2;
  clipbyvalue->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  clipbyvalue->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = ge::DT_FLOAT;
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.mem.tensor_id = 7;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}
