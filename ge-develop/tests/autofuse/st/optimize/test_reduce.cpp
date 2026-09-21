/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#include "autoschedule/autoschedule.h"
#undef private
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "attr_utils.h"
#include "graph/debug/ge_op_types.h"
#include "autoschedule/axis_group.h"
#include "schedule_utils.h"
#include "fused_graph/fused_graph_unfolder.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using optimize::autoschedule::AxisGroup;

void Construct_Reduce_RARA(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT;
  *b0_load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  b0_max.x = abs.y;
  b0_max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_max.y.repeats = {One, s1, One, s3};
  *b0_max.y.strides = {Zero, s3, Zero, One};

  Store b3_store("b3_store");
  b3_store.x = b0_max.y;
  b3_store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b3_store.attr.api.compute_type = ComputeType::kComputeStore;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b3_store.y.repeats = {One, s1, One, s3};
  *b3_store.y.strides = {Zero, s3, Zero, One};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf3.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
  buf3.ir_attr.SetIndex(0);
}

void Construct_Reduce_ARAR(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT;
  *b0_load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  b0_max.x = abs.y;
  b0_max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_max.y.repeats = {s0, One, s2, One};
  *b0_max.y.strides = {s2, Zero, One, Zero};

  Store b3_store("b3_store");
  b3_store.x = b0_max.y;
  b3_store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b3_store.attr.api.compute_type = ComputeType::kComputeStore;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b3_store.y.repeats = {s0, One, s2, One};
  *b3_store.y.strides = {s2, Zero, One, Zero};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf3.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
  buf3.ir_attr.SetIndex(0);
}

void Construct_Reduce_RR(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, ge::ops::One};
  *load.y.repeats = {s1, s0};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {ge::ops::One, ge::ops::One};
  *sum.y.repeats = {ge::ops::Zero, ge::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = sum.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.axis = {z0.id, z1.id};
  *store_op1.y.strides = {ge::ops::One, ge::ops::One};
  *store_op1.y.repeats = {ge::ops::Zero, ge::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void Construct_Mul_Consumer_Struct(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT16;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  b1_exp.x = b0_load.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.api.compute_type = ComputeType::kComputeElewise;
  b1_exp.attr.api.type = ge::ApiType::kAPITypeCompute;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Abs b0_abs("b0_abs");
  b0_abs.x = b1_exp.y;
  b0_abs.attr.sched.axis = axis;
  b0_abs.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_abs.y.dtype = ge::DT_FLOAT16;
  *b0_abs.y.axis = axis;
  *b0_abs.y.repeats = {s0 * s1 * s2, s3};
  *b0_abs.y.strides = {s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_abs.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, s3};
  *b0_max.y.strides = {One, Zero};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  Store b0_store("b0_store");
  b0_store.x = b1_broadcast.y;
  b0_store.attr.sched.axis = axis;
  b0_store.attr.api.compute_type = ComputeType::kComputeStore;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0 * s1 * s2, s3};
  *b0_store.y.strides = {s3, One};

  Output buf0("buf0");
  buf0.x = b0_store.y;
  buf0.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf0.y.dtype = ge::DT_FLOAT;
  buf0.ir_attr.SetIndex(1);

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = b1_exp.y;
  b0_relu.attr.sched.axis = axis;
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT16;
  *b0_relu.y.axis = axis;
  *b0_relu.y.repeats = {s0 * s1 * s2, s3};
  *b0_relu.y.strides = {s3, One};

  Store b1_store("b1_store");
  b1_store.x = b0_relu.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};

  Output buf1("buf1");
  buf1.x = b1_store.y;
  buf1.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf1.y.dtype = ge::DT_FLOAT;
  buf1.ir_attr.SetIndex(2);
}

void ConstructNormStruct(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Exp exp("exp");
  exp.x = load.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {s1 ,ge::ops::One};
  *exp.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = exp.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {ge::ops::One, ge::ops::One};
  *sum.y.repeats = {ge::ops::Zero, ge::ops::Zero};

  Broadcast broadcast("broadcast");
  broadcast.x = sum.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1 ,ge::ops::One};
  *broadcast.y.repeats = {s0, s1};

  Sub sub("sub");
  sub.x1 = broadcast.y;
  sub.x2 = exp.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  *sub.y.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.strides = {s1 ,ge::ops::One};
  *sub.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = sub.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1 ,ge::ops::One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct3Elewise(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Exp exp("exp");
  exp.x = abs.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *exp.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = exp.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct1Elewise(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = abs.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Tanh tanh("tanh");
  tanh.x = abs.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Exp exp("exp");
  exp.x = tanh.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *exp.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = exp.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise4ReduceMultipleCitations(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Tanh tanh("tanh");
  tanh.x = sum.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Add add("add");
  add.x1 = abs.y;
  add.x2 = tanh.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *add.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = add.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise3ReduceMultipleCitations(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Tanh tanh("tanh");
  tanh.x = sum.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Add add("add");
  add.x1 = abs.y;
  add.x2 = tanh.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *add.y.repeats = {ge::ops::One, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise3(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {s1 ,ge::ops::One};
  *load1.y.repeats = {s0, s1};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = load0.y;
  mul.x2 = load1.y;
  *mul.y.axis = {z0.id, z1.id};
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.strides = {s1 ,ge::ops::One};
  *mul.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = mul.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = sum.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Tanh tanh("tanh");
  tanh.x = b0_relu.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Add add("add");
  add.x1 = tanh.y;
  add.x2 = b0_relu.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *add.y.repeats = {ge::ops::One, ge::ops::One};

  Abs abs("abs");
  abs.x = add.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = abs.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void Construct_Reduce_Cast_RR(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, ge::ops::One};
  *load.y.repeats = {s1, s0};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {ge::ops::One, ge::ops::One};
  *sum.y.repeats = {ge::ops::Zero, ge::ops::Zero};

  Cast cast("cast");
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.x = sum.y;
  *cast.y.axis = {z0.id, z1.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.strides = {ge::ops::One, ge::ops::One};
  *cast.y.repeats = {ge::ops::Zero, ge::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = cast.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.axis = {z0.id, z1.id};
  *store_op1.y.strides = {ge::ops::One, ge::ops::One};
  *store_op1.y.repeats = {ge::ops::Zero, ge::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4MulReduce(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(39);
  auto s2 = graph.CreateSizeVar(80);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);


  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1, One};
  *data0.y.strides = {s1, One, Zero};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, One, Zero};
  *load0.y.repeats = {s0, s1 ,ge::ops::One};

  Broadcast brc("brc");
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.x = load0.y;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.strides = {s1 * s2, s2, One};
  *brc.y.repeats = {s0, s1 ,s2};

  Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  data1.y.dtype = ge::DT_FLOAT;
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1 ,s2};
  *data1.y.strides = {s1 * s2, s2, One};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1 ,s2};
  *load1.y.strides = {s1 * s2, s2, One};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul.x1 = brc.y;
  mul.x2 = load1.y;
  *mul.y.axis = {z0.id, z1.id, z2.id};
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.strides = {s1 * s2, s2, One};
  *mul.y.repeats = {s0, s1 ,s2};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op1.x = mul.y;
  *store_op1.y.axis = {z0.id, z1.id, z2.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1 * s2, s2, One};
  *store_op1.y.repeats = {s0, s1 ,s2};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id, z2.id};
  sum.x = mul.y;
  *sum.y.axis = {z0.id, z1.id, z2.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {s0, ge::ops::One, s2};
  *sum.y.strides = {s2, ge::ops::Zero, One};

  Store store_op2("store2");
  store_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op2.x = sum.y;
  *store_op2.y.axis = {z0.id, z1.id, z2.id};
  store_op2.y.dtype = ge::DT_FLOAT;
  *store_op2.y.strides = {s2, ge::ops::Zero, One};
  *store_op2.y.repeats = {s0, ge::ops::One, s2};

  Output output_op2("output2");
  output_op2.x = store_op2.y;
  output_op2.y.dtype = ge::DT_FLOAT;
  output_op2.ir_attr.SetIndex(1);

  Mul mul1("mul1");
  mul1.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul1.x1 = mul.y;
  mul1.x2 = mul.y;
  *mul1.y.axis = {z0.id, z1.id, z2.id};
  mul1.y.dtype = ge::DT_FLOAT;
  *mul1.y.strides = {s1 * s2, s2, One};
  *mul1.y.repeats = {s0, s1 ,s2};


  Sum sum1("sum1");
  sum1.attr.sched.axis = {z0.id, z1.id, z2.id};
  sum1.x = mul1.y;
  *sum1.y.axis = {z0.id, z1.id, z2.id};
  sum1.y.dtype = ge::DT_FLOAT;
  *sum1.y.repeats = {s0, ge::ops::One, s2};
  *sum1.y.strides = {s2, ge::ops::Zero, One};


  Store store_op3("store3");
  store_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op3.x = sum1.y;
  *store_op3.y.axis = {z0.id, z1.id, z2.id};
  store_op3.y.dtype = ge::DT_FLOAT;
  *store_op3.y.strides = {s2, ge::ops::Zero, One};
  *store_op3.y.repeats = {s0, ge::ops::One, s2};

  Output output_op3("output3");
  output_op3.x = store_op3.y;
  output_op3.y.dtype = ge::DT_FLOAT;
  output_op3.ir_attr.SetIndex(2);
}

void ConstructNormStruct3ElemwiseReducePostMulInput(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {s1 ,ge::ops::One};
  *load1.y.repeats = {s0, s1};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = load0.y;
  mul.x2 = load1.y;
  *mul.y.axis = {z0.id, z1.id};
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.strides = {s1 ,ge::ops::One};
  *mul.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = mul.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = sum.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Tanh tanh("tanh");
  tanh.x = b0_relu.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Add add("add");
  add.x1 = tanh.y;
  add.x2 = b0_relu.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *add.y.repeats = {ge::ops::One, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct3ElemwiseReducePostMulInputV2(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {s1 ,ge::ops::One};
  *load1.y.repeats = {s0, s1};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = load0.y;
  mul.x2 = load1.y;
  *mul.y.axis = {z0.id, z1.id};
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.strides = {s1 ,ge::ops::One};
  *mul.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = mul.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = sum.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Tanh tanh("tanh");
  tanh.x = b0_relu.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Add add("add");
  add.x1 = tanh.y;
  add.x2 = mul.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *add.y.repeats = {ge::ops::One, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise4ReduceMultipleCitationsMulOut(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 ,ge::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Tanh tanh("tanh");
  tanh.x = sum.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Add add("add");
  add.x1 = sum.y;
  add.x2 = tanh.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *add.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = sum.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {ge::ops::One, ge::ops::One};
  *b0_relu.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op1.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  Store store_op2("store2");
  store_op2.attr.sched.axis = {z0.id, z1.id};
  store_op2.x = add.y;
  *store_op2.y.axis = {z0.id, z1.id};
  store_op2.y.dtype = ge::DT_FLOAT;
  *store_op2.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op2.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op1("output1");
  output_op1.x = store_op2.y;
  output_op1.y.dtype = ge::DT_FLOAT;
  output_op1.ir_attr.SetIndex(1);

  Store store_op3("store3");
  store_op3.attr.sched.axis = {z0.id, z1.id};
  store_op3.x = abs.y;
  *store_op3.y.axis = {z0.id, z1.id};
  store_op3.y.dtype = ge::DT_FLOAT;
  *store_op3.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *store_op3.y.repeats = {ge::ops::One, ge::ops::One};

  Output output_op2("output2");
  output_op1.x = store_op3.y;
  output_op1.y.dtype = ge::DT_FLOAT;
  output_op1.ir_attr.SetIndex(2);
}

namespace optimize {
class OptimizerReduceSt : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  optimize::Optimizer optimizer;

  OptimizerReduceSt() : optimizer(optimize::OptimizerOptions{}) {}

  static std::string ExpressToStr(std::vector<ge::Expression> exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }

  static std::string AxisToStr(ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    std::stringstream ss;
    for (auto axis_id : node->outputs[0].attr.axis) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss.str();
  }
};

TEST_F(OptimizerReduceSt, TestReduce_RARA) {
  ge::AscGraph graph("REDUCE_RARA");
  Construct_Reduce_RARA(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 5UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
}

TEST_F(OptimizerReduceSt, TestReduce_MUL_CONSUMER) {
  ge::AscGraph graph("REDUCE_MUL_CONSUMER");
  Construct_Mul_Consumer_Struct(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_ARAR) {
  ge::AscGraph graph("REDUCE_ARAR");
  Construct_Reduce_ARAR(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 5UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
}

TEST_F(OptimizerReduceSt, TestReduce_RR) {
  ge::AscGraph graph("REDUCE_RR");
  Construct_Reduce_RR(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Cast_RR) {
  ge::AscGraph graph("REDUCE_Cast_RR");
  Construct_Reduce_Cast_RR(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_PatitionNorm) {
  ge::AscGraph graph("reduce_patition_norm");
  ConstructNormStruct(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Store) {
  ge::AscGraph graph("reduce_three_elewise_store");
  ConstructNormStruct3Elewise(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_One_Elewise_Store) {
  ge::AscGraph graph("reduce_one_elewise_store");
  ConstructNormStruct1Elewise(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Four_Elewise_Store) {
  ge::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4Elewise(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Four_Elewise_Store_V2) {
  ge::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4Elewise4ReduceMultipleCitations(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Store_Multi_Citation) {
  ge::AscGraph graph("reduce_three_elewise_store");
  ConstructNormStruct4Elewise3ReduceMultipleCitations(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Four_Elewise_Store_V3) {
  ge::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4Elewise3(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Elewise_Store_MulReduce) {
  ge::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4MulReduce(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 6UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][3UL].schedule_groups.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][4UL].schedule_groups.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][5UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Reduce_Post_Node_Multi_Input_V1) {
  ge::AscGraph graph("reduce_three_elewise_store");
  ConstructNormStruct3ElemwiseReducePostMulInput(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Reduce_Post_Node_Multi_Input_V2) {
  ge::AscGraph graph("reduce_three_elewise_store");
  ConstructNormStruct3ElemwiseReducePostMulInputV2(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Store_Multi_Citation_Multi_Out) {
  ge::AscGraph graph("reduce_three_elewise_store");
  ConstructNormStruct4Elewise4ReduceMultipleCitationsMulOut(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}
}  // namespace optimize
