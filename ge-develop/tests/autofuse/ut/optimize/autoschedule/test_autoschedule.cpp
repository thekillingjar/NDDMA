/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ascendc_ir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "graph_utils_ex.h"

#define private public
#include "autoschedule/autoschedule.h"
#include "autoschedule/alignment_handler.h"
#undef private
#include "ascir_ops_utils.h"
#include "autoschedule/tiling_group.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"
#include "ascgraph_info_complete.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace optimize::autoschedule;

static void Construct_LoadAbsStore(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}

static void Construct_LoadGatherStore(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s2 + s3);

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x2.y.dtype = ge::DT_INT32;
  x2.attr.sched.axis = {z4.id};
  *x2.y.axis = {z4.id};
  *x2.y.repeats = {s2 + s3};
  *x2.y.strides = {One};

  ge::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.api.compute_type = ComputeType::kComputeGather;
  gather.y.dtype = ge::DT_FLOAT;
  gather.attr.sched.axis = {z0.id, z4.id};
  *gather.y.axis = {z0.id, z4.id};
  *gather.y.repeats = {s0, s2 + s3};
  *gather.y.strides = {s2 + s3, One};
  gather.ir_attr.SetAxis(1);

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = gather.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT;
  store.attr.sched.axis = {z0.id, z4.id};
  *store.y.axis = {z0.id, z4.id};
  *store.y.repeats = {s0, s2 + s3};
  *store.y.strides = {s2 + s3, One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT;
  y.attr.sched.axis = {z0.id, z4.id};
  *y.y.axis = {z0.id, z4.id};
  *y.y.repeats = {s0, s2 + s3};
  *y.y.strides = {s2 + s3, One};
}

static void Construct_ElementwiseAbs(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");

  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id};
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id};
  *x.y.repeats = {s0};
  *x.y.strides = {One};

  ge::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id};
  *y.y.repeats = {s0};
  *y.y.strides = {One};
}

static void Construct_ElementwiseFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");

  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id};
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id};
  *x.y.repeats = {s0};
  *x.y.strides = {One};

  ge::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load.y;
  abs0.attr.sched.axis = {z0.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id};
  *abs0.y.repeats = {s0};
  *abs0.y.strides = {One};

  ge::ascir_op::Abs abs1("abs1");
  graph.AddNode(abs1);
  abs1.x = abs0.y;
  abs1.attr.sched.axis = {z0.id};
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.y.dtype = ge::DT_FLOAT16;
  *abs1.y.axis = {z0.id};
  *abs1.y.repeats = {s0};
  *abs1.y.strides = {One};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs1.y;
  store.attr.sched.axis = {z0.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id};
  *y.y.repeats = {s0};
  *y.y.strides = {One};
}

/**
 *                   store
 *                     |
 *                   mul0
 *                  /   \
 *               add0  exp1
 *              /    \ /
 *    (remove)brc1    \
 *             |      |
 *            exp0   brc0(remove)
 *              \   /
 *              abs0
 *               |
 *             load0
 *              |
 *            data0
 */
static void Construct_RedundantBroadcastFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id};
  *abs0.y.repeats = {One, s1, s2};
  *abs0.y.strides = {Zero, s2, One};

  ge::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = abs0.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp0.attr.api.compute_type = ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id};
  *exp0.y.repeats = {One, s1, s2};
  *exp0.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = exp0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = brc0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp1.attr.api.compute_type = ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id};
  *exp1.y.repeats = {s0, s1, s2};
  *exp1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Mul mul0("mul0");
  graph.AddNode(mul0);
  mul0.x1 = add0.y;
  mul0.x2 = exp1.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = mul0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}

/**
 *                    store (s0,s1,s2,s3,s4)
 *                      |
 *                    add0 (s0,s1,s2,s3,s4)
 *                   /    \
 *                  |     exp0 (s0,s1,s2,s3,s4)
 *                   \    /
 *                    \  /
 *                   brc4 (s0,s1,s2,s3,s4)
 *                    |
 *                  brc3 (1,s1,s2,s3,s4)
 *                   |
 *                 brc2 (1,1,s2,s3,s4)
 *                  |
 *                brc1 (1,1,1,s3,s4)
 *                 |
 *               brc0 (1,1,1,1,s4)
 *                |
 *              abs0 (1,1,1,1,1)
 *               |
 *             load0 (1,1,1,1,1)
 *              |
 *            data0 (1,1,1,1,1)
 */
static void Construct_ContinuesBroadcastScalarFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, One, One, One};
  *data0.y.strides = {Zero, Zero, Zero, Zero, Zero};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, One, One, One};
  *load0.y.strides = {Zero, Zero, Zero, Zero, Zero};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, One, One, One};
  *abs0.y.strides = {Zero, Zero, Zero, Zero, Zero};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, One, One, s4};
  *brc0.y.strides = {Zero, Zero, Zero, Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, One, s3, s4};
  *brc1.y.strides = {Zero, Zero, Zero, s4, One};

  ge::ascir_op::Broadcast brc2("brc2");
  graph.AddNode(brc2);
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {One, One, s2, s3, s4};
  *brc2.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc3("brc3");
  graph.AddNode(brc3);
  brc3.x = brc2.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc4("brc4");
  graph.AddNode(brc4);
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = brc4.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.attr.api.compute_type = ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = exp0.y;
  add0.x2 = brc4.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = add0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, s4};
  *y.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
}

/**
 *                    store0
 *                      |
 *                    add0
 *                   /    \
 *                  |     exp0 (s0,s1,s2,s3,s4)
 *                   \    /
 *                    \  /
 *                   brc4 (s0,s1,s2,s3,s4)
 *                    |
 *                  brc3 (1,s1,s2,s3,s4)
 *                   |
 *         store1  brc1 (1,1,s2,s3,s4)
 *           \      |
 *          exp1  brc0 (1,1,s2,1,s4)
 *            \   /
 *            abs0 (1,1,s2,1,1)
 *             |
 *           load0 (1,1,s2,1,1)
 *            |
 *          data0 (1,1,s2,1,1)
 */
static void Construct_ContinuesBroadcastPartBrcFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, s2, One, One};
  *data0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, s2, One, One};
  *load0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, s2, One, One};
  *abs0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, s2, One, s4};
  *brc0.y.strides = {Zero, Zero, s4, Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc3("brc3");
  graph.AddNode(brc3);
  brc3.x = brc1.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc4("brc4");
  graph.AddNode(brc4);
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = brc4.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.attr.api.compute_type = ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = exp0.y;
  add0.x2 = brc4.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, s4};
  *y.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = abs0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp1.attr.api.compute_type = ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp1.y.repeats = {One, One, s2, One, One};
  *exp1.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = exp1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, One};
  *store1.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, One};
  *y1.y.strides = {Zero, Zero, One, Zero, Zero};
}

/**
 *              store0
 *                |
 *              brc1 (s1,s2,s3,s4)
 *               |
 *             brc0 (1,s2,s3,s4)
 *              |
 *            abs0 (1,s2,1,s4)
 *             |
 *           load0 (1,s2,1,s4)
 *            |
 *          data0 (1,s2,1,s4)
 */
static void Construct_ContinuesBroadcastIntervalBrcBABAFusion(ge::AscGraph &graph) {
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, s2, One, s4};
  *data0.y.strides = {Zero, s4, Zero, One};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, s2, One, s4};
  *load0.y.strides = {Zero, s4, Zero, One};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, s2, One, s4};
  *abs0.y.strides = {Zero, s4, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, s2, s3, s4};
  *brc0.y.strides = {Zero, s3 * s4, One, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s1, s2, s3, s4};
  *brc1.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = brc1.y;
  store0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s1, s2, s3, s4};
  *store0.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s1, s2, s3, s4};
  *y.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};
}

/**
 *              store0
 *                |
 *              brc1 (s1,s2,s3,s4)
 *               |
 *             brc0 (s1,1,s3,s4)
 *              |
 *            abs0 (s1,1,s3,1)
 *             |
 *           load0 (s1,1,s3,1)
 *            |
 *          data0 (s1,1,s3,1)
 */
static void Construct_ContinuesBroadcastIntervalBrcABABFusion(ge::AscGraph &graph) {
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {s1, One, s2, One};
  *data0.y.strides = {s2, Zero, One, Zero};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s1, One, s2, One};
  *load0.y.strides = {s2, Zero, One, Zero};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {s1, One, s2, One};
  *abs0.y.strides = {s2, Zero, One, Zero};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s1, One, s2, s4};
  *brc0.y.strides = {s2 * s4, Zero, s4, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s1, s2, s3, s4};
  *brc1.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  store0.x = brc1.y;
  store0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s1, s2, s3, s4};
  *store0.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s1, s2, s3, s4};
  *y.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};
}

/**
 *               store0
 *                 |
 *             reduce_sum (s0,s1,1,s3,s4)
 *                |
 *              brc1 (s0,s1,s2,s3,s4)
 *               |
 *             brc0 (s0,1,s2,s3,s4)
 *             |
 *           load0 (s0,1,s2,s3,1)
 *            |
 *          data0 (s0,1,s2,s3,1)
 */
static void Construct_ContinuesBroadcastIntervalBrcBAABFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {s0, One, s2, s3, One};
  *data0.y.strides = {s2 * s3, Zero, s3, One, Zero};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, One, s2, s3, One};
  *load0.y.strides = {s2 * s3, Zero, s3, One, Zero};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, One, s2, s3, s4};
  *brc0.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, s4};
  *brc1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Sum sum("reduce_sum");
  sum.x = brc1.y;
  sum.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sum.attr.api.compute_type = ComputeType::kComputeBroadcast;
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sum.y.repeats = {s0, s1, One, s3, s4};
  *sum.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  ge::ascir_op::Store store0("store0");
  store0.x = brc1.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, One, s3, s4};
  *store0.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, One, s3, s4};
  *y.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};
}

/**
 *                    store0
 *                      |
 *                    add0
 *                   /    \
 *                  |     exp0 (s0,s1,s2,s3,s4)
 *                  \    /
 *                  \  /
 *                  brc2 (s0,s1,s2,s3,s4)
 *                   |
 *         store1  brc1 (s0,1,s2,s3,s4)
 *           \      |
 *          exp1  brc0 (s0,1,1,s3,s4)
 *            \   /
 *            abs0 (s0,1,1,1,s4)
 *             |
 *           load0 (s0,1,1,1,s4)
 *            |
 *          data0 (s0,1,1,1,s4)
 */
static void Construct_ContinuesBroadcastPartBrcInMiddleFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {s0, One, One, One, s4};
  *data0.y.strides = {s4, Zero, Zero, Zero, One};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, One, One, One, s4};
  *load0.y.strides = {s4, Zero, Zero, Zero, One};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {s0, One, One, One, s4};
  *abs0.y.strides = {s4, Zero, Zero, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, One, One, s3, s4};
  *brc0.y.strides = {s3 * s4, Zero, Zero, s4, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, One, s2, s3, s4};
  *brc1.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc2("brc2");
  graph.AddNode(brc2);
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {s0, s1, s2, s3, s4};
  *brc2.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = brc2.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.attr.api.compute_type = ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = exp0.y;
  add0.x2 = brc2.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, s4};
  *y.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = abs0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp1.attr.api.compute_type = ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp1.y.repeats = {One, One, s2, One, One};
  *exp1.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = exp1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, One};
  *store1.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, One};
  *y1.y.strides = {Zero, Zero, One, Zero, Zero};
}

/**
 *         store1  store0
 *           \      /
 *         abs0  brc1 (1,1,s2,s3,s4)
 *            \  /
 *            brc0 (1,1,s2,1,s4)
 *             |
 *           load0 (1,1,s2,1,1)
 *            |
 *          data0 (1,1,s2,1,1)
 */
static void Construct_ContinuesBroadcastMultiOutFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, s2, One, One};
  *data0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, s2, One, One};
  *load0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, s2, One, s4};
  *brc0.y.strides = {Zero, Zero, s4, Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = brc1.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {One, One, s2, s3, s4};
  *store0.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {One, One, s2, s3, s4};
  *y.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = brc0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, s2, One, s4};
  *abs0.y.strides = {Zero, Zero, s4, Zero, One};

  ge::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = abs0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, s4};
  *store1.y.strides = {Zero, Zero, s4, Zero, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, s4};
  *y1.y.strides = {Zero, Zero, s4, Zero, One};
}

/**
 *                    store0
 *                      |
 *                   reduce0 (s0,s1,s2,s3,1)
 *                     |
 *                   brc4 (s0,s1,s2,s3,s4)
 *                    |
 *                  brc3 (1,s1,s2,s3,s4)
 *                   |
 *         store1  brc1 (1,1,s2,s3,s4)
 *           \      |
 *          exp1  brc0 (1,1,s2,1,s4)
 *            \   /
 *            abs0 (1,1,s2,1,1)
 *             |
 *           load0 (1,1,s2,1,1)
 *            |
 *          data0 (1,1,s2,1,1)
 */
static void Construct_ContinuesBroadcastWithReduceFusion(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, s2, One, One};
  *data0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, s2, One, One};
  *load0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, s2, One, One};
  *abs0.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, s2, One, s4};
  *brc0.y.strides = {Zero, Zero, s4, Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc3("brc3");
  graph.AddNode(brc3);
  brc3.x = brc1.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc4("brc4");
  graph.AddNode(brc4);
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Sum reduce_sum("reduce_sum");
  graph.AddNode(reduce_sum);
  reduce_sum.x = brc4.y;
  reduce_sum.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  reduce_sum.attr.api.compute_type = ComputeType::kComputeReduce;
  reduce_sum.y.dtype = ge::DT_FLOAT16;
  *reduce_sum.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *reduce_sum.y.repeats = {s0, s1, s2, s3, One};
  *reduce_sum.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};

  ge::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = reduce_sum.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, One};
  *store0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, One};
  *y.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};

  ge::ascir_op::Exp exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = abs0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp1.attr.api.compute_type = ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp1.y.repeats = {One, One, s2, One, One};
  *exp1.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = exp1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, One};
  *store1.y.strides = {Zero, Zero, One, Zero, Zero};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, One};
  *y1.y.strides = {Zero, Zero, One, Zero, Zero};
}

void Construct_Softmax(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  //   graph.AddNode(arg4_1);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT16;

  Load b0_load("b0_load");
  //   graph.AddNode(b0_load);
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  //   graph.AddNode(b0_max);
  b0_max.x = b0_load.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, s3};
  *b0_max.y.strides = {One, Zero};

  Output buf1("buf1");
  //   graph.AddNode(buf1);
  Load b1_load("b1_load");
  //   graph.AddNode(b1_load);
  b1_load.x = arg4_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0 * s1 * s2, s3};
  *b1_load.y.strides = {s3, One};

  Broadcast b1_broadcast("b1_broadcast");
  //   graph.AddNode(b1_broadcast);
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  ge::ascir_op::Sub b1_sub("b1_sub");
  //   graph.AddNode(b1_sub);
  b1_sub.x1 = b1_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.attr.api.compute_type = ComputeType::kComputeElewise;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0 * s1 * s2, s3};
  *b1_sub.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  //   graph.AddNode(b1_exp);
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.api.compute_type = ComputeType::kComputeElewise;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Store b1_store("b1_store");
  //   graph.AddNode(b1_store);
  b1_store.x = b1_exp.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};

  buf1.x = b1_store.y;
  buf1.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf1.y.dtype = ge::DT_FLOAT16;

  Load b2_load("b2_load");
  //   graph.AddNode(b2_load);
  b2_load.x = buf1.y;
  b2_load.attr.sched.axis = axis;
  b2_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b2_load.y.dtype = ge::DT_FLOAT16;
  *b2_load.y.axis = axis;
  *b2_load.y.repeats = {s0 * s1 * s2, s3};
  *b2_load.y.strides = {s3, One};

  Sum b2_sum("b2_sum");
  //   graph.AddNode(b2_sum);
  b2_sum.x = b2_load.y;
  b2_sum.attr.sched.axis = axis;
  b2_sum.attr.api.compute_type = ComputeType::kComputeReduce;
  b2_sum.y.dtype = ge::DT_FLOAT16;
  *b2_sum.y.axis = axis;
  *b2_sum.y.repeats = {s0 * s1 * s2, s3};
  *b2_sum.y.strides = {One, Zero};

  Output buf3("buf3");
  //   graph.AddNode(buf3);
  Load b3_load("b3_load");
  //   graph.AddNode(b3_load);
  b3_load.x = buf1.y;
  b3_load.attr.sched.axis = axis;
  b3_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b3_load.y.dtype = ge::DT_FLOAT16;
  *b3_load.y.axis = axis;
  *b3_load.y.repeats = {s0 * s1 * s2, s3};
  *b3_load.y.strides = {s3, One};

  Broadcast b3_broadcast("b3_broadcast");
  //   graph.AddNode(b3_broadcast);
  b3_broadcast.x = b2_sum.y;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;
  b3_broadcast.y.dtype = ge::DT_FLOAT16;
  *b3_broadcast.y.axis = axis;
  *b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b3_broadcast.y.strides = {s3, One};

  ge::ascir_op::Div b3_div("b3_div");
  //   graph.AddNode(b3_div);
  b3_div.x1 = b3_load.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.axis = axis;
  b3_div.attr.api.compute_type = ComputeType::kComputeElewise;
  b3_div.y.dtype = ge::DT_FLOAT16;
  *b3_div.y.axis = axis;
  *b3_div.y.repeats = {s0 * s1 * s2, s3};
  *b3_div.y.strides = {s3, One};

  Store b3_store("b3_store");
  //   graph.AddNode(b3_store);
  b3_store.x = b3_div.y;
  b3_store.attr.sched.axis = axis;
  b3_store.attr.api.compute_type = ComputeType::kComputeStore;
  b3_store.y.dtype = ge::DT_FLOAT16;
  *b3_store.y.axis = axis;
  *b3_store.y.repeats = {s0 * s1 * s2, s3};
  *b3_store.y.strides = {s3, One};

  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf3.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT16;
}

void Construct_Normal_Struct(ge::AscGraph &graph) {
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

  ge::ascir_op::Sub b1_sub("b1_sub");
  b1_sub.x1 = b1_exp.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.attr.api.compute_type = ComputeType::kComputeElewise;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0 * s1 * s2, s3};
  *b1_sub.y.strides = {s3, One};

  Store b1_store("b1_store");
  b1_store.x = b1_sub.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};
}

namespace optimize {
class AutoSchedulerUT : public ::testing::Test {
 protected:
  void SetUp() override {}
};
TEST_F(AutoSchedulerUT, TilingGroup_gen_elementwise_tilingGroup) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.FindNode("abs");

  AxisGroup group;
  TilingGroup::GenElewiseTilingGroup(*abs, group);

  AxisGroup expect_group;
  size_t base_order = 0UL;
  for (auto axis : abs->attr.sched.axis) {
    expect_group.y_group.push_back(axis);
    expect_group.axes_order.push_back(base_order++);
  }

  EXPECT_EQ(group, expect_group);
}

TEST_F(AutoSchedulerUT, AutoSchedule_eletwise_gen_tilingGroup) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.FindNode("abs");

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoSchedule(graph, results);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoSchedule.axes_group_), 0);
  EXPECT_EQ(autoSchedule.axes_group_.x_group.size(), 0UL);
  EXPECT_EQ(autoSchedule.axes_group_.y_group.size(), 3UL);
  EXPECT_EQ(autoSchedule.axes_group_.r_group.size(), 0UL);
  std::vector<ge::AxisId> y_group(abs->attr.sched.axis);
  EXPECT_EQ(autoSchedule.axes_group_.y_group, y_group);
}

TEST_F(AutoSchedulerUT, AutoSchedule_eletwise_fusion_gen_tilingGroup) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_ElementwiseFusion(graph);

  auto abs = graph.FindNode("abs1");

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoSchedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoSchedule.axes_group_), 0);
  EXPECT_EQ(autoSchedule.axes_group_.x_group.size(), 0);
  EXPECT_EQ(autoSchedule.axes_group_.y_group.size(), 1);
  EXPECT_EQ(autoSchedule.axes_group_.r_group.size(), 0);

  std::vector<ge::AxisId> y_group(abs->attr.sched.axis);
  EXPECT_EQ(autoSchedule.axes_group_.y_group, y_group);
}

TEST_F(AutoSchedulerUT, AutoSchedule_Tiling) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.FindNode("abs");
  std::vector<ge::AxisId> y_group = abs->attr.sched.axis;
  AxisGroup axes_group;
  axes_group.y_group = y_group;
  axes_group.axes_order = {0, 1, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = y_group[1];
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  std::vector<ge::AxisId> new_axis_order;
  scheduler.DoScheduler();

  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_x, -1);
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_y, 1);
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_r, -1);
  EXPECT_EQ(scheduler.tiling_case_.block_tiling_id, 5);
  EXPECT_EQ(graph.GetAllAxis().size(), 8);
  EXPECT_EQ(std::get<0>(scheduler.tiling_case_.ub_tiling_y)->id, 3);
  EXPECT_EQ(std::get<1>(scheduler.tiling_case_.ub_tiling_y)->id, 4);
  EXPECT_EQ(std::get<0>(scheduler.tiling_case_.block_tiling)->id, 6);
  EXPECT_EQ(std::get<1>(scheduler.tiling_case_.block_tiling)->id, 7);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_elementwise_3axis) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  ge::AscGraph except_graph("LoadAbsStore");
  except_graph.CopyFrom(graph);

  auto abs = graph.FindNode("abs");
  std::vector<ge::AxisId> y_group = abs->attr.sched.axis;

  AxisGroup axes_group;
  axes_group.y_group = y_group;
  axes_group.axes_order = {0, 1, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = 1;
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  scheduler.DoScheduler();
  auto dump_graph = utils::DebugStr(graph);

  int ub_axis_id = 1;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // fuse outer axes
  std::vector<AxisId> axes{output->attr.sched.axis[0], z_ub_out->id};
  auto block_axis = except_graph.MergeAxis(axes);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(block_axis->id);

  vector<AxisId> vectorize_axis = {z_ub_in->id, output->attr.sched.axis[2]};
  vector<ge::Expression> vectorized_strides{graph.FindAxis(vectorize_axis[1])->size, One};

  for (auto n : except_graph.GetAllNodes()) {
    if (n->attr.api.type == ge::ApiType::kAPITypeBuffer) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySchedAxisMerge(n, block_axis->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_elementwise_1axis) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_ElementwiseAbs(graph);

  ge::AscGraph except_graph("LoadAbsStore");
  except_graph.CopyFrom(graph);

  auto abs = graph.FindNode("abs");
  std::vector<ge::AxisId> y_group = abs->attr.sched.axis;

  AxisGroup axes_group;
  axes_group.y_group = y_group;
  axes_group.axes_order = {0};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = 0;
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  scheduler.DoScheduler();
  auto dump_graph = utils::DebugStr(graph);

  int ub_axis_id = 0;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out->id);

  vector<AxisId> vectorize_axis = {z_ub_in->id};
  vector<ge::Expression> vectorized_strides{One};

  for (auto n : except_graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_elementwise_1axis) {
  ge::AscGraph graph("LoadAbsStore");
  Construct_ElementwiseAbs(graph);

  ge::AscGraph except_graph("LoadAbsStore_B0Y0");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  ASSERT_EQ(results.size(), 1);
  auto dump_graph = utils::DebugStr(results[0].scheduled_graph);

  int ub_axis_id = 0;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out->id);

  vector<AxisId> vectorize_axis = {z_ub_in->id};
  vector<ge::Expression> vectorized_strides{One};

  for (auto n : except_graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_elementwise_fusion) {
  ge::AscGraph graph("AbsFusion");
  Construct_ElementwiseFusion(graph);

  ge::AscGraph except_graph("AbsFusion_B0Y0");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  auto dump_graph = utils::DebugStr(impl_graphs[0].scheduled_graph);

  int ub_axis_id = 0;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out->id);

  vector<AxisId> vectorize_axis = {z_ub_in->id};
  vector<ge::Expression> vectorized_strides{One};

  for (auto n : except_graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, ConcatGraph_scheduler) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto zo = graph.CreateAxis("zo", so);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_i.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Data data_j("data_j", graph);
  data_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.y.dtype = ge::DT_FLOAT16;
  *data_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_j.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Data data_k("data_k", graph);
  data_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.y.dtype = ge::DT_FLOAT16;
  *data_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.attr.api.compute_type = ComputeType::kComputeInvalid;
  data_k.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.repeats = {s0, s1, si, s2};
  *load_i.y.strides = {s1 * si * s2, si * s2, s2, ge::ops::One};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Load load_j("load_j");
  load_j.x = data_j.y;
  load_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.repeats = {s0, s1, sj, s2};
  *load_j.y.strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One};
  load_j.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Load load_k("load_k");
  load_k.x = data_k.y;
  load_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.repeats = {s0, s1, sk, s2};
  *load_k.y.strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};
  load_k.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Concat concat("concat");

  concat.x = {load_i.y, load_j.y, load_k.y};

  concat.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *concat.y.repeats = {s0, s1, so, s2};
  *concat.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store.y.repeats = {s0, s1, so, s2};
  *store.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  std::string res = utils::DebugStr(graph);
  ASSERT_NE(res, "");

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.DoAutoSchedule();
  ASSERT_EQ(results.size(), 2UL);
}

TEST_F(AutoSchedulerUT, align_vectorized_strides) {
  ge::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = ops::One;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1, One, Zero};

  ge::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1, One, Zero};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1, One, Zero};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1, One, Zero};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1, One, Zero};

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  }

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), SUCCESS);
  auto abs_node = graph.FindNode("abs");

  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[0].Str().get()), "s1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[2].Str().get()), "0");
}

TEST_F(AutoSchedulerUT, align_vectorized_strides_last_zero) {
  ge::AscGraph graph("align_vectorize");
  auto s0 = ge::Symbol(10);
  auto s1 = ge::Symbol(10);
  auto s2 = ge::Symbol(20);
  auto s3 = ge::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2, s2, One, Zero};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), SUCCESS);

  auto load_node = graph.FindNode("load_i");
  std::vector<Expression> golden_stride = {Symbol(20), One, Zero};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(AutoSchedulerUT, align_vectorized_strides_store_same_with_input) {
  ge::AscGraph graph("align_vectorize");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(9);
  auto s2 = ge::Symbol(8);
  auto s3 = ge::Symbol(1);

  auto s10 = Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.y.dtype = ge::DT_FLOAT16;
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s10};
  *load.y.strides = {s1 * s2 * s10, s2 * s10, s10, One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = load.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s2 * s3, s3, One, Zero};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), SUCCESS);

  auto s16 = Symbol(16);
  std::vector<Expression> golden_stride = {s2 * s16, s16, One};

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(AutoSchedulerUT, align_vectorized_strides_by_repeat) {
  ge::AscGraph graph("LoadAbsStore");
  auto s0 = ge::Symbol(10);
  auto s1 = ge::Symbol(9);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  ge::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, Zero};
  *load1.y.strides = {One, Zero};

  ge::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + One};
  *concat.y.strides = {s1 + One, One};

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + One};
  *store.y.strides = {s1 + One, One};

  ge::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  }

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), SUCCESS);

  auto load1_node = graph.FindNode("load1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "9");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(AutoSchedulerUT, merge_axis_load_broadcast) {
  ge::AscGraph graph("graph_merge_axis");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.repeats = {s0, ge::ops::One, ge::ops::One, s3, s4};
  *load.y.strides = {s3 * s4, ge::ops::Zero, ge::ops::Zero, s4, ge::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = load.y;
  broadcast.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast.y.repeats = {s0, s1, s2, s3, s4};
  *broadcast.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, ge::ops::One};
  broadcast.attr.api.compute_type = ComputeType::kComputeBroadcast;

  ge::ascir_op::Store store("store");
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.DoAutoSchedule();
  EXPECT_EQ(schedule.schd_outputs_.size(), 5);

  EXPECT_EQ(load.attr.sched.axis.size(), 3);
  EXPECT_EQ(load.y.axis->size(), 3);
  std::vector<ge::Expression> golden_repeats{s0, ge::ops::One, s3 * s4};
  EXPECT_EQ(*load.y.repeats, golden_repeats);
  std::vector<ge::Expression> golden_strides{s3 * s4, ge::ops::Zero, ge::ops::One};
  EXPECT_EQ(*load.y.strides, golden_strides);

  EXPECT_EQ(broadcast.attr.sched.axis.size(), 3);
  EXPECT_EQ(broadcast.y.axis->size(), 3);
  std::vector<ge::Expression> golden_repeats_broadcast{s0, s1 * s2, s3 * s4};
  EXPECT_EQ(*broadcast.y.repeats, golden_repeats_broadcast);
  std::vector<ge::Expression> golden_strides_broadcast{s1 * s2 * s3 * s4, s3 * s4, ge::ops::One};
  EXPECT_EQ(*broadcast.y.strides, golden_strides_broadcast);
}

TEST_F(AutoSchedulerUT, prune_zero_axis_skip_last_one) {
  ge::AscGraph graph("graph_merge_axis");
  auto s0 = ge::sym::kSymbolOne;
  auto s1 = ge::sym::kSymbolOne;
  auto s2 = ge::sym::kSymbolOne;
  auto s3 = ge::sym::kSymbolOne;
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {ge::sym::kSymbolZero, ge::sym::kSymbolZero, ge::sym::kSymbolZero, ge::sym::kSymbolZero};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.axes_group_.x_group = {-1};
  schedule.axes_group_.y_group = {0, 1, 2, 3};
  schedule.axes_group_.r_group = {-1};
  schedule.axes_group_.axes_order = {0, 1, 2, 3};

  std::vector<TilingCase> tiling_cases;
  schedule.GenTilingCase(tiling_cases);
  EXPECT_EQ(tiling_cases.size(), 4UL);
  schedule.PruneTilingCase(tiling_cases);
  ASSERT_EQ(tiling_cases.size(), 1UL);
  auto all_axis = graph.GetAllAxis();
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, 3);
}

TEST_F(AutoSchedulerUT, reorder_vectorized_axes_ok) {
  ge::AscGraph graph("reorder_vectorized_axes");
  auto size2 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", size2);
  auto z1 = graph.CreateAxis("z1", size2);
  auto z2 = graph.CreateAxis("z2", size2);
  auto z3 = graph.CreateAxis("z3", size2);
  auto z4 = graph.CreateAxis("z4", size2);
  auto z5 = graph.CreateAxis("z5", size2);
  auto z6 = graph.CreateAxis("z6", size2);
  auto z7 = graph.CreateAxis("z7", size2);
  auto z8 = graph.CreateAxis("z8", size2);
  auto z9 = graph.CreateAxis("z9", size2);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.repeats = {size2, size2, size2, size2, size2, size2, size2, size2, size2, size2};
  *load.y.strides = {ge::Symbol(512), ge::Symbol(256), ge::Symbol(128), ge::Symbol(64), ge::Symbol(32),
                     ge::Symbol(16),  ge::Symbol(8),   ge::Symbol(4),   ge::Symbol(2),  ge::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.repeats = {size2, One, size2, One, size2, One, size2, One, size2, One};
  *max.y.strides = {ge::Symbol(16), ge::ops::Zero, ge::Symbol(8), ge::ops::Zero, ge::Symbol(4),
                    ge::ops::Zero,  ge::Symbol(2), ge::ops::Zero, ge::ops::One,  ge::ops::Zero};
  max.attr.api.compute_type = ComputeType::kComputeReduce;

  AxisGroup axes_group;
  axes_group.y_group = {z0.id, z2.id, z4.id, z6.id, z8.id};
  axes_group.r_group = {z1.id, z3.id, z5.id, z7.id, z9.id};
  axes_group.axes_order = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z0.id;
  tiling_case.ub_tiling_id_r = z9.id;

  Scheduler schedule(graph, axes_group, tiling_case, true, optimize::ReduceTemplateType::kRCore);
  EXPECT_EQ(schedule.DoScheduler(), 0);

  std::vector<int64_t> golden_axis{15, 16, 14, 11, 2, 4, 6, 8, 13};
  std::vector<int64_t> golden_output{15, 16, 11, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13};
  std::vector<int64_t> golden_vectorized_axis{11, 2, 4, 6, 8, 13};

  auto load_node = graph.FindNode("load_i");
  EXPECT_EQ(load_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(load_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);

  auto max_node = graph.FindNode("max");
  EXPECT_EQ(max_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(max_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(max_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);
}

TEST_F(AutoSchedulerUT, reorder_vectorized_reduce_full_load) {
  ge::AscGraph graph("reorder_vectorized_axes");
  auto size2 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", size2);
  auto z1 = graph.CreateAxis("z1", size2);
  auto z2 = graph.CreateAxis("z2", size2);
  auto z3 = graph.CreateAxis("z3", size2);
  auto z4 = graph.CreateAxis("z4", size2);
  auto z5 = graph.CreateAxis("z5", size2);
  auto z6 = graph.CreateAxis("z6", size2);
  auto z7 = graph.CreateAxis("z7", size2);
  auto z8 = graph.CreateAxis("z8", size2);
  auto z9 = graph.CreateAxis("z9", size2);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.repeats = {size2, size2, size2, size2, size2, size2, size2, size2, size2, size2};
  *load.y.strides = {ge::Symbol(512), ge::Symbol(256), ge::Symbol(128), ge::Symbol(64), ge::Symbol(32),
                     ge::Symbol(16),  ge::Symbol(8),   ge::Symbol(4),   ge::Symbol(2),  ge::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.repeats = {size2, One, size2, One, size2, One, size2, One, size2, One};
  *max.y.strides = {ge::ops::Zero, ge::Symbol(16), ge::ops::Zero, ge::Symbol(8), ge::ops::Zero,
                    ge::Symbol(4), ge::ops::Zero,  ge::Symbol(2), ge::ops::Zero, ge::ops::One};
  max.attr.api.compute_type = ComputeType::kComputeReduce;

  AxisGroup axes_group;
  axes_group.y_group = {z1.id, z3.id, z5.id, z7.id, z9.id};
  axes_group.n_group = {z0.id, z2.id, z4.id, z6.id, z8.id};
  axes_group.axes_order = {1, 3, 5, 7, 9};

  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z5.id;

  Scheduler schedule(graph, axes_group, tiling_case, false, optimize::ReduceTemplateType::kAllLoad);
  EXPECT_EQ(schedule.DoScheduler(), 0);

  std::vector<int64_t> golden_axis{13, 14, 0, 2, 4, 6, 8, 11, 7, 9};
  std::vector<int64_t> golden_output{0, 1, 2, 3, 4, 10, 11, 6, 7, 8, 9};
  std::vector<int64_t> golden_vectorized_axis{0, 2, 4, 6, 8, 11, 7, 9};

  auto load_node = graph.FindNode("load_i");
  EXPECT_EQ(load_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(load_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);

  auto max_node = graph.FindNode("max");
  EXPECT_EQ(max_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(max_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(max_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);
}

TEST_F(AutoSchedulerUT, RemoveDuplicatedAxisFromGroup) {
  ge::AscGraph graph("reorder_vectorized_axes");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  AxisGroup axes_group;
  axes_group.x_group = {z0.id, z2.id, z4.id};
  axes_group.y_group = {z1.id, z3.id, z4.id};
  axes_group.axes_order = {0, 2, 4, 1, 3, 4};

  // remove x
  TilingCase case1;
  case1.ub_tiling_id_x = z0.id;
  case1.ub_tiling_id_y = z1.id;
  Scheduler sch1(graph, axes_group, case1);
  sch1.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x1 = {z0.id, z2.id};
  std::vector<int64_t> golden_y1 = {z1.id, z3.id, z4.id};
  std::vector<size_t> golden_order1 = {0, 2, 1, 3, 4};
  EXPECT_EQ(sch1.axes_group_.x_group, golden_x1);
  EXPECT_EQ(sch1.axes_group_.y_group, golden_y1);
  EXPECT_EQ(sch1.axes_group_.axes_order, golden_order1);

  TilingCase case2;
  case2.ub_tiling_id_x = z0.id;
  case2.ub_tiling_id_y = z4.id;
  Scheduler sch2(graph, axes_group, case2);
  sch2.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x2 = {z0.id, z2.id};
  std::vector<int64_t> golden_y2 = {z1.id, z3.id, z4.id};
  std::vector<size_t> golden_order2 = {0, 2, 1, 3, 4};
  EXPECT_EQ(sch2.axes_group_.x_group, golden_x2);
  EXPECT_EQ(sch2.axes_group_.y_group, golden_y2);
  EXPECT_EQ(sch2.axes_group_.axes_order, golden_order2);

  TilingCase case3;
  case3.ub_tiling_id_x = z4.id;
  case3.ub_tiling_id_y = z3.id;
  Scheduler sch3(graph, axes_group, case3);
  sch3.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x3 = {z0.id, z2.id, z4.id};
  std::vector<int64_t> golden_y3 = {z1.id, z3.id};
  std::vector<size_t> golden_order3 = {0, 2, 4, 1, 3};
  EXPECT_EQ(sch3.axes_group_.x_group, golden_x3);
  EXPECT_EQ(sch3.axes_group_.y_group, golden_y3);
  EXPECT_EQ(sch3.axes_group_.axes_order, golden_order3);
}

TEST_F(AutoSchedulerUT, AutoSchedulerUT_reorder_vectorized_axes_ok) {
  ge::AscGraph graph("reorder_vectorized_axes");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, ge::sym::kSymbolOne};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Max transpose("transpose");
  transpose.x = load.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, ge::sym::kSymbolOne};
  transpose.attr.api.compute_type = ComputeType::kComputeTranspose;

  ge::ascir_op::Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z1.id, z0.id};
  *store.y.repeats = {s1, s0};
  *store.y.strides = {s0, ge::sym::kSymbolOne};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule auto_schedule(graph, impl_graphs);

  auto_schedule.axes_group_.x_group = {z0.id};
  auto_schedule.axes_group_.y_group = {z1.id};
  auto_schedule.axes_group_.r_group = {-1};
  auto_schedule.axes_group_.axes_order = {0, 1};

  std::vector<TilingCase> tiling_cases;
  auto_schedule.GenTilingCase(tiling_cases);

  ASSERT_EQ(tiling_cases.size(), 1UL);
  Scheduler scheduler(graph, auto_schedule.axes_group_, tiling_cases[0UL]);
  EXPECT_EQ(scheduler.DoScheduler(), ge::SUCCESS);

  auto all_axis = graph.GetAllAxis();
  int64_t z0t_id = std::numeric_limits<int64_t>::max();
  int64_t z1t_id = std::numeric_limits<int64_t>::max();
  for (auto &axis : all_axis) {
    if (axis->name == "z0t") {
      z0t_id = axis->id;
    }
    if (axis->name == "z1t") {
      z1t_id = axis->id;
    }
  }

  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(load_node->attr.sched.axis[2], z0t_id);
  EXPECT_EQ(load_node->attr.sched.axis[3], z1t_id);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis[0], z0t_id);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis[1], z1t_id);

  auto transpose_node = graph.FindNode("transpose");
  ASSERT_NE(transpose_node, nullptr);
  EXPECT_EQ(transpose_node->attr.sched.axis[2], z1t_id);
  EXPECT_EQ(transpose_node->attr.sched.axis[3], z0t_id);
  EXPECT_EQ(transpose_node->outputs[0].attr.vectorized_axis[0], z1t_id);
  EXPECT_EQ(transpose_node->outputs[0].attr.vectorized_axis[1], z0t_id);

  auto store_node = graph.FindNode("store");
  ASSERT_NE(store_node, nullptr);
  EXPECT_EQ(store_node->attr.sched.axis[2], z1t_id);
  EXPECT_EQ(store_node->attr.sched.axis[3], z0t_id);
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_axis[0], z1t_id);
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_axis[1], z0t_id);
}

TEST_F(AutoSchedulerUT, DoTilingOk) {
  ge::AscGraph graph("apply_tiling_pk");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  ge::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.repeats = {s0, s1, s2, s3, s4};
  *load.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, ge::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *max.y.repeats = {s0, ge::ops::One, s2, One, s4};
  *max.y.strides = {s2 * s4, ge::ops::Zero, s4, Zero, ge::ops::One};
  max.attr.api.compute_type = ComputeType::kComputeReduce;

  ge::ascir_op::Store store("store");
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, ge::ops::One, s2, One, s4};
  *store.y.strides = {s2 * s4, ge::ops::Zero, s4, Zero, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.axes_group_.x_group = {z0.id};
  autoschedule.axes_group_.y_group = {z4.id};
  autoschedule.axes_group_.r_group = {z1.id, z3.id};
  autoschedule.axes_group_.n_group = {z2.id};
  autoschedule.axes_group_.axes_order = {0, 4, 1, 3};

  std::vector<TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  EXPECT_EQ(tiling_cases.size(), 2UL);

  Scheduler scheduler(graph, autoschedule.axes_group_, tiling_cases[0UL]);
  EXPECT_EQ(scheduler.DoScheduler(), 0);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_softmax_fusion_axesgroup) {
  ge::AscGraph graph("SoftmaxFusion");
  Construct_Softmax(graph);

  ge::AscGraph except_graph("SoftmaxFusion_B0Y0");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<ge::AxisId> axes = store->attr.sched.axis;
  std::vector<ge::AxisId> y_group = {axes[0]};
  std::vector<ge::AxisId> r_group = {axes[1]};

  AxisGroup axis_group;
  ASSERT_EQ(TilingGroup::GenTilingGroup(graph, axis_group), 0);
  EXPECT_EQ(axis_group.x_group.size(), 0);
  EXPECT_EQ(axis_group.y_group.size(), 1);
  EXPECT_EQ(axis_group.y_group, y_group);
  EXPECT_EQ(axis_group.r_group.size(), 1);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_softmax_fusion_tilingcase) {
  ge::AscGraph graph("SoftmaxFusion");
  Construct_Softmax(graph);

  ge::AscGraph except_graph("SoftmaxFusion_B0Y0");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<ge::AxisId> axes = store->attr.sched.axis;

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoschedule.axes_group_), 0);
  TilingGroup::NormGroup(autoschedule.axes_group_);
  std::vector<TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  ASSERT_EQ(tiling_cases.size(), 1);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_x, -1);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, axes[0]);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_r, axes[1]);
  EXPECT_EQ(tiling_cases[0].block_tiling_id, 0);
  EXPECT_EQ(tiling_cases[0].reduce_is_block, false);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_softmax_fusion) {
  ge::AscGraph graph("SoftmaxFusion");
  Construct_Softmax(graph);

  ge::AscGraph except_graph("SoftmaxFusion_B0Y0");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  auto dump_graph = utils::DebugStr(impl_graphs[0].scheduled_graph);

  EXPECT_TRUE(ScheduleUtils::IsReduceArFullLoad(impl_graphs[0].scheduled_graph));
  EXPECT_FALSE(ScheduleUtils::IsNormStruct(impl_graphs[0].scheduled_graph));
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_normal_struct) {
  ge::AscGraph graph("NormalStructFussion");
  Construct_Normal_Struct(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  auto dump_graph = utils::DebugStr(impl_graphs[0].scheduled_graph);
  EXPECT_TRUE(ScheduleUtils::IsNormStruct(impl_graphs[0].scheduled_graph));
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_scalar) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_scalar");
  Construct_ContinuesBroadcastScalarFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl0_brc4 = impl_graphs[0].scheduled_graph.FindNode("brc4");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[0].Str().get()), "(s0 / (z0Tb_size * z0t_size))");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[1].Str().get()), "z0Tb_size");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[2].Str().get()), "z0t_size");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[3].Str().get()), "s1");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[4].Str().get()), "s2");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[5].Str().get()), "s3");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[6].Str().get()), "s4");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl1_brc3 = impl_graphs[1].scheduled_graph.FindNode("brc3");
  // EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[0].Str().get()), "s0");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[1].Str().get()), "(s1 / (z1t_size))");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[2].Str().get()), "z1t_size");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[3].Str().get()), "s2");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[4].Str().get()), "s3");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[5].Str().get()), "s4");

  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl2_brc2 = impl_graphs[2].scheduled_graph.FindNode("brc2");
  // EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[0].Str().get()), "s0");
  // EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[1].Str().get()), "s1");
  EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[2].Str().get()), "(s2 / (z2t_size))");
  EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[3].Str().get()), "z2t_size");
  EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[4].Str().get()), "s3");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[5].Str().get()), "s4");

  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl3_brc1 = impl_graphs[3].scheduled_graph.FindNode("brc1");
  // EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[0].Str().get()), "s0");
  // EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[1].Str().get()), "s1");
  // EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[2].Str().get()), "s2");
  EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[3].Str().get()), "(s3 / (z3t_size))");
  EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[4].Str().get()), "z3t_size");
  EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[5].Str().get()), "s4");

  EXPECT_NE(impl_graphs[4].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl4_brc0 = impl_graphs[4].scheduled_graph.FindNode("brc0");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[0].Str().get()), "s0");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[1].Str().get()), "s1");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[2].Str().get()), "s2");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[3].Str().get()), "s3");
  EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[4].Str().get()), "(s4 / (z4t_size))");
  EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[5].Str().get()), "z4t_size");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_part_one) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_part_one");
  Construct_ContinuesBroadcastPartBrcFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc3"), nullptr);
  auto impl_grp_0_brc1 = impl_graphs[0].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_0_brc1, nullptr);
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_0_brc4 = impl_graphs[0].scheduled_graph.FindNode("brc4");
  EXPECT_NE(impl_grp_0_brc4, nullptr);
  EXPECT_EQ(impl_grp_0_brc4->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc4->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_1_brc1 = impl_graphs[1].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_1_brc1, nullptr);
  EXPECT_EQ(impl_grp_1_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_1_brc3 = impl_graphs[1].scheduled_graph.FindNode("brc3");
  EXPECT_NE(impl_grp_1_brc3, nullptr);
  EXPECT_EQ(impl_grp_1_brc3->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc3->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_2_brc1 = impl_graphs[2].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_2_brc1, nullptr);
  EXPECT_EQ(impl_grp_2_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_2_exp0 = impl_graphs[2].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_2_exp0, nullptr);
  EXPECT_EQ(impl_grp_2_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_3_brc1 = impl_graphs[3].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_3_brc1, nullptr);
  EXPECT_EQ(impl_grp_3_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_4_exp0 = impl_graphs[4].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_4_exp0, nullptr);
  EXPECT_EQ(impl_grp_4_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_interval_baba) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_interval_baba");
  Construct_ContinuesBroadcastIntervalBrcBABAFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 7);

  EXPECT_EQ(results[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[0].scheduled_graph.FindNode("brc1"), nullptr);
  auto impl_grp_0_brc1 = results[0].scheduled_graph.FindNode("brc1");
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_interval_abab) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_interval_abab");
  Construct_ContinuesBroadcastIntervalBrcABABFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 7);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  auto impl_grp_0_brc1 = impl_graphs[0].scheduled_graph.FindNode("brc1");
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_interval_baab) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_interval_baab");
  Construct_ContinuesBroadcastIntervalBrcBAABFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  auto impl_grp_0_brc0 = impl_graphs[0].scheduled_graph.FindNode("brc0");
  EXPECT_EQ(impl_grp_0_brc0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");

  auto impl_grp_0_brc1 = impl_graphs[0].scheduled_graph.FindNode("brc1");
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_part_one_in_middle) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_part_one_in_middle");
  Construct_ContinuesBroadcastPartBrcInMiddleFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_0_brc2 = impl_graphs[0].scheduled_graph.FindNode("brc2");
  EXPECT_NE(impl_grp_0_brc2, nullptr);
  EXPECT_EQ(impl_grp_0_brc2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_1_brc2 = impl_graphs[1].scheduled_graph.FindNode("brc2");
  EXPECT_NE(impl_grp_1_brc2, nullptr);
  EXPECT_EQ(impl_grp_1_brc2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_2_brc1 = impl_graphs[2].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_2_brc1, nullptr);
  EXPECT_EQ(impl_grp_2_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_2_exp0 = impl_graphs[2].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_2_exp0, nullptr);
  EXPECT_EQ(impl_grp_2_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_NE(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_3_brc0 = impl_graphs[3].scheduled_graph.FindNode("brc0");
  EXPECT_NE(impl_grp_3_brc0, nullptr);
  EXPECT_EQ(impl_grp_3_brc0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_brc0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_3_exp0 = impl_graphs[3].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_3_exp0, nullptr);
  EXPECT_EQ(impl_grp_3_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  auto impl_grp_3_add0 = impl_graphs[3].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_3_add0, nullptr);
  EXPECT_EQ(impl_grp_3_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_3_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");

  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_4_exp0 = impl_graphs[4].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_4_exp0, nullptr);
  EXPECT_EQ(impl_grp_4_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_4_add0 = impl_graphs[4].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_4_add0, nullptr);
  EXPECT_EQ(impl_grp_4_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_4_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_4_exp1 = impl_graphs[4].scheduled_graph.FindNode("exp1");
  EXPECT_NE(impl_grp_4_exp1, nullptr);
  EXPECT_EQ(impl_grp_4_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_multi_out) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_multi_out");
  Construct_ContinuesBroadcastMultiOutFusion(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 7);

  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_NE(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_with_reduce) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_with_reduce");
  Construct_ContinuesBroadcastWithReduceFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 4);

  EXPECT_EQ(results[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[0].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(results[0].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_NE(results[0].scheduled_graph.FindNode("brc4"), nullptr);

  EXPECT_EQ(results[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(results[1].scheduled_graph.FindNode("brc3"), nullptr);

  EXPECT_EQ(results[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[2].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_EQ(results[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[3].scheduled_graph.FindNode("brc1"), nullptr);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_redundant_broadcast) {
  ge::AscGraph graph("RemoveRedundantBroadcast");
  Construct_RedundantBroadcastFusion(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  // check don't remove brc
  auto impl_grp_0_exp1 = impl_graphs[0].scheduled_graph.FindNode("exp1");
  EXPECT_NE(impl_grp_0_exp1, nullptr);
  EXPECT_EQ(impl_grp_0_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");

  auto impl_grp_0_add0 = impl_graphs[0].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_0_add0, nullptr);
  EXPECT_EQ(impl_grp_0_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_0_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  EXPECT_EQ(impl_grp_0_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);

  // check remove brc
  auto impl_grp_1_exp1 = impl_graphs[1].scheduled_graph.FindNode("exp1");
  EXPECT_NE(impl_grp_1_exp1, nullptr);
  EXPECT_EQ(impl_grp_1_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  auto impl_grp_1_add0 = impl_graphs[1].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_1_add0, nullptr);
  EXPECT_EQ(impl_grp_1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  EXPECT_EQ(impl_grp_1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "exp0");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_gather_3axis) {
  ge::AscGraph graph("LoadGatherStore");
  Construct_LoadGatherStore(graph);

  ge::AscGraph except_graph("LoadGatherStore");
  except_graph.CopyFrom(graph);

  auto gather = graph.FindNode("gather");
  std::vector<ge::AxisId> y_group = gather->attr.sched.axis;

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.DoAutoSchedule();
  ASSERT_EQ(results.size(), 1UL);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_gather_param_is_1_axis) {
  ge::AscGraph graph("LoadGatherStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id};
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x2.y.dtype = ge::DT_INT32;
  x2.attr.sched.axis = {z1.id};
  *x2.y.axis = {z1.id};
  *x2.y.repeats = {s1};
  *x2.y.strides = {One};

  ge::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.api.compute_type = ComputeType::kComputeGather;
  gather.y.dtype = ge::DT_FLOAT;
  gather.attr.sched.axis = {z1.id};
  *gather.y.axis = {z1.id};
  *gather.y.repeats = {s1};
  *gather.y.strides = {One};
  gather.ir_attr.SetAxis(0);

  ge::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = gather.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT;
  store.attr.sched.axis = {z1.id};
  *store.y.axis = {z1.id};
  *store.y.repeats = {s1};
  *store.y.strides = {One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT;
  y.attr.sched.axis = {z1.id};
  *y.y.axis = {z1.id};
  *y.y.repeats = {s1};
  *y.y.strides = {One};

  ge::AscGraph except_graph("LoadGatherStore");
  except_graph.CopyFrom(graph);

  auto gather_node = graph.FindNode("gather");

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  auto status = schedule.DoAutoSchedule();
  ASSERT_EQ(status, ge::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_broadcast) {
  ge::AscGraph graph("Autoschedule_autoschedule_removepad_support_broadcast");
  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(3);
  auto s2 = graph.CreateSizeVar(3);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1, s2};
  *data1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  data2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data2.y.dtype = ge::DT_FLOAT16;
  *data2.y.axis = {z0.id, z1.id, z2.id};
  *data2.y.repeats = {One, s1, s2};
  *data2.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, s1, s2};
  *load2.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = load2.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {s0, s1, s2};
  *brc2.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = load1.y;
  mul0.x2 = brc2.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store1("store1");
  store1.x = mul0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};

  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "Autoschedule_autoschedule_removepad_support_broadcast_B0Y3");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(),
            "Autoschedule_autoschedule_removepad_support_broadcast_B0Y0_unaligned");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(),
            "Autoschedule_autoschedule_removepad_support_broadcast_B0Y0_inline");

  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc0_remove_pad_0"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc2_remove_pad_0"), nullptr);
  EXPECT_EQ(AscGraphUtils::GetComputeGraph(impl_graphs[1].scheduled_graph)->GetAllNodesSize(), 16);
  const auto &impl1_remove_pad0 = impl_graphs[1].scheduled_graph.FindNode("brc0_remove_pad_0");
  EXPECT_EQ(impl1_remove_pad0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl1_remove_pad0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  EXPECT_EQ(impl1_remove_pad0->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "add0");
  const auto &impl1_remove_pad2 = impl_graphs[1].scheduled_graph.FindNode("brc2_remove_pad_0");
  EXPECT_EQ(impl1_remove_pad2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl1_remove_pad2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc2");
  EXPECT_EQ(impl1_remove_pad2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "mul0");

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0_remove_pad_0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc2_remove_pad_0"), nullptr);
  EXPECT_EQ(AscGraphUtils::GetComputeGraph(impl_graphs[0].scheduled_graph)->GetAllNodesSize(), 12);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_not_support_reduce) {
  ge::AscGraph graph("Autoschedule_autoschedule_remove_pad_not_support_reduce");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  ge::ascir_op::Sum sum0("reduce0");
  sum0.x = load0.y;
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id};
  sum0.attr.api.compute_type = ComputeType::kComputeReduce;
  sum0.y.dtype = ge::DT_FLOAT16;
  *sum0.y.axis = {z0.id, z1.id, z2.id};
  *sum0.y.repeats = {One, s1, One};
  *sum0.y.strides = {Zero, One, Zero};

  ge::ascir_op::Store store0("store0");
  store0.x = sum0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {One, s1, One};
  *store0.y.strides = {Zero, One, Zero};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {One, s1, One};
  *y0.y.strides = {Zero, One, Zero};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  for (const auto &impl_graph : impl_graphs) {
    EXPECT_EQ(AscGraphUtils::GetComputeGraph(impl_graph.scheduled_graph)->GetAllNodesSize(), 5);
  }
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_not_support_dtype) {
  ge::AscGraph graph("Autoschedule_autoschedule_removepad_not_support_dtype");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_INT64;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_INT64;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_INT64;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store0("store0");
  store0.x = brc0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_INT64;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_INT64;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  for (const auto &impl_graph : impl_graphs) {
    EXPECT_EQ(impl_graph.scheduled_graph.FindNode("brc0_remove_pad"), nullptr);
  }
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_load_slice) {
  ge::AscGraph graph("removepad_support_load_slice");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto offset = Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2 * offset, s2 * offset, One};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id, z2.id};
  *abs0.y.repeats = {s0, s1, s2};
  *abs0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store0("store0");
  store0.x = abs0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 3);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(results[2].scheduled_graph)->GetAllNodesSize(), 6);
  const auto &remove_pad = results[2].scheduled_graph.FindNode("load0_remove_pad_0");
  EXPECT_NE(remove_pad, nullptr);
  EXPECT_EQ(remove_pad->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(remove_pad->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(remove_pad->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_load_slice_inductor) {
  ge::AscGraph graph("removepad_support_load_slice_inductor)");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto offset = Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2 * s0, s2 * s0, s0};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id, z2.id};
  *abs0.y.repeats = {s0, s1, s2};
  *abs0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store0("store0");
  store0.x = abs0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 3);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(results[2].scheduled_graph)->GetAllNodesSize(), 6);
  const auto &remove_pad = results[2].scheduled_graph.FindNode("load0_remove_pad_0");
  EXPECT_NE(remove_pad, nullptr);
  EXPECT_EQ(remove_pad->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(remove_pad->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(remove_pad->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_concat) {
  ge::AscGraph graph("Autoschedule_autoschedule_removepad_support_concat");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = ge::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = ge::DT_FLOAT16;
  x.ir_attr.SetIndex(0);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT16;
  x1.ir_attr.SetIndex(1);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, One};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, One};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id};
  output0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT16;
  output0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load3("load3");
  load3.x = output0.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};
  load3.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Store store1("store1");
  store1.x = load3.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};
  store1.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store1.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_concat_all_input_cols_aligned) {
  ge::AscGraph graph("Autoschedule_autoschedule_removepad_support_concat");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = ge::Symbol(128);
  auto s2 = ge::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  x.ir_attr.SetIndex(0);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2, s2, One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Concat concat("concat");
  concat.x = {load.y, load.y};
  concat.attr.sched.axis = {z0.id, z1.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, z2.id};
  *concat.y.repeats = {s0, s1 + s1, s2};
  *concat.y.strides = {(s1 + s1) * s2, s2, One};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1 + s1, s2};
  *store.y.strides = {(s1 + s1) * s2, s2, One};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id, z2.id};
  output0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT16;
  output0.ir_attr.SetIndex(0);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_NE(impl_graphs[0].scheduled_graph.GetName().find("unaligned"), std::string::npos);
}

/**
 *                 add0
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1,s2)     |(1,s1,s2)
 *     load0           load1
 *       |               |
 *     data0           data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_brc_inline_static_support) {
  ge::AscGraph graph("Autoschedule_autoschedule_brc_inline_static_support");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(AscGraphUtils::GetComputeGraph(impl_graphs[1].scheduled_graph)->GetAllNodesSize(), 7);
  const auto &impl1_add0 = impl_graphs[1].scheduled_graph.FindNode("add0");
  EXPECT_EQ(impl1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(impl1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load1");

  EXPECT_EQ("Autoschedule_autoschedule_brc_inline_static_support_B0Y0_inline", impl_graphs[1].scheduled_graph.GetName());
}

/**
 *                 add0
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1,s2)     |(1,s1,s2)
 *     load0           load1
 *       |               |
 *     data0           data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_brc_inline_dynamic_support) {
  ge::AscGraph graph("Autoschedule_autoschedule_brc_inline_dynamic_support");
  auto s0 = ge::Symbol("20");
  auto s1 = ge::Symbol("32");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);

  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(AscGraphUtils::GetComputeGraph(impl_graphs[3].scheduled_graph)->GetAllNodesSize(), 7);
  const auto &impl1_add0 = impl_graphs[3].scheduled_graph.FindNode("add0");
  EXPECT_EQ(impl1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(impl1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load1");
}

/**
 *                 eq
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1,s2)     |(1,s1,s2)
 *     load0           load1
 *       |               |
 *     data0           data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_brc_inline_static_not_support) {
  ge::AscGraph graph("Autoschedule_autoschedule_brc_inline_static_not_support");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Eq eq0("eq0");
  eq0.x1 = load0.y;
  eq0.x2 = brc1.y;
  eq0.attr.sched.axis = {z0.id, z1.id};
  eq0.attr.api.compute_type = ComputeType::kComputeElewise;
  eq0.y.dtype = ge::DT_FLOAT16;
  *eq0.y.axis = {z0.id, z1.id};
  *eq0.y.repeats = {s0, s1};
  *eq0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = eq0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  EXPECT_EQ("Autoschedule_autoschedule_brc_inline_static_not_support_B0Y1", impl_graphs[1].scheduled_graph.GetName());
}

TEST_F(AutoSchedulerUT, test_vectorized_reduce_ra) {
  ge::AscGraph graph("test");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Max max("max");
  max.x = arg4_1.y;
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  max.attr.api.type = ge::ApiType::kAPITypeCompute;
  max.y.dtype = ge::DT_FLOAT;

  AxisGroup axis_group;
  axis_group.y_group = {z1.id};
  axis_group.n_group = {z0.id};
  axis_group.axes_order = {1};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z1.id;
  tiling_case.ub_tiling_y = graph.TileSplit(z1.id, "z1T", "z1t");
  Scheduler scheduler(graph, axis_group, tiling_case, false);
  std::vector<int64_t> vectorized_axes;
  std::vector<size_t> vectorized_axes_order;
  scheduler.FindVectorizedAxes(vectorized_axes, vectorized_axes_order);

  std::vector<int64_t> golden_axes{3, 0};
  std::vector<size_t> golden_axes_order{4, 1};
  EXPECT_EQ(vectorized_axes, golden_axes);
  EXPECT_EQ(vectorized_axes_order, golden_axes_order);
}

TEST_F(AutoSchedulerUT, test_vectorized_reduce_ara) {
  ge::AscGraph graph("test");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(32);
  auto s2 = ge::Symbol(20);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Max max("max");
  max.x = arg4_1.y;
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  max.attr.api.type = ge::ApiType::kAPITypeCompute;
  max.y.dtype = ge::DT_FLOAT;

  AxisGroup axis_group;
  axis_group.y_group = {z0.id, z2.id};
  axis_group.n_group = {z1.id};
  axis_group.axes_order = {0, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z0.id;
  tiling_case.ub_tiling_y = graph.TileSplit(z0.id, "z0T", "z0t");
  Scheduler scheduler(graph, axis_group, tiling_case, false);
  std::vector<int64_t> vectorized_axes;
  std::vector<size_t> vectorized_axes_order;
  scheduler.FindVectorizedAxes(vectorized_axes, vectorized_axes_order);

  std::vector<int64_t> golden_axes{4, 2, 1};
  std::vector<size_t> golden_axes_order{5, 7, 2};
  EXPECT_EQ(vectorized_axes, golden_axes);
  EXPECT_EQ(vectorized_axes_order, golden_axes_order);
}

/**
 *                 add
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1)        |(s0,1)
 *      brc0           load1
 *       |               |
 *     scalar          data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_node_cache_marker_broadcast) {
  ge::AscGraph graph("Autoschedule_node_cache_marker_broadcast");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Scalar data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = data0.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, Zero};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {One, s1};
  *store0.y.strides = {Zero, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};

  // 验证防止重复判断
  ge::ascir_op::Add add1("add1");
  add1.x1 = brc0.y;
  add1.x2 = brc1.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.attr.api.compute_type = ComputeType::kComputeElewise;
  add1.y.dtype = ge::DT_FLOAT16;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, One};

  ge::ascir_op::Store store1("store1");
  store1.x = add0.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {One, s1};
  *store1.y.strides = {Zero, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(0);
  y1.x = store0.y;
  y1.attr.sched.axis = {z0.id, z1.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  const auto &impl0 = impl_graphs[0];
  const auto &impl0_scalar_node = impl0.scheduled_graph.FindNode("data0");
  EXPECT_NE(impl0_scalar_node, nullptr);
  EXPECT_EQ(impl0_scalar_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl0_brc0_node = impl0.scheduled_graph.FindNode("brc0");
  EXPECT_NE(impl0_brc0_node, nullptr);
  EXPECT_EQ(impl0_brc0_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_brc1_node = impl0.scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl0_brc1_node, nullptr);
  EXPECT_EQ(impl0_brc1_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl0_add0_node = impl0.scheduled_graph.FindNode("add0");
  EXPECT_NE(impl0_add0_node, nullptr);
  EXPECT_EQ(impl0_add0_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);

  const auto &impl1 = impl_graphs[1];
  const auto &impl1_scalar_node = impl1.scheduled_graph.FindNode("data0");
  EXPECT_NE(impl1_scalar_node, nullptr);
  EXPECT_EQ(impl1_scalar_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl1_brc0_node = impl1.scheduled_graph.FindNode("brc0");
  EXPECT_NE(impl1_brc0_node, nullptr);
  EXPECT_EQ(impl1_brc0_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_brc1_node = impl1.scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl1_brc1_node, nullptr);
  EXPECT_EQ(impl1_brc1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_load1_node = impl1.scheduled_graph.FindNode("load1");
  EXPECT_NE(impl1_load1_node, nullptr);
  EXPECT_EQ(impl1_load1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_add0_node = impl1.scheduled_graph.FindNode("add0");
  EXPECT_NE(impl1_add0_node, nullptr);
  EXPECT_EQ(impl1_add0_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
}

/**
 *                 add
 *              /      \
 *            /      removepad
 *          /            \
 *        /             brc1
 *       |(s0,s1)        |(1,s1)
 *      load0          load1
 *       |              |
 *     data0          data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_node_cache_marker_removepad) {
  ge::AscGraph graph("Autoschedule_node_cache_marker_removepad");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::RemovePad remove_pad("remove_pad");
  remove_pad.x = brc1.y;
  remove_pad.attr.sched.axis = {z0.id, z1.id};
  remove_pad.attr.api.compute_type = ComputeType::kComputeElewise;
  remove_pad.y.dtype = ge::DT_FLOAT16;
  *remove_pad.y.axis = {z0.id, z1.id};
  *remove_pad.y.repeats = {s0, s1};
  *remove_pad.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = remove_pad.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  const auto &impl0 = impl_graphs[0];
  const auto &impl0_scalar_node = impl0.scheduled_graph.FindNode("data0");
  EXPECT_NE(impl0_scalar_node, nullptr);
  EXPECT_EQ(impl0_scalar_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl0_load1_node = impl0.scheduled_graph.FindNode("load1");
  EXPECT_NE(impl0_load1_node, nullptr);
  EXPECT_EQ(impl0_load1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_brc1_node = impl0.scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl0_brc1_node, nullptr);
  EXPECT_EQ(impl0_brc1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_remove_pad_node = impl0.scheduled_graph.FindNode("remove_pad");
  EXPECT_NE(impl0_remove_pad_node, nullptr);
  EXPECT_EQ(impl0_remove_pad_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_add0_node = impl0.scheduled_graph.FindNode("add0");
  EXPECT_NE(impl0_add0_node, nullptr);
  EXPECT_EQ(impl0_add0_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);

  const auto &impl1 = impl_graphs[1];
  for (const auto &node : impl1.scheduled_graph.GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  }
}

TEST_F(AutoSchedulerUT, AutoSchedule_cube_gen_tilingGroup) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  MatMulOffset matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoSchedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoSchedule.axes_group_), 0);
  EXPECT_EQ(autoSchedule.axes_group_.x_group.size(), 0UL);
  EXPECT_EQ(autoSchedule.axes_group_.y_group.size(), 2UL);
  EXPECT_EQ(autoSchedule.axes_group_.r_group.size(), 0UL);
}
}  // namespace optimize