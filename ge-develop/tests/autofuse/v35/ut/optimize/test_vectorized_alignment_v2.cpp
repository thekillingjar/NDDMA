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

#include "gtest/gtest.h"
#include "ascendc_ir_def.h"
#define private public
#include "autoschedule/alignment_handler.h"
#undef private
#include "ascir_ops_utils.h"
#include "runtime_stub.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "platformv2.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;
class VectorizedAlignmentUTV2 : public testing::Test {
 protected:
  void SetUp() override {
    optimize::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(VectorizedAlignmentUTV2, brc_align_input_not_align) {
  ge::AscGraph graph("Concat");
  auto s0 = ge::Symbol(999);
  auto s1 = ge::Symbol(10);
  auto s2 = ge::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, One};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s2};
  *load1.y.strides = {One, Zero};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Concat concat("concat");
  concat.x = {load1.y, brc.y};
  concat.attr.api.compute_type = ge::ComputeType::kComputeConcat;
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + s2};
  *concat.y.strides = {s1 + s2, One};
  *concat.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + s2};
  *store.y.strides = {s1 + s2 + s0, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  auto brc_node = graph.FindNode("brc");
  ASSERT_NE(brc_node, nullptr);
  auto concat_node = graph.FindNode("concat");
  ASSERT_NE(concat_node, nullptr);

  std::vector<Expression> golden_strides_load1 = {One, Zero};
  std::vector<Expression> golden_strides_load0 = {One, Zero};
  std::vector<Expression> golden_strides_brc = {s1 + s2, One};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden_strides_load1);
  EXPECT_EQ(concat_node->outputs[0].attr.vectorized_strides, golden_strides_brc);
}

TEST_F(VectorizedAlignmentUTV2, discontinuous_write_not_align) {
  ge::AscGraph graph("Concat");
  auto s0 = ge::Symbol(10);
  auto s1 = ge::Symbol(1);
  auto stride_val = ge::Symbol(66);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Store store("store");
  store.x = load0.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {stride_val, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), ge::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto stode_node = graph.FindNode("store");
  ASSERT_NE(stode_node, nullptr);

  std::vector<Expression> golden_strides = {s1, Zero};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides);
  EXPECT_EQ(stode_node->outputs[0].attr.vectorized_strides, golden_strides);
}

TEST_F(VectorizedAlignmentUTV2, align_vectorized_strides) {
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

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), SUCCESS);
  auto abs_node = graph.FindNode("abs");

  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[0].Str().get()), "s1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[2].Str().get()), "0");
}

TEST_F(VectorizedAlignmentUTV2, align_vectorized_strides_last_zero) {
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

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), SUCCESS);

  auto load_node = graph.FindNode("load_i");
  std::vector<Expression> golden_stride = {Symbol(20), One, Zero};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, reduce_alignment) {
  ge::AscGraph graph("reduce_align");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(9);
  auto s3 = ge::Symbol(10);

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
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Max max("max");
  max.y.dtype = ge::DT_FLOAT16;
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, One, s3};
  *max.y.strides = {s1 * s3, s3, Zero, One};
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, One, s3};
  *store.y.strides = {s1 * s3, s3, Zero, One};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), SUCCESS);

  auto s16 = Symbol(16);
  std::vector<Expression> load_golden_stride = {s2 * s16, s16, One};
  std::vector<Expression> golden_stride = {s16, Zero, One};

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, load_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, align_vectorized_strides_by_repeat) {
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
  *load1.y.repeats = {s0, One};
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
  ::ascir::utils::DumpImplGraphs({graph}, "xxx");

  auto load1_node = graph.FindNode("load1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "9");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(VectorizedAlignmentUTV2, tail_brc_tail_reduce_alignment) {
  ge::AscGraph graph("tail_brc_tail_reduce");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(9);
  auto s3 = ge::Symbol(10);

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
  *load.y.repeats = {s0, s1, s2, One};
  *load.y.strides = {s1 * s2, s2, One, Zero};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Broadcast brc("brc");
  brc.y.dtype = ge::DT_FLOAT16;
  brc.x = load.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.repeats = {s0, s1, s2, s3};
  *brc.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  brc.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = brc.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  *abs.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Max max("max");
  max.x = brc.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, s2, One};
  *max.y.strides = {s1 * s2, s2, One, Zero};
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Max max1("max1");
  max1.x = brc.y;
  max1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max1.y.repeats = {s0, s1, s2, One};
  *max1.y.strides = {s1 * s2, s2, One, Zero};
  max1.attr.api.compute_type = ComputeType::kComputeReduce;
  *max1.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ascir_op::Add add("add");
  add.x1 = max.y;
  add.x2 = max1.y;
  add.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add.y.repeats = {s0, s1, s2, One};
  *add.y.strides = {s1 * s2, s2, One, Zero};
  *add.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = add.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, One};
  *store.y.strides = {s1 * s2, s2, One, Zero};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), SUCCESS);

  auto s16 = Symbol(16);
  std::vector<Expression> load_golden_stride = {s2, One, Zero};
  std::vector<Expression> brc_golden_stride = {s2 * s16, s16, One};
  std::vector<Expression> max_golden_stride = {s2, One, Zero};

  auto load_node = graph.FindNode("load");  // (9, 1, 0)
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, load_golden_stride);

  auto brc_node = graph.FindNode("brc");  // (9 * 16, 16, 1)
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, brc_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, max_golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, max_golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, scalar_brc_tail_reduce_alignment) {
  ge::AscGraph graph("scalar_brc_tail_reduce");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(9);
  auto s3 = ge::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ascir_op::Scalar scalar0("data0", graph);
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = ge::DT_FLOAT16;
  scalar0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Broadcast brc("brc");
  brc.y.dtype = ge::DT_FLOAT16;
  brc.x = scalar0.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.repeats = {s0, s1, s2, s3};
  *brc.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  brc.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Max max("max");
  max.x = brc.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, s2, One};
  *max.y.strides = {s1 * s2, s2, One, Zero};
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  ge::ascir_op::Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, One};
  *store.y.strides = {s1 * s2, s2, One, Zero};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), SUCCESS);

  auto s16 = Symbol(16);
  std::vector<Expression> load_golden_stride = {s2, One, Zero};
  std::vector<Expression> brc_golden_stride = {s2 * s16, s16, One};
  std::vector<Expression> max_golden_stride = {s2, One, Zero};

  auto brc_node = graph.FindNode("brc");  // (9 * 16, 16, 1)
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, brc_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, max_golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, max_golden_stride);
}
}  // namespace optimize
