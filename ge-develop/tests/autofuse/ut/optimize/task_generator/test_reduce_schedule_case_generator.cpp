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
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "asc_graph_utils.h"
#include "ascir_ops.h"
#include "task_generator/schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

class ReduceScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

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
void ConstructNormStruct3Elewise(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

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
  sum.attr.api.compute_type = ge::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Exp exp("exp");
  exp.x = abs.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.attr.api.compute_type = ge::ComputeType::kComputeElewise;
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
  store_op1.attr.api.compute_type = ge::ComputeType::kComputeStore;
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
  auto z1 = graph.CreateAxis("z1", s0);

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
  sum.attr.api.compute_type = ge::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = ge::ComputeType::kComputeStore;
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

void ConstructNormStructMultiplyCitations(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

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

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {s1 ,ge::ops::One};
  *load1.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = ge::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Add add("add");
  add.x1 = sum.y;
  add.x2 = load1.y;
  add.attr.sched.axis = {z0.id, z1.id};
  add.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {s1, One};
  *add.y.repeats =  {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

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
  sum.attr.api.compute_type = ge::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {ge::ops::One, ge::ops::One};
  *sum.y.strides = {ge::ops::Zero, ge::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *abs.y.repeats =  {ge::ops::One, ge::ops::One};

  Tanh tanh("tanh");
  tanh.x = abs.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  tanh.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *tanh.y.repeats =  {ge::ops::One, ge::ops::One};

  Exp exp("exp");
  exp.x = tanh.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.attr.api.compute_type = ge::ComputeType::kComputeElewise;
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
  store_op1.attr.api.compute_type = ge::ComputeType::kComputeStore;
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

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Three_Elewise_Store) {
  ge::AscGraph graph("reduce_Three_elewise_store");
  ConstructNormStruct3Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}


TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_One_Elewise_Store) {
  ge::AscGraph graph("reduce_one_elewise_store");
  ConstructNormStruct1Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Four_Elewise_Store) {
  ge::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 3UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_Store) {
  ge::AscGraph graph("reduce_multi_citation_store");
  ConstructNormStructMultiplyCitations(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 3UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 4UL);
}
}