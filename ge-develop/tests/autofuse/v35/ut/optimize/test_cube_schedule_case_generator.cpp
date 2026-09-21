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
#include "task_generator/cube_schedule_case_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

class CubeScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    setenv("DUMP_GE_GRAPH", "2", false);
    setenv("DUMP_GRAPH_LEVEL", "1", false);
    setenv("DUMP_GRAPH_PATH", "/home/tl", false);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

void ConstructJustMatMul(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
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

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.attr.api.compute_type = ComputeType::kComputeCube;
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
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
  store_op.attr.api.compute_type = ComputeType::kComputeStore;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructMatMulAndAdd(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
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
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.attr.api.compute_type = ComputeType::kComputeCube;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructJustMatMulBias(AscGraph &graph) {
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

  MatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.attr.api.compute_type = ComputeType::kComputeCube;
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
  store_op.attr.api.compute_type = ComputeType::kComputeStore;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(CubeScheduleCaseGeneratorTest, Test_Just_Matmul_Store) {
  ge::AscGraph graph("just_matmul");
  ConstructJustMatMul(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}


TEST_F(CubeScheduleCaseGeneratorTest, Test_MatMul_Add_Store) {
  ge::AscGraph graph("matmul_add_store");
  ConstructMatMulAndAdd(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(CubeScheduleCaseGeneratorTest, Test_MatMul_Bias_Store) {
  ge::AscGraph graph("matmul_bias_store");
  ConstructJustMatMulBias(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}
}