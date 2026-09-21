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
#include "task_generator/recompute_case_generator.h"
#include "tests/autofuse/ut/optimize/easy_graph/easy_asc_graph.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

class RecomputeCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

TEST_F(RecomputeCaseGeneratorTest, TestDynamicGraphSplit) {
  ge::AscGraph graph("brc_abs");
  Data data0("RE_data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load("RE_load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  Abs abs("RE_abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  Broadcast brc1("brc1");
  brc1.x = abs.y;
  brc1.y.dtype = ge::DT_FLOAT16;

  Store store("store");
  store.x = brc1.y;
  store.y.dtype = ge::DT_UINT64;

  Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  Abs abs2("ST_abs2");
  abs2.x = abs.y;
  abs2.y.dtype = ge::DT_FLOAT16;

  Store store1("ST_store1");
  store1.x = abs2.y;
  store1.y.dtype = ge::DT_UINT64;

  Output out1("ST_out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  auto eg = ge::EaseAscGraph(graph)
                .Loops({ge::Symbol("s0"), ge::Symbol("s1"), Symbol("s2"), ge::Symbol("s3"), Symbol("s4")})
                .Broadcast("brc1", {0, 1, 2});
  eg.Build();

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestDynamicGraphSplitTwoLine) {
  ge::AscGraph graph("brc_abs");
  Data data0("RE_data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load("RE_load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  Abs abs("RE_abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  Broadcast brc1("brc1");
  brc1.x = abs.y;
  brc1.y.dtype = ge::DT_FLOAT16;

  Store store("store");
  store.x = brc1.y;
  store.y.dtype = ge::DT_UINT64;

  Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  Abs abs2("ST_abs2");
  abs2.x = abs.y;
  abs2.y.dtype = ge::DT_FLOAT16;

  Store store1("ST_store1");
  store1.x = abs2.y;
  store1.y.dtype = ge::DT_UINT64;

  Output out1("ST_out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  Abs abs3("ST_abs3");
  abs3.x = abs2.y;
  abs3.y.dtype = ge::DT_FLOAT16;

  Abs abs4("ST_abs4");
  abs4.x = abs3.y;
  abs4.y.dtype = ge::DT_FLOAT16;

  Abs abs5("ST_abs5");
  abs5.x = abs4.y;
  abs5.y.dtype = ge::DT_FLOAT16;

  Abs abs6("ST_abs6");
  abs6.x = abs5.y;
  abs6.y.dtype = ge::DT_FLOAT16;

  Store store2("ST_store2");
  store2.x = abs6.y;
  store2.y.dtype = ge::DT_UINT64;

  Output out2("ST_out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(2);

  auto eg = ge::EaseAscGraph(graph)
                .Loops({ge::Symbol("s0"), ge::Symbol("s1"), Symbol("s2"), ge::Symbol("s3"), Symbol("s4")})
                .Broadcast("brc1", {0, 1, 2});
  eg.Build();

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestStaticGraphSplitTwoLine) {
  ge::AscGraph graph("brc_abs");
  Data data0("RE_data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load("RE_load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  Abs abs("RE_abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  Broadcast brc1("brc1");
  brc1.x = abs.y;
  brc1.y.dtype = ge::DT_FLOAT16;

  Store store("store");
  store.x = brc1.y;
  store.y.dtype = ge::DT_UINT64;

  Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  Abs abs2("ST_abs2");
  abs2.x = abs.y;
  abs2.y.dtype = ge::DT_FLOAT16;

  Store store1("ST_store1");
  store1.x = abs2.y;
  store1.y.dtype = ge::DT_UINT64;

  Output out1("ST_out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  Abs abs3("ST_abs3");
  abs3.x = abs2.y;
  abs3.y.dtype = ge::DT_FLOAT16;

  Abs abs4("ST_abs4");
  abs4.x = abs3.y;
  abs4.y.dtype = ge::DT_FLOAT16;

  Abs abs5("ST_abs5");
  abs5.x = abs4.y;
  abs5.y.dtype = ge::DT_FLOAT16;

  Abs abs6("ST_abs6");
  abs6.x = abs5.y;
  abs6.y.dtype = ge::DT_FLOAT16;

  Store store2("ST_store2");
  store2.x = abs6.y;
  store2.y.dtype = ge::DT_UINT64;

  Output out2("ST_out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(2);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(1024), Symbol(256)}).Broadcast("brc1", {0});
  eg.Build();

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestStaticGraphSplitWithBrc) {
  ge::AscGraph graph("brc_abs");
  Data data0("RE_data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load("RE_load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  Abs abs("RE_abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  Broadcast brc1("brc1");
  brc1.x = abs.y;
  brc1.y.dtype = ge::DT_FLOAT16;

  Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.y.dtype = ge::DT_FLOAT16;

  Relu relu0("relu0");
  relu0.x = brc2.y;
  relu0.y.dtype = ge::DT_FLOAT16;

  Relu relu1("relu1");
  relu1.x = relu0.y;
  relu1.y.dtype = ge::DT_FLOAT16;

  Relu relu2("relu2");
  relu2.x = relu1.y;
  relu2.y.dtype = ge::DT_FLOAT16;

  Store store("store");
  store.x = relu2.y;
  store.y.dtype = ge::DT_UINT64;

  Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  Abs abs2("ST_abs2");
  abs2.x = brc1.y;
  abs2.y.dtype = ge::DT_FLOAT16;

  Exp exp("ST_exp");
  exp.x = brc1.y;
  exp.y.dtype = ge::DT_FLOAT16;

  Add add("ST_add");
  add.x1 = abs2.y;
  add.x2 = exp.y;
  add.y.dtype = ge::DT_FLOAT16;

  Store store1("ST_store1");
  store1.x = add.y;
  store1.y.dtype = ge::DT_UINT64;

  Output out1("ST_out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  Abs abs3("ST_abs3");
  abs3.x = abs2.y;
  abs3.y.dtype = ge::DT_FLOAT16;

  Abs abs4("ST_abs4");
  abs4.x = abs3.y;
  abs4.y.dtype = ge::DT_FLOAT16;

  Abs abs5("ST_abs5");
  abs5.x = abs4.y;
  abs5.y.dtype = ge::DT_FLOAT16;

  Abs abs6("ST_abs6");
  abs6.x = abs5.y;
  abs6.y.dtype = ge::DT_FLOAT16;

  Store store2("ST_store2");
  store2.x = abs6.y;
  store2.y.dtype = ge::DT_UINT64;

  Output out2("ST_out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(2);

  auto eg = ge::EaseAscGraph(graph)
                .Loops({ge::Symbol(1024), Symbol(32), Symbol(32)})
                .Broadcast("brc1", {1})
                .Broadcast("brc2", {0});
  eg.Build();

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}
}  // namespace schedule
