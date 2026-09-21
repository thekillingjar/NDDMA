/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "e2e_constant_load_gt_store.h"
#include "ascgraph_info_complete.h"
#include "gtest/gtest.h"
#include "ascir_utils.h"
#include "optimize.h"

using ascir::utils::DebugHintGraphStr;
using ascir::utils::DebugImplGraphStr;

class E2E_ConstantLoadGTStore: public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;

  E2E_ConstantLoadGTStore(): optimizer(optimize::OptimizerOptions{}) {};
};

TEST_F(E2E_ConstantLoadGTStore, GetApiInfo) {
  ge::AscGraph test_graph("test_constant_load_gt_store");
  ConstantLoadGtStore_BeforeAutofuse(test_graph);

  ge::AscGraph test_impl_graph("test_constant_load_gt_store");
  test_impl_graph.CopyFrom(test_graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(test_impl_graph);

  ge::AscGraph expect_impl_graph("test_constant_load_gt_store");
  expect_impl_graph.CopyFrom(test_graph);
  ConstantLoadGtStore_AfterGetApiInfo(expect_impl_graph);

  EXPECT_EQ(DebugImplGraphStr(test_impl_graph), DebugImplGraphStr(expect_impl_graph));
}

TEST_F(E2E_ConstantLoadGTStore, AfterSchedule) {
  GTEST_SKIP() << "Need support scheduler with constant.";
  ge::AscGraph test_graph("test_constant_load_gt_store");
  ConstantLoadGtStore_BeforeAutofuse(test_graph);
  ConstantLoadGtStore_AfterInferOutput(test_graph);

  ge::AscGraph test_impl_graph("test_constant_load_gt_store");
  test_impl_graph.CopyFrom(test_graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(test_impl_graph);

  ge::AscGraph expect_impl_graph("test_constant_load_gt_store");
  expect_impl_graph.CopyFrom(test_impl_graph);
  ConstantLoadGtStore_AfterScheduler(expect_impl_graph);

  std::vector<ge::AscGraph> test_sched_graphs;
  optimizer.AutoScheduler(test_graph, test_impl_graph, test_sched_graphs);

  EXPECT_EQ(DebugImplGraphStr(test_sched_graphs[0]), DebugImplGraphStr(expect_impl_graph));
}
