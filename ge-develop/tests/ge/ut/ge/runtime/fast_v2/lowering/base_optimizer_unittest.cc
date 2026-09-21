/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/base_optimizer.h"
#include <gtest/gtest.h>
namespace gert {
namespace {
class AlwaysSuccessPass : public bg::Pass {
 public:
  ge::graphStatus Run(ge::ExecuteGraph *graph, bool &changed) override {
    return ge::GRAPH_SUCCESS;
  }
};
class AlwaysChangedPass : public bg::Pass {
 public:
  ge::graphStatus Run(ge::ExecuteGraph *graph, bool &changed) override {
    changed = true;
    return ge::GRAPH_SUCCESS;
  }
};
}  // namespace
class BaseOptimizerUT : public testing::Test {};
TEST_F(BaseOptimizerUT, Run_NotChanged_WhenAllPassesNotChanged) {
  bg::BaseOptimizer optimizer;
  optimizer.AddPass("Bar1", std::unique_ptr<bg::Pass>(new AlwaysSuccessPass));
  optimizer.AddPass("Bar2", std::unique_ptr<bg::Pass>(new AlwaysSuccessPass));
  optimizer.AddPass("Bar3", std::unique_ptr<bg::Pass>(new AlwaysSuccessPass));
  auto graph = std::make_shared<ge::ExecuteGraph>("graph");

  bool changed = false;
  ASSERT_EQ(optimizer.Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
TEST_F(BaseOptimizerUT, AddPass_IgnoreNullptr) {
  bg::BaseOptimizer optimizer;
  optimizer.AddPass("Foo", nullptr);
  optimizer.AddPass("Bar", std::unique_ptr<bg::Pass>(new AlwaysSuccessPass));
  auto graph = std::make_shared<ge::ExecuteGraph>("graph");

  bool changed = false;
  ASSERT_EQ(optimizer.Run(graph.get(), changed), ge::GRAPH_SUCCESS);
}
TEST_F(BaseOptimizerUT, Run_Exit_ExceedsMaxTurnCount) {
  bg::BaseOptimizer optimizer;
  optimizer.AddPass("Test", std::unique_ptr<bg::Pass>(new AlwaysChangedPass));
  optimizer.AddPass("Test", std::unique_ptr<bg::Pass>(new AlwaysChangedPass));
  auto graph = std::make_shared<ge::ExecuteGraph>("graph");
  // set info log level to test PrintChangedPasses
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  bool changed = false;
  ASSERT_EQ(optimizer.Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}
}  // namespace gert
