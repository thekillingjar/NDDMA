/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/pass_manager.h"
#include "ge/ge_api_types.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/ge_local_context.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class SuccessGraphPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) {
    ++run_count_;
    return SUCCESS; 
  }
  int32_t GetRunCount() const {
    return run_count_;
  }

 private:
  int32_t run_count_ = 0;
};
REG_PASS_OPTION("SuccessGraphPass").LEVELS(OoLevel::kO3);

class NotChangedGraphPass : public GraphPass {
  Status Run(ComputeGraphPtr graph) { return NOT_CHANGED; }
};
REG_PASS_OPTION("NotChangedGraphPass").LEVELS(OoLevel::kO3);

class ErrorGraphPass : public GraphPass {
  Status Run(ComputeGraphPtr graph) { return FAILED; }
};

class UtestGraphPassesPassManagerPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

NodePtr AddNode(ComputeGraphPtr graph) {
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  OpDescPtr opdesc = std::make_shared<OpDesc>("test", "Add");
  opdesc->AddInputDesc(tensor_desc);
  opdesc->AddOutputDesc(tensor_desc);
  NodePtr node = graph->AddNode(opdesc);
  return node;
}

ComputeGraphPtr CreatePadGraph() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  return graph;
}

TEST_F(UtestGraphPassesPassManagerPass, all_pass_success) {
  PassManager manager;
  manager.AddPass("SuccessGraphPass", new SuccessGraphPass);
  EXPECT_EQ(manager.GraphPasses().size(), 1);

  ComputeGraphPtr graph = CreatePadGraph();
  Status status = manager.Run(graph);
  EXPECT_EQ(SUCCESS, status);
}

REG_PASS_OPTION("DisableButNotInWhiteList").LEVELS(OoLevel::kO3);
TEST_F(UtestGraphPassesPassManagerPass, test_disable_pass) {
  PassManager manager;
  auto enabled_graph_pass_no_name = new SuccessGraphPass;
  auto enabled_graph_pass_with_stage = new SuccessGraphPass;
  auto enabled_graph_pass_without_stage = new SuccessGraphPass;
  auto disabled_graph_pass_with_stage = new SuccessGraphPass;
  auto disabled_graph_pass_without_stage = new SuccessGraphPass;
  manager.AddPass("SuccessGraphPass", enabled_graph_pass_no_name);
  manager.AddPass("OptimizeStage::DisableButNotInWhiteList", enabled_graph_pass_with_stage);
  manager.AddPass("DisableButNotInWhiteList", enabled_graph_pass_without_stage);
  manager.AddPass("OptimizeStage::RemoveSameConstPass", disabled_graph_pass_with_stage);
  manager.AddPass("RemoveSameConstPass", disabled_graph_pass_without_stage);
  EXPECT_EQ(manager.GraphPasses().size(), 5);

  auto origin_graph_options = GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> disable_optimization_option;
  disable_optimization_option.emplace(OPTION_DISABLE_OPTIMIZATIONS, "RemoveSameConstPass, DisableButNotInWhiteList");
  GetThreadLocalContext().SetGraphOption(disable_optimization_option);

  ComputeGraphPtr graph = CreatePadGraph();
  Status status = manager.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(enabled_graph_pass_no_name->GetRunCount(), 1);
  EXPECT_EQ(enabled_graph_pass_with_stage->GetRunCount(), 1);
  EXPECT_EQ(enabled_graph_pass_without_stage->GetRunCount(), 1);
  EXPECT_EQ(disabled_graph_pass_with_stage->GetRunCount(), 0);
  EXPECT_EQ(disabled_graph_pass_without_stage->GetRunCount(), 0);
  GetThreadLocalContext().SetGraphOption(origin_graph_options);
}

TEST_F(UtestGraphPassesPassManagerPass, graph_pass_not_changed) {
  ComputeGraphPtr graph = CreatePadGraph();
  NotChangedGraphPass pass;
  std::vector<std::pair<string, GraphPass*>> passes = {{"NotChangedGraphPass", &pass}};
  Status status = PassManager::Run(graph, passes);
  //EXPECT_EQ(NOT_CHANGED, status);
  EXPECT_EQ(status, SUCCESS);
}
