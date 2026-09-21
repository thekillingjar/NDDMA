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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "engines/local_engine/engine/ge_local_engine.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace ge_local;

class UtestGeLocalEngine : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestGeLocalEngine, Initialize) {
    auto &instance = GeLocalEngine::Instance();
    std::map<std::string, std::string> options;
    EXPECT_EQ(instance.Initialize(options), SUCCESS);
    EXPECT_EQ(Initialize(options), SUCCESS);
}

TEST_F(UtestGeLocalEngine, GetOpsKernelInfoStores) {
    auto &instance = GeLocalEngine::Instance();
    std::map<std::string, OpsKernelInfoStorePtr> ops_kernel_map;
    instance.GetOpsKernelInfoStores(ops_kernel_map);
    EXPECT_NO_THROW(GetOpsKernelInfoStores(ops_kernel_map));
}

TEST_F(UtestGeLocalEngine, GetGraphOptimizerObjs) {
    auto &instance = GeLocalEngine::Instance();
    std::map<std::string, GraphOptimizerPtr> grpgh_opt_map;
    instance.GetGraphOptimizerObjs(grpgh_opt_map);
    EXPECT_NO_THROW(GetGraphOptimizerObjs(grpgh_opt_map));
}

TEST_F(UtestGeLocalEngine, Finalize) {
    auto &instance = GeLocalEngine::Instance();
    EXPECT_EQ(instance.Finalize(), SUCCESS);
    EXPECT_EQ(Finalize(), SUCCESS);
}

TEST_F(UtestGeLocalEngine, GeLocalGraphOptimizer_Normal) {
    GeLocalGraphOptimizer optimizer;
    std::map<std::string, std::string> options;
    EXPECT_EQ(optimizer.Initialize(options, nullptr), SUCCESS);
    
    ComputeGraph *graph = nullptr;
    EXPECT_EQ(optimizer.OptimizeOriginalGraph(*graph), SUCCESS);
    EXPECT_EQ(optimizer.OptimizeFusedGraph(*graph), SUCCESS);
    EXPECT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);

    GraphOptimizerAttribute attrs;
    EXPECT_EQ(optimizer.GetAttributes(attrs), SUCCESS);

    EXPECT_EQ(optimizer.Finalize(), SUCCESS);
}

TEST_F(UtestGeLocalEngine, GeLocalGraphOptimizer_JudegInsert) {
  DEF_GRAPH(test1) {
    CHAIN(NODE("mul1", "Mul")->NODE("phony_concat", "PhonyConcat")->NODE("add", "Add")->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("mul2", "Mul")->NODE("phony_concat", "PhonyConcat"));
  };
  auto graph1 = ToGeGraph(test1);
  auto compute_graph1 = GraphUtilsEx::GetComputeGraph(graph1);

  DEF_GRAPH(test2) {
    CHAIN(NODE("add", "Add")->NODE("phony_split", "PhonySplit")->NODE("mul1", "Mul")->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("phony_split", "PhonySplit")->NODE("mul2", "Mul")->NODE("netoutput", "NetOutput"));
  };
  auto graph2 = ToGeGraph(test2);
  auto compute_graph2 = GraphUtilsEx::GetComputeGraph(graph2);
  bool bool_attr = false;
  int64_t int_attr = 10;

  GeLocalGraphOptimizer optimizer;
  auto node_phony_concat = compute_graph1->FindNode("phony_concat");
  EXPECT_EQ(optimizer.OptimizeOriginalGraphJudgeInsert(*compute_graph1.get()), SUCCESS);
  EXPECT_EQ(ge::AttrUtils::GetBool(node_phony_concat->GetOpDesc(), ATTR_NAME_NOTASK, bool_attr), true);
  EXPECT_EQ(bool_attr, true);
  EXPECT_EQ(ge::AttrUtils::GetBool(node_phony_concat->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, bool_attr), true);
  EXPECT_EQ(bool_attr, true);
  EXPECT_EQ(ge::AttrUtils::GetBool(node_phony_concat->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, bool_attr), true);
  EXPECT_EQ(bool_attr, true);
  EXPECT_EQ(ge::AttrUtils::GetInt(node_phony_concat->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, int_attr), true);
  EXPECT_EQ(int_attr, 0);

  auto node_phony_split = compute_graph2->FindNode("phony_split");
  EXPECT_EQ(optimizer.OptimizeOriginalGraphJudgeInsert(*compute_graph2.get()), SUCCESS);
  EXPECT_EQ(ge::AttrUtils::GetBool(node_phony_split->GetOpDesc(), ATTR_NAME_NOTASK, bool_attr), true);
  EXPECT_EQ(bool_attr, true);
  EXPECT_EQ(ge::AttrUtils::GetBool(node_phony_split->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, bool_attr), true);
  EXPECT_EQ(bool_attr, true);
  EXPECT_EQ(ge::AttrUtils::GetBool(node_phony_split->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, bool_attr), true);
  EXPECT_EQ(bool_attr, true);
  EXPECT_EQ(ge::AttrUtils::GetInt(node_phony_split->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, int_attr), true);
  EXPECT_EQ(int_attr, 0);
}

} // namespace ge
