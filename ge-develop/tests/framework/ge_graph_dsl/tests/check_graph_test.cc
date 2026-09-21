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
#include "easy_graph/layout/graph_layout.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "framework/common/types.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/model.h"
#include "graph/buffer.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"

USING_GE_NS

class CheckGraphTest : public testing::Test {
 private:
  EG_NS::GraphEasyExecutor executor;

 protected:
  void SetUp() { EG_NS::GraphLayout::GetInstance().Config(executor, nullptr); }
  void TearDown() {}
};

TEST_F(CheckGraphTest, test_ge_graph_dump_is_work) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  DUMP_GRAPH_WHEN("after_build");
  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g1), "after_build");

  CHECK_GRAPH(after_build) {
    ASSERT_EQ(graph->GetName(), "g1");
    ASSERT_EQ(graph->GetAllNodesSize(), 2);
  };
}

TEST_F(CheckGraphTest, test_ge_graph_dump_two_phase) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };
  DEF_GRAPH(g2) {
    CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CTRL_CHAIN(NODE("data2", DATA)->NODE("add", ADD));
  };

  DUMP_GRAPH_WHEN("before_build", "after_build");

  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g1), "before_build");
  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g2), "after_build");

  CHECK_GRAPH(before_build) {
    ASSERT_EQ(graph->GetName(), "g1");
    ASSERT_EQ(graph->GetAllNodesSize(), 2);
  };

  CHECK_GRAPH(after_build) {
    ASSERT_EQ(graph->GetName(), "g2");
    ASSERT_EQ(graph->GetAllNodesSize(), 3);
  };
}

TEST_F(CheckGraphTest, test_ge_graph_dump_one_phase_two_times) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };
  DEF_GRAPH(g2) {
    CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CTRL_CHAIN(NODE("data2", DATA)->NODE("add", ADD));
  };

  DUMP_GRAPH_WHEN("before_build")

  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g1), "before_build");
  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g2), "before_build");

  CHECK_GRAPH(before_build) {
    ASSERT_EQ(graph->GetName(), "g2");
    ASSERT_EQ(graph->GetAllNodesSize(), 3);
  };
}

TEST_F(CheckGraphTest, test_check_phases_is_work) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  DUMP_GRAPH_WHEN("before_build");
  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g1), "after_build");
  auto ret = ::GE_NS::CheckUtils::CheckGraph("after_build", [&](const ::GE_NS::ComputeGraphPtr &graph) {});
  ASSERT_FALSE(ret);
}

TEST_F(CheckGraphTest, test_check_one_phase_dump_another_not_dump) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };

  DUMP_GRAPH_WHEN("before_build");
  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g1), "before_build");
  GraphDumperRegistry::GetDumper().Dump(ToComputeGraph(g1), "after_build");

  CHECK_GRAPH(before_build) {
    ASSERT_EQ(graph->GetName(), "g1");
    ASSERT_EQ(graph->GetAllNodesSize(), 2);
  };
}

TEST_F(CheckGraphTest, test_model_serialize_and_unserialize_success) {
  DEF_GRAPH(g1) { CTRL_CHAIN(NODE("data1", DATA)->NODE("add", ADD)); };
  auto ge_graph = ToGeGraph(g1);

  ge::Model model("", "");
  model.SetGraph(ge::GraphUtilsEx::GetComputeGraph(ge_graph));
  Buffer buffer;
  model.Save(buffer, true);

  ge::Model loadModel("", "");
  Model::Load(buffer.GetData(), buffer.GetSize(), loadModel);
  auto load_graph = loadModel.GetGraph();

  ASSERT_EQ(load_graph->GetName(), "g1");
  ASSERT_EQ(load_graph->GetAllNodes().size(), 2);
}
