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
#include "common/mem_conflict_share_graph.h"
#include "graph/compute_graph.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"

namespace ge {
class UtestCheckerLogUtil : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCheckerLogUtil, CheckNodeIndexIoStr) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  NodeIndexIO node_index_io{a, 0U, kOut};
  auto str = CheckerLog::ToStr(node_index_io);
  EXPECT_STRCASEEQ(str.c_str(), "a_out_0");
}

TEST_F(UtestCheckerLogUtil, CheckNodeIndexIoStr_Len200) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  a->GetOpDesc()->SetName("200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  NodeIndexIO node_index_io{a, 0U, kOut};
  auto str = CheckerLog::ToStr(node_index_io);
  EXPECT_STRCASEEQ(str.c_str(), "200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...TopoId_0_out_0");
}

TEST_F(UtestCheckerLogUtil, CheckNodeIndexIoStr_Len205) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  a->GetOpDesc()->SetName("200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  NodeIndexIO node_index_io{a, 0U, kOut};
  auto str = CheckerLog::ToStr(node_index_io);
  EXPECT_STRCASEEQ(str.c_str(), "200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...TopoId_0_out_0");
}

TEST_F(UtestCheckerLogUtil, CheckInDataAnchorToStr_Len205) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("b", RELU)->NODE("a", RELU)->NODE("c", RELU));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  a->GetOpDesc()->SetName("200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  auto in_anchor = a->GetInDataAnchor(0);
  ASSERT_NE(in_anchor, nullptr);
  auto str = CheckerLog::ToStr(in_anchor);
  EXPECT_STRCASEEQ(str.c_str(), "200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...TopoId_1_in_0");
}

TEST_F(UtestCheckerLogUtil, CheckOutDataAnchorToStr_Len205) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("b", RELU)->NODE("a", RELU)->NODE("c", RELU));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  a->GetOpDesc()->SetName("200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  auto anchor = a->GetOutDataAnchor(0);
  ASSERT_NE(anchor, nullptr);
  auto str = CheckerLog::ToStr(anchor);
  EXPECT_STRCASEEQ(str.c_str(), "200aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...TopoId_1_out_0");
}
} // namespace ge
