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
#include <iostream>
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "graph/utils/connection_matrix.h"
#include "graph/utils/connection_matrix_impl.h"

using namespace ge;
class UtestCycleDetection : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class FusionTestPass : public fe::PatternFusionBasePass {
 public:
  FusionTestPass() {};
  ~FusionTestPass() override {};
  vector<fe::FusionPattern *> DefinePatterns() override {return {nullptr};};
  fe::Status Fusion(ComputeGraph &graph, Mapping &mapping,
                    vector<NodePtr> &new_nodes) override {

    vector<vector<NodePtr>> fusion_nodes;
    bool ret = CycleDetection(graph, fusion_nodes);
    EXPECT_EQ(ret, false);

    vector<NodePtr> scope_nodes;
    for (auto &node : graph.GetDirectNode()) {
      if (std::find(new_nodes.begin(), new_nodes.end(), node) != new_nodes.end()) {
        scope_nodes.emplace_back(node);
      }
    }
    fusion_nodes.emplace_back(scope_nodes);

    ret = CycleDetection(graph, fusion_nodes);
    if (ret) {
      return fe::NOT_CHANGED;
    } else {
      return fe::SUCCESS;
    }
  }
};


/*               A
 *             /  \
 *            B    \
 *           /      \
 *          D------->C
 *          |        |
 * After fusion A/B/C, the graph looks like:
 *              <---
 *             /    \
 *           ABC--->D */
static ComputeGraphPtr BuildFusionGraph01(std::vector<ge::NodePtr> &fusion_nodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("fusion_graph");
  auto a = builder.AddNode("A", "A", 0, 1);
  auto b = builder.AddNode("B", "B", 1, 1);
  auto c = builder.AddNode("C", "C", 2, 1);
  auto d = builder.AddNode("D", "D", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 2, 0);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, d, 0);
  builder.AddDataEdge(d, 0, c, 1);

  builder.AddDataEdge(a, 0, c, 0);
  builder.AddDataEdge(c, 0, netoutput, 0);
  builder.AddDataEdge(d, 0, netoutput, 1);
  auto graph = builder.GetGraph();
  fusion_nodes = {a, b, c};
  return graph;
}
using Mapping = std::map<const std::shared_ptr<fe::FusionPattern::OpDesc>, std::vector<ge::NodePtr>, fe::CmpKey>;
TEST_F(UtestCycleDetection, cycle_detection_01) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph01(fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}


/*               A
 *             /  \
 *            B    \
 *           /      \
 *          D        C
 *          \       /
 *          Netoutput
 * After fusion A/B/C, the graph looks like:
 *
 *           ABC--->D
 *            \     /
 *           Netoutput
 *    No cycle will be generated if fusing. */
static ComputeGraphPtr BuildFusionGraph02(std::vector<ge::NodePtr> &fusion_nodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("fusion_graph");
  auto a = builder.AddNode("A", "A", 0, 1);
  auto b = builder.AddNode("B", "B", 1, 1);
  auto c = builder.AddNode("C", "C", 1, 1);
  auto d = builder.AddNode("D", "D", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 2, 0);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, d, 0);

  builder.AddDataEdge(a, 0, c, 0);
  builder.AddDataEdge(c, 0, netoutput, 0);
  builder.AddDataEdge(d, 0, netoutput, 1);
  auto graph = builder.GetGraph();
  fusion_nodes = {a, b, c};
  return graph;
}

TEST_F(UtestCycleDetection, cycle_detection_02) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph02(fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

/*   A--->B---->C---->D
 *     \-----E-------/
 *
 *   A, B, C, D will be fused.
 *   Cycle will be generated if fusing.
 */
static ComputeGraphPtr BuildFusionGraph03(std::vector<ge::NodePtr> &fusion_nodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("fusion_graph");
  auto a = builder.AddNode("A", "A", 0, 1);
  auto b = builder.AddNode("B", "B", 1, 1);
  auto c = builder.AddNode("C", "C", 1, 1);
  auto d = builder.AddNode("D", "D", 2, 1);
  auto e = builder.AddNode("E", "E", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 1, 0);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, c, 0);

  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(a, 0, e, 0);
  builder.AddDataEdge(e, 0, d, 1);
  builder.AddDataEdge(d, 0, netoutput, 0);

  auto graph = builder.GetGraph();
  fusion_nodes = {a, b, c, d};
  return graph;
}

TEST_F(UtestCycleDetection, cycle_detection_03) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph03(fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

/*   A--->B---->C------->D
 *     \-----E---F------/
 *
 *   A, B, C, D will be fused.
 *   Cycle will be generated if fusing.
 */
static ComputeGraphPtr BuildFusionGraph04(std::vector<ge::NodePtr> &fusion_nodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("fusion_graph");
  auto a = builder.AddNode("A", "A", 0, 1);
  auto b = builder.AddNode("B", "B", 1, 1);
  auto c = builder.AddNode("C", "C", 1, 1);
  auto d = builder.AddNode("D", "D", 2, 1);
  auto e = builder.AddNode("E", "E", 1, 1);
  auto f = builder.AddNode("F", "F", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 1, 0);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, c, 0);
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(a, 0, e, 0);
  builder.AddDataEdge(e, 0, f, 0);
  builder.AddDataEdge(f, 0, d, 1);

  builder.AddDataEdge(d, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  fusion_nodes = {a, b, c, d};
  return graph;
}

TEST_F(UtestCycleDetection, cycle_detection_04) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph04(fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

/*   A--->B---->C------->D
 *     \-----E---F------/
 *
 *   B/C will be fused.
 *   No Cycle will be generated if fusing.
 */
static ComputeGraphPtr BuildFusionGraph05(std::vector<ge::NodePtr> &fusion_nodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("fusion_graph");
  auto a = builder.AddNode("A", "A", 0, 1);
  auto b = builder.AddNode("B", "B", 1, 1);
  auto c = builder.AddNode("C", "C", 1, 1);
  auto d = builder.AddNode("D", "D", 2, 1);
  auto e = builder.AddNode("E", "E", 1, 1);
  auto f = builder.AddNode("F", "F", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 1, 0);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, c, 0);
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(a, 0, e, 0);
  builder.AddDataEdge(e, 0, f, 0);
  builder.AddDataEdge(f, 0, d, 0);

  builder.AddDataEdge(d, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  fusion_nodes = {b, c};
  return graph;
}

TEST_F(UtestCycleDetection, cycle_detection_05) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph05(fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

const int kContainCycle = 0;
const int kNoCycleCase1 = 1;
const int kNoCycleCase2 = 2;
const int kNoCycleCase3 = 3;
/*
 *        /-----H----------------\
 *       /------G---------\       \
 *      /    /------I------\       \
 *     A--->B---->C------->D---NetOutput
 *      \------E---F------------/
 *
 *   B/C will be fused.
 *   No Cycle will be generated if fusing.
 */
ComputeGraphPtr CreateGraph06(int case_num, std::vector<ge::NodePtr> &fusion_nodes) {
  ut::GraphBuilder builder = ut::GraphBuilder("fusion_graph");
  auto a = builder.AddNode("A", "A", 0, 4);
  auto b = builder.AddNode("B", "B", 1, 1);
  auto c = builder.AddNode("C", "C", 1, 1);
  auto d = builder.AddNode("D", "D", 3, 1);
  auto e = builder.AddNode("E", "E", 1, 1);
  auto f = builder.AddNode("F", "F", 1, 1);
  auto g = builder.AddNode("G", "G", 1, 1);
  auto h = builder.AddNode("H", "H", 1, 1);
  auto i = builder.AddNode("I", "I", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 3, 0);

  builder.AddControlEdge(a, b);
  builder.AddDataEdge(a, 0, e, 0);
  builder.AddDataEdge(a, 1, g, 0);
  builder.AddDataEdge(a, 2, h, 0);
  builder.AddDataEdge(h, 0, netoutput, 0);

  builder.AddDataEdge(b, 0, c, 0);
  builder.AddDataEdge(b, 0, i, 0);
  builder.AddDataEdge(i, 0, d, 0);
  builder.AddDataEdge(c, 0, d, 1);
  builder.AddDataEdge(d, 0, netoutput, 1);

  builder.AddDataEdge(g, 0, d, 2);

  builder.AddDataEdge(e, 0, f, 0);
  builder.AddDataEdge(f, 0, netoutput, 3);

  auto graph = builder.GetGraph();
  if (case_num == kNoCycleCase1) {
    fusion_nodes = {a, b, e, g, h};
  } else if (case_num == kContainCycle) {
    fusion_nodes = {b, c, d};
  } else if (case_num == kNoCycleCase2) {
    fusion_nodes = {b, c, i};
  } else if (case_num == kNoCycleCase3) {
    fusion_nodes = {b, c, d, i};
  }
  return graph;
}


static ComputeGraphPtr BuildFusionGraph06(int case_num,
                                          std::vector<ge::NodePtr> &fusion_nodes) {
  auto graph = CreateGraph06(case_num, fusion_nodes);
  return graph;
}

TEST_F(UtestCycleDetection, cycle_detection_06) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase1, fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UtestCycleDetection, cycle_detection_07) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kContainCycle, fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(UtestCycleDetection, cycle_detection_08) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UtestCycleDetection, cycle_detection_09) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
  FusionTestPass pass;
  Mapping mapping;

  fe::Status ret = pass.Fusion(*graph, mapping, fusion_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UtestCycleDetection, Coverage_01) {
  ge::LargeBitmap a(5);
  a.SetValues(1);

  ge::LargeBitmap b(5);
  b.SetValues(2);

  a.Or(b);

  ge::LargeBitmap c(5);
  c.SetValues(3);
  EXPECT_EQ(a == c, true);
}

TEST_F(UtestCycleDetection, Coverage_02) {
  ge::LargeBitmap a(5);
  a.SetValues(1);

  ge::LargeBitmap b(5);
  b.SetValues(2);

  a.And(b);

  ge::LargeBitmap c(5);
  c.SetValues(0);
  EXPECT_EQ(a == c, true);

  EXPECT_EQ(a != c, false);
}

TEST_F(UtestCycleDetection, Coverage_03) {
  ge::LargeBitmap a(6);
  a.SetValues(1);
  a.SetBit(10000);
  a.GetBit(10000);

  ge::LargeBitmap b(5);
  b.SetValues(2);

  a.And(b);
  a.Or(b);

  ge::LargeBitmap c(5);
  c.SetValues(0);
  EXPECT_EQ(a == c, false);

  EXPECT_EQ(a != c, true);
}

TEST_F(UtestCycleDetection, Coverage_04) {
  EXPECT_NO_THROW(
    std::vector<ge::NodePtr> fusion_nodes;
    auto graph = BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
    auto connectivity = std::shared_ptr<fe::ConnectionMatrix>(new(std::nothrow) fe::ConnectionMatrix(*graph));
    connectivity->Generate(*graph);
    connectivity->Update(*graph, fusion_nodes);
  );
}

TEST_F(UtestCycleDetection, Coverage_05) {
  EXPECT_NO_THROW(
    FusionTestPass pass;
    std::unique_ptr<fe::ConnectionMatrix> connection_matrix;
    pass.GetConnectionMatrix(connection_matrix);
    pass.SetConnectionMatrix(connection_matrix);
  );
}

TEST_F(UtestCycleDetection, Coverage_06) {
  EXPECT_NO_THROW(
    auto graph = std::make_shared<ge::ComputeGraph>("test");
    auto connection_matrix = std::shared_ptr<fe::ConnectionMatrix>(new(std::nothrow) fe::ConnectionMatrix(*graph));

    std::vector<ge::NodePtr> fusion_nodes;
    BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
    connection_matrix->GetIndex(fusion_nodes[0]);
  );
}

TEST_F(UtestCycleDetection, Coverage_07) {
  EXPECT_NO_THROW(
    auto graph = std::make_shared<ge::ComputeGraph>("test");
    auto connection_matrix = std::shared_ptr<ge::ConnectionMatrixImpl>(new(std::nothrow)
        ge::ConnectionMatrixImpl(graph));

    std::vector<ge::NodePtr> fusion_nodes;
    BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
    connection_matrix->GetIndex(fusion_nodes[0]);
  );
}

