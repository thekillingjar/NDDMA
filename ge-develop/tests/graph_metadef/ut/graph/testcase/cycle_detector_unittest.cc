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
#include <vector>
#include <sstream>
#include "graph/utils/cycle_detector.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/connection_matrix.h"
#include "graph/utils/connection_matrix_impl.h"
#include "graph_builder_utils.h"
using namespace std;
using namespace ge;
namespace {
const int kContainCycle = 0;
const int kNoCycleCase1 = 1;
const int kNoCycleCase2 = 2;
const int kNoCycleCase3 = 3;

/*
*  if we want to fusion cast1 and cast2
*  it will cause a cycle between fusion_cast and transdata
*       data1
*       /    \
*      /      \
*    cast1     \
*      |        \
*   trandata---> cast2
*/
void BuildGraphMayCauseCycleWhenFusion(ComputeGraphPtr &graph) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data1 = root_builder.AddNode("data1", "Data", 1, 1);
  const auto &cast1 = root_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = root_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &transdata = root_builder.AddNode("transdata", "TransData", 1, 1);

  root_builder.AddDataEdge(data1, 0, cast1, 0);
  root_builder.AddDataEdge(data1, 0, cast2, 0);
  root_builder.AddDataEdge(cast1, 0, transdata, 0);
  root_builder.AddControlEdge(transdata, cast2);
  graph = root_builder.GetGraph();
}

}  // namespace
class UtestCycleDetector : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
/*
*  if we want to fusion cast1 and cast2
*  it will cause a cycle between fusion_cast and transdata
*       data1                           data1
*       /    \                           |
*      /      \             ===>      fusion_cast  <------
*    cast1     \                         |               |
*      |        \                     transdata-----------
*   trandata---> cast2          wrong graph, which has cycle between
*                               transdata and fusion_cast.
*/
TEST_F(UtestCycleDetector, TestCycleDetection_00) {
  ComputeGraphPtr graph;
  BuildGraphMayCauseCycleWhenFusion(graph);

  auto cast1 = graph->FindNode("cast1");
  auto cast2 = graph->FindNode("cast2");
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  
  bool has_cycle = detector->HasDetectedCycle({{cast1, cast2}});
  EXPECT_TRUE(has_cycle);
}

/*               A
 *             /  \
 *            B    \
 *           /      \
 *          D------->C
 *          |        |
 * After fusion A/B/C, the graph looks like:
 *              <---
 *             /    \
 *           ABC--->D 
 */
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

TEST_F(UtestCycleDetector, TestCycleDetection_01) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph01(fusion_nodes);
  
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_TRUE(has_cycle);
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

/*
ori connection_matrix(5x5):
1 0 0 0 0
1 1 0 0 0
1 0 1 0 0
1 0 1 1 0
1 1 1 1 1
After update(6x6):
1 1 1 0 0 1
1 1 1 0 0 1
1 1 1 0 0 1
1 1 1 1 0 1
1 1 1 1 1 1
1 1 1 0 0 1
*/

TEST_F(UtestCycleDetector, cycle_detection_02) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph02(fusion_nodes);

  CycleDetectorSharedPtr detector = GraphUtils::CreateSharedCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_FALSE(has_cycle);
  std::string res_ori = "1000011000101001011011111";
  std::string res_update = "111001111001111001111101111111111001";
  std::stringstream val_ori;
  for (size_t i = 0; i < 5; ++i) {
    auto bit_map = detector->connectivity_->impl_->bit_maps_[i];
    for (size_t j = 0; j < 5; ++j) {
      val_ori << bit_map.GetBit(j);
    }
  }
  EXPECT_EQ(val_ori.str(), res_ori);
  detector->ExpandAndUpdate(fusion_nodes, "ABC");
  std::stringstream val_update;
  for (size_t i = 0; i < 6; ++i) {
    auto bit_map = detector->connectivity_->impl_->bit_maps_[i];
    for (size_t j = 0; j < 6; ++j) {
      val_update << bit_map.GetBit(j);
    }
  }
  EXPECT_EQ(val_update.str(), res_update);
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

TEST_F(UtestCycleDetector, cycle_detection_03) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph03(fusion_nodes);

  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_TRUE(has_cycle);
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

TEST_F(UtestCycleDetector, cycle_detection_04) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph04(fusion_nodes);

  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_TRUE(has_cycle);
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

TEST_F(UtestCycleDetector, cycle_detection_05) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph05(fusion_nodes);

  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_FALSE(has_cycle);
}

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

TEST_F(UtestCycleDetector, cycle_detection_06) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase1, fusion_nodes);
  
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_FALSE(has_cycle);
}

TEST_F(UtestCycleDetector, cycle_detection_07) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kContainCycle, fusion_nodes);
  
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_TRUE(has_cycle);
}

TEST_F(UtestCycleDetector, cycle_detection_08) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
  
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_FALSE(has_cycle);
}

TEST_F(UtestCycleDetector, cycle_detection_09) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase3, fusion_nodes);
  
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  bool has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_FALSE(has_cycle);
}

TEST_F(UtestCycleDetector, ConnectionMatrixCoverage_00) {
  std::vector<ge::NodePtr> fusion_nodes;
  auto graph = BuildFusionGraph06(kNoCycleCase2, fusion_nodes);
  CycleDetectorPtr detector = GraphUtils::CreateCycleDetector(graph);
  EXPECT_NE(detector, nullptr);
  detector->Update(graph, fusion_nodes);
  auto has_cycle = detector->HasDetectedCycle({fusion_nodes});
  EXPECT_FALSE(has_cycle);
  detector->Update(graph, fusion_nodes);
}
