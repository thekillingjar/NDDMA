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
#include "common/context/properties_manager.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge
{
class UtestPropertiesManager : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

static ComputeGraphPtr BuildSubComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  return graph;
}

static ComputeGraphPtr BuildComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  transdata->GetOpDesc()->AddSubgraphName("subgraph");
  transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  // add subgraph
  transdata->SetOwnerComputeGraph(graph);
  ComputeGraphPtr subgraph = BuildSubComputeGraph();
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(transdata);
  graph->AddSubgraph("subgraph", subgraph);
  return graph;
}

TEST_F(UtestPropertiesManager, Init) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  PropertiesManager properties_manager;
  //EXPECT_EQ(properties_manager.Init("./ut_graph1.txt"), true);
  EXPECT_EQ(properties_manager.Init("./ut_graph1.txt"), false);
  system("rm -rf ./ut_graph1.txt");
}

TEST_F(UtestPropertiesManager, ParseLine) {
  PropertiesManager properties_manager;
  std::string str_line("#123");
  EXPECT_EQ(properties_manager.ParseLine(str_line), true);
  str_line = "=test_value";
  EXPECT_EQ(properties_manager.ParseLine(str_line), false);
  PropertiesManager::Instance().SetPropertyValue("test_key", "test_value");
  str_line = "test_key=test_value";
  EXPECT_EQ(properties_manager.ParseLine(str_line), true);
}

TEST_F(UtestPropertiesManager, LoadFileContent) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  EXPECT_EQ(PropertiesManager::Instance().LoadFileContent("./ut_graph1.txt"), false);
  graph.SaveToFile("./ut_graph1.txt");
  EXPECT_EQ(PropertiesManager::Instance().LoadFileContent("./ut_graph1.txt"), false);
  system("rm -rf ./ut_graph1.txt");
}

TEST_F(UtestPropertiesManager, GetPropertyMap_success) {
  PropertiesManager::Instance().SetPropertyValue("test_key", "test_value");
  EXPECT_EQ(PropertiesManager::Instance().GetPropertyMap().size(), 1);
}

TEST_F(UtestPropertiesManager, TrimStr) {
  std::string str("");
  std::string strRet = PropertiesManager::Instance().TrimStr(str);
  EXPECT_EQ(strRet, "");

  str = "123\t\r\n";
  strRet = PropertiesManager::Instance().TrimStr(str);
  EXPECT_EQ(strRet, "123");
}

TEST_F(UtestPropertiesManager, GetPropertyValue) {
  PropertiesManager::Instance().SetPropertyValue("test_key", "test_value");
  EXPECT_EQ(PropertiesManager::Instance().GetPropertyValue("test_key"), "test_value");
  EXPECT_EQ(PropertiesManager::Instance().GetPropertyValue("test_key_t"), "");
}
} // namespace ge
