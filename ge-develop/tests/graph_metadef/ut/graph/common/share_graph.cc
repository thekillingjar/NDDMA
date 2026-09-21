/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "share_graph.h"

#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
template<typename GraphT, typename BuilderT>
GraphT SharedGraph<GraphT, BuilderT>::BuildGraphWithSubGraph() {
  auto root_builder = CreateBuilder("root");
  const auto &data0 = root_builder.AddNode("data0", "Data", 1, 1);
  const auto &case0 = root_builder.AddNode("case0", "Case", 1, 1);
  const auto &relu0 = root_builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = root_builder.AddNode("relu1", "Relu", 1, 1);
  const auto &netoutput = root_builder.AddNode("netoutput", "NetOutput", 1, 1);
  auto root_graph = root_builder.GetGraph();
  root_builder.AddDataEdge(data0, 0, case0, 0);
  root_builder.AddDataEdge(case0, 0, relu0, 0);
  root_builder.AddDataEdge(relu0, 0, relu1, 0);
  root_builder.AddDataEdge(relu1, 0, netoutput, 0);

  auto sub_builder1 = CreateBuilder("sub1");
  auto sub1_data1 = sub_builder1.AddNode("data1", "Data", 0, 1);
  const auto &sub1_netoutput = sub_builder1.AddNode("sub1_netoutput", "NetOutput", 1, 1);
  sub_builder1.AddDataEdge(sub1_data1, 0, sub1_netoutput, 0);
  auto sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  BuilderUtils<GraphT>::SetParentGraph(sub_graph1, root_graph);
  case0->GetOpDescBarePtr()->AddSubgraphName("branch1");
  case0->GetOpDescBarePtr()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = CreateBuilder("sub2");
  auto sub2_data2 = sub_builder2.AddNode("data2", "Data", 0, 1);
  const auto &sub2_netoutput = sub_builder2.AddNode("sub2_netoutput", "NetOutput", 1, 1);
  sub_builder2.AddDataEdge(sub2_data2, 0, sub2_netoutput, 0);
  auto sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case0);
  BuilderUtils<GraphT>::SetParentGraph(sub_graph2, root_graph);
  case0->GetOpDescBarePtr()->AddSubgraphName("branch2");
  case0->GetOpDescBarePtr()->SetSubgraphInstanceName(1, "sub2");
  root_graph->TopologicalSorting();
  return root_graph;
}

template<typename GraphT, typename BuilderT>
GraphT SharedGraph<GraphT, BuilderT>::BuildGraphWithConst() {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  auto builder = CreateBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDescBarePtr(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  AttrUtils::SetStr(const_node->GetOpDescBarePtr(), "fake_attr_name", "fake_attr_value");
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  AttrUtils::SetStr(add_node->GetOpDescBarePtr(), "fake_attr_name", "fake_attr_value");
  AttrUtils::SetStr(add_node->GetOpDescBarePtr(), ge::ATTR_NAME_WEIGHTS, "fake_attr_value");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  return builder.GetGraph();
}

/**            n5
 *        /    |   \\c
 *      n2    n3    n4
 *        \   |    //c
 *           n1
 */
template<typename GraphT, typename BuilderT>
GraphT SharedGraph<GraphT, BuilderT>::BuildGraphWithControlEdge() {
  auto builder = CreateBuilder("graph_with_ctrl_edge");
  auto n1 = builder.AddNode("n1", "Data", 1, 1);
  auto n2 = builder.AddNode("n2", "Op", 1, 1);
  auto n3 = builder.AddNode("n3", "Op", 1, 1);
  auto n4 = builder.AddNode("n4", "Op", 1, 1);
  auto n5 = builder.AddNode("n5", "NetOutput", 3, 1);
  builder.AddDataEdge(n1, 0, n2, 0);
  builder.AddDataEdge(n1, 0, n3, 0);
  builder.AddControlEdge(n1, n4);
  builder.AddDataEdge(n2, 0, n5, 0);
  builder.AddDataEdge(n3, 0, n5, 1);
  builder.AddControlEdge(n4, n5);
  builder.AddDataEdge(n1, 0, n4, 0);
  builder.AddDataEdge(n4, 0, n5, 2);
  return builder.GetGraph();
}

template class SharedGraph<ComputeGraphPtr, ut::GraphBuilder>;
template class SharedGraph<ExecuteGraphPtr, ut::ExecuteGraphBuilder>;
}  // namespace ge
