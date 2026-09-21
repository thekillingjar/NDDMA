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

#include "graph/tuning_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/operator_factory_impl.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/anchor.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestTuningUtils : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  	TuningUtils::netoutput_nodes_.clear();
    TuningUtils::data_node_2_end_node_.clear();
  }
};

/*                                  -------------------------
*                                  |  partitioncall_0_const1* |
*     partitioncall_0--------------|             |           |
*           |                      |          netoutput      |
*           |                      --------------------------
*           |                       ------------------         -------------
*           |                      |        data      |       |    data     |
*           |                      |          |       |       |     |       |
*     partitioncall_1--------------|        case -----|-------|   squeeze*  |
*                                  |          |       |       |     |       |
*                                  |      netoutput   |       |  netoutput  |
*                                   ------------------         -------------
*/
ComputeGraphPtr BuildGraphPartitionCall1() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  // 2.build partitioncall_1 sub graph
  auto p2_sub_builder = ut::GraphBuilder("partitioncall_1_sub");
  const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &partitioncall_1_case = p2_sub_builder.AddNode("partitioncall_1_case", "Case", 1, 1);
  const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", NETOUTPUT, 1, 1);
  p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_case, 0);
  p2_sub_builder.AddDataEdge(partitioncall_1_case, 0, partitioncall_1_netoutput, 0);
  const auto &sub_graph2 = p2_sub_builder.GetGraph();
  sub_graph2->SetParentNode(partitioncall_1);
  sub_graph2->SetParentGraph(root_graph);
  partitioncall_1->GetOpDesc()->AddSubgraphName("f");
  partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");

  // 2.1 build case sub graph
  auto case_sub_builder = ut::GraphBuilder("case_sub");
  const auto &case_data = case_sub_builder.AddNode("case_data", DATA, 0, 1);
  AttrUtils::SetInt(case_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &case_squeeze = case_sub_builder.AddNode("case_squeeze", SQUEEZE, 1, 1);
  const auto &case_netoutput = case_sub_builder.AddNode("case_netoutput", NETOUTPUT, 1, 1);
  case_sub_builder.AddDataEdge(case_data, 0, case_squeeze, 0);
  case_sub_builder.AddDataEdge(case_squeeze, 0, case_netoutput, 0);
  const auto &case_sub_graph = case_sub_builder.GetGraph();
  case_sub_graph->SetParentNode(partitioncall_1_case);
  case_sub_graph->SetParentGraph(sub_graph2);
  partitioncall_1_case->GetOpDesc()->AddSubgraphName("branches");
  partitioncall_1_case->GetOpDesc()->SetSubgraphInstanceName(0, "case_sub");

  root_graph->AddSubgraph(case_sub_graph->GetName(), case_sub_graph);
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);
  return root_graph;
}

TEST_F(UtestTuningUtils, ConvertGraphToFile) {
    std::vector<ComputeGraphPtr> tuning_subgraphs;
    std::vector<ComputeGraphPtr> non_tuning_subgraphs;
    auto builder = ut::GraphBuilder("non_tun");
    const auto data0 = builder.AddNode("data_0", DATA, 0, 1);
    const auto data1 = builder.AddNode("data_1", DATA, 1, 1);
    auto nongraph = builder.GetGraph();
    tuning_subgraphs.push_back(BuildGraphPartitionCall1());
    non_tuning_subgraphs.push_back(nongraph);
    EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs), GRAPH_SUCCESS);
    auto nonnodes = non_tuning_subgraphs.at(0)->GetDirectNode();
    auto nonfirst = nonnodes.at(0);
    nonfirst->impl_->op_ = nullptr;
    EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs), GRAPH_FAILED);
    auto nodes = tuning_subgraphs.at(0)->GetDirectNode();
    auto first = nodes.at(0);
    first->impl_->op_ = nullptr;
    EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs), GRAPH_FAILED);
}

TEST_F(UtestTuningUtils, ConvertGraphToFile_HelpInfo) {
    std::vector<ComputeGraphPtr> tuning_subgraphs;
    std::vector<ComputeGraphPtr> non_tuning_subgraphs;
    auto builder = ut::GraphBuilder("non_tun");
    const auto data0 = builder.AddNode("data_0", DATA, 0, 1);
    const auto data1 = builder.AddNode("data_1", DATA, 1, 1);
    auto nongraph = builder.GetGraph();
    tuning_subgraphs.push_back(BuildGraphPartitionCall1());
    non_tuning_subgraphs.push_back(nongraph);
    EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs, true, "path", "user_path"), GRAPH_SUCCESS);
}

TEST_F(UtestTuningUtils, ConvertGraphToFile_Placehodler) {
    std::vector<ComputeGraphPtr> tuning_subgraphs;
    std::vector<ComputeGraphPtr> non_tuning_subgraphs;
    auto builder = ut::GraphBuilder("non_tun");
    const auto plhd0 = builder.AddNode("placeholder_0", PLACEHOLDER, 1, 1);
    auto nongraph = builder.GetGraph();
    tuning_subgraphs.push_back(BuildGraphPartitionCall1());
    non_tuning_subgraphs.push_back(nongraph);
    EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs), GRAPH_FAILED);
}

TEST_F(UtestTuningUtils, ConvertGraphToFile_End) {
    std::vector<ComputeGraphPtr> tuning_subgraphs;
    std::vector<ComputeGraphPtr> non_tuning_subgraphs;
    auto builder = ut::GraphBuilder("non_tun");
    const auto end0 = builder.AddNode("end_0", END, 1, 1);
    auto nongraph = builder.GetGraph();
    tuning_subgraphs.push_back(BuildGraphPartitionCall1());
    non_tuning_subgraphs.push_back(nongraph);
    EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs), GRAPH_FAILED);
}

TEST_F(UtestTuningUtils, PrintCheckLog) {
    EXPECT_NE(TuningUtils::PrintCheckLog(), "");
    TuningUtils::data_2_end_["data"] = "end";
    EXPECT_NE(TuningUtils::PrintCheckLog(), "");
    auto builder = ut::GraphBuilder("graph");
    const auto data0 = builder.AddNode("data_0", DATA, 0, 1);
    const auto data1 = builder.AddNode("data_1", DATA, 1, 1);
    TuningUtils::netoutput_nodes_.push_back(data0);
    TuningUtils::netoutput_nodes_.push_back(data1);
    EXPECT_NE(TuningUtils::PrintCheckLog(), "");
}

TEST_F(UtestTuningUtils, GetNodeNameByAnchor) {
    EXPECT_EQ(TuningUtils::GetNodeNameByAnchor(nullptr), "Null");
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node = builder.AddNode("Data", "Data", 1, 1);
    InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(node, 111);
    EXPECT_EQ(TuningUtils::GetNodeNameByAnchor(in_anch.get()), "Data");
}

TEST_F(UtestTuningUtils, CreateDataNode) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node = builder.AddNode("Data", "Data", 1, 1);
    NodePtr data_node;
    EXPECT_EQ(TuningUtils::CreateDataNode(node, "", data_node), GRAPH_SUCCESS);
    ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
    std::vector<uint8_t> value{1, 2, 3};
    std::vector<int64_t> shape{3};
    tensor->MutableTensorDesc().SetShape(GeShape(shape));
    tensor->SetData(value);
    tensor->MutableTensorDesc().SetDataType(DT_UINT8);
    map<int, ge::GeTensorPtr> weight1;
    weight1[1] = tensor;
    EXPECT_EQ(ge::OpDescUtils::SetWeights(*node, weight1), 0);
    auto node_tensor = OpDescUtils::MutableWeights(node);
    EXPECT_EQ(TuningUtils::CreateDataNode(node, "", data_node), GRAPH_SUCCESS);


    auto sub_builder = ut::GraphBuilder("sub");
    const auto &partitioncall_0_const1 = sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
    const auto &partitioncall_0_netoutput = sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
    AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
    sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
    const auto &sub_graph = sub_builder.GetGraph();
    sub_graph->SetParentNode(node);
    EXPECT_EQ(TuningUtils::CreateDataNode(node, "", data_node), GRAPH_SUCCESS);
}

TEST_F(UtestTuningUtils, CreateDataNode_Weight) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node = builder.AddNode("Data", DATA, 1, 1);
    auto node1 = builder.AddNode("Data1", "Data", 1, 1);
    NodePtr data_node;
    node->GetOpDesc()->SetExtAttr<NodePtr>("parentNode", node1);
    EXPECT_EQ(TuningUtils::CreateDataNode(node, "", data_node), GRAPH_SUCCESS);

    auto pld = builder.AddNode("pld", PLACEHOLDER, 0, 1);
    uint8_t val = 1;
    auto const_tensor = std::make_shared<GeTensor>(GeTensorDesc(), &val, sizeof(val));
    ASSERT_NE(pld->GetOpDesc(), nullptr);
    EXPECT_EQ(ge::AttrUtils::SetTensor(pld->GetOpDesc(), "value", const_tensor), true);
    EXPECT_EQ(ge::AttrUtils::SetStr(pld->GetOpDesc(), "_parentNodeName", "src_const"), true);
    EXPECT_EQ(TuningUtils::CreateDataNode(pld, "", data_node), GRAPH_SUCCESS);
    std::string parent_node_name;
    EXPECT_EQ(ge::AttrUtils::GetStr(data_node->GetOpDesc(), ATTR_NAME_SRC_CONST_NAME, parent_node_name), true);
    EXPECT_EQ(parent_node_name, "src_const");
}

TEST_F(UtestTuningUtils, AddAttrToDataNodeForMergeGraph) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto node1 = builder.AddNode("Data1", "Data", 1, 1);
    EXPECT_EQ(TuningUtils::AddAttrToDataNodeForMergeGraph(node0, node1), FAILED);
    AttrUtils::SetStr(node0->GetOpDesc(), "parentOpType", "Hello world");
    EXPECT_EQ(TuningUtils::AddAttrToDataNodeForMergeGraph(node0, node1), FAILED);
    AttrUtils::SetStr(node0->GetOpDesc(), "_parentNodeName", "Hello world0");
    EXPECT_EQ(TuningUtils::AddAttrToDataNodeForMergeGraph(node0, node1), FAILED);
    AttrUtils::SetInt(node0->GetOpDesc(), "anchorIndex", 1);
    EXPECT_EQ(TuningUtils::AddAttrToDataNodeForMergeGraph(node0, node1), FAILED);
    AttrUtils::SetStr(node0->GetOpDesc(), "_peerNodeName", "Hello world0");
    EXPECT_EQ(TuningUtils::AddAttrToDataNodeForMergeGraph(node0, node1), SUCCESS);
}

TEST_F(UtestTuningUtils, ChangePld2Data) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto node1 = builder.AddNode("Data1", "Data", 1, 1);
    EXPECT_EQ(TuningUtils::ChangePld2Data(node0, node1), FAILED);
    auto node2 = builder.AddNode("placeholder2", PLACEHOLDER, 1, 1);
    auto node3 = builder.AddNode("data3", DATA, 1, 1);
    EXPECT_EQ(TuningUtils::ChangePld2Data(node2, node3), SUCCESS);
    node3->impl_->out_data_anchors_.push_back(nullptr);
    EXPECT_EQ(TuningUtils::ChangePld2Data(node2, node3), FAILED);
}

TEST_F(UtestTuningUtils, HandlePld) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    EXPECT_EQ(TuningUtils::HandlePld(node0, ""), FAILED);
    AttrUtils::SetStr(node0->GetOpDesc(), "parentOpType", "Hello world");
    AttrUtils::SetStr(node0->GetOpDesc(), "_parentNodeName", "Hello world0");
    AttrUtils::SetInt(node0->GetOpDesc(), "anchorIndex", 1);
    AttrUtils::SetStr(node0->GetOpDesc(), "_peerNodeName", "Hello world0");
    EXPECT_EQ(TuningUtils::HandlePld(node0, ""), FAILED);
    auto node2 = builder.AddNode("placeholder2", PLACEHOLDER, 1, 1);
    auto node3 = builder.AddNode("data3", DATA, 1, 1);
    AttrUtils::SetStr(node2->GetOpDesc(), "parentOpType", "Hello world");
    AttrUtils::SetStr(node2->GetOpDesc(), "_parentNodeName", "Hello world0");
    AttrUtils::SetInt(node2->GetOpDesc(), "anchorIndex", 1);
    AttrUtils::SetStr(node2->GetOpDesc(), "_peerNodeName", "Hello world0");
    EXPECT_EQ(TuningUtils::HandlePld(node2, ""), SUCCESS);
}

TEST_F(UtestTuningUtils, CreateNetOutput) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    NodePtr node1;
    auto graph = builder.GetGraph();
    EXPECT_EQ(TuningUtils::CreateNetOutput(node0, node1), FAILED);
    TuningUtils::create_output_[graph] = node0;
    EXPECT_EQ(TuningUtils::CreateNetOutput(node0, node1), SUCCESS);
    TuningUtils::create_output_[graph] = nullptr;
    EXPECT_EQ(TuningUtils::CreateNetOutput(node0, node1), SUCCESS);
}

TEST_F(UtestTuningUtils, AddAttrToNetOutputForMergeGraph) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto node1 = builder.AddNode("Data1", "Data", 1, 1);
    EXPECT_EQ(TuningUtils::AddAttrToNetOutputForMergeGraph(node0, node1, 0), SUCCESS);
}

TEST_F(UtestTuningUtils, LinkEnd2NetOutput) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto node1 = builder.AddNode("Data1", "Data", 1, 1);
    EXPECT_EQ(TuningUtils::LinkEnd2NetOutput(node0, node1), PARAM_INVALID);
    auto node2 = builder.AddNode("Data2", "Data", 0, 1);
    auto node3 = builder.AddNode("Data3", "Data", 1, 1);
    EXPECT_EQ(node2->AddLinkFrom(node3), GRAPH_SUCCESS);
    EXPECT_EQ(TuningUtils::LinkEnd2NetOutput(node2, node3), SUCCESS);
}

TEST_F(UtestTuningUtils, LinkEnd2NetOutput_OutControlAnchor) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node2 = builder.AddNode("Data2", "Data", 1, 1);
    auto node3 = builder.AddNode("Data3", "Data", 1, 1);
    auto node4 = builder.AddNode("Data4", "Data", 1, 1);
    EXPECT_EQ(node2->GetAllInDataAnchors().size(), 1);
    EXPECT_EQ(node2->GetInDataAnchor(0)->GetFirstPeerAnchor(), nullptr);
    EXPECT_EQ(node2->GetInControlAnchor()->LinkFrom(node4->GetOutControlAnchor()), GRAPH_SUCCESS);
    EXPECT_EQ(node2->AddLinkFrom(node3), GRAPH_SUCCESS);
    EXPECT_EQ(TuningUtils::LinkEnd2NetOutput(node2, node3), SUCCESS);
}


TEST_F(UtestTuningUtils, ChangeEnd2NetOutput) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto node1 = builder.AddNode("Data1", "Data", 1, 1);
    EXPECT_EQ(TuningUtils::ChangeEnd2NetOutput(node0, node1), FAILED);
    auto node2 = builder.AddNode("Node2", END, 1, 1);
    auto node3 = builder.AddNode("Node3", NETOUTPUT, 1, 1);
    EXPECT_EQ(TuningUtils::ChangeEnd2NetOutput(node2, node3), FAILED);
    auto node4 = builder.AddNode("Node4", END, 0, 1);
    auto node5 = builder.AddNode("Node5", NETOUTPUT, 1, 1);
    EXPECT_EQ(node4->AddLinkFrom(node5), GRAPH_SUCCESS);
    EXPECT_EQ(TuningUtils::ChangeEnd2NetOutput(node4, node5), SUCCESS);
    auto graph = node4->GetOwnerComputeGraph();
    graph->impl_ = nullptr;
    EXPECT_EQ(TuningUtils::ChangeEnd2NetOutput(node4, node5), FAILED);
}

TEST_F(UtestTuningUtils, HandleEnd) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", DATA, 0, 1);
    auto graph = builder.GetGraph();
    EXPECT_EQ(TuningUtils::HandleEnd(node0), FAILED);
    TuningUtils::create_output_[graph] = node0;
    EXPECT_EQ(TuningUtils::HandleEnd(node0), FAILED);
    auto node4 = builder.AddNode("Node4", END, 0, 1);
    auto node5 = builder.AddNode("Node5", NETOUTPUT, 1, 1);
    EXPECT_EQ(node4->AddLinkFrom(node5), GRAPH_SUCCESS);
    TuningUtils::create_output_[graph] = node5;
    EXPECT_EQ(TuningUtils::HandleEnd(node4), SUCCESS);
}

TEST_F(UtestTuningUtils, ConvertFileToGraph) {
  // build root graph
  auto root_graph_builder = ut::GraphBuilder("root_graph");
  const auto &data_0 = root_graph_builder.AddNode("data_0", DATA, 0, 1);
  AttrUtils::SetInt(data_0->GetOpDesc(), "_parent_node_index", 0);
  const auto &case_0 = root_graph_builder.AddNode("case_0", "Case", 1, 1);
  const auto &netoutput_0 = root_graph_builder.AddNode("netoutput_0", NETOUTPUT, 1, 1);
  root_graph_builder.AddDataEdge(data_0, 0, case_0, 0);
  root_graph_builder.AddDataEdge(case_0, 0, netoutput_0, 0);
  case_0->GetOpDesc()->AddSubgraphName("branches");
  case_0->GetOpDesc()->SetSubgraphInstanceName(0, "case_sub");
  const auto &root_graph = root_graph_builder.GetGraph();
  EXPECT_EQ(AttrUtils::SetBool(root_graph, ATTR_NAME_IS_ROOT_GRAPH, true), true);
  EXPECT_EQ(AttrUtils::SetStr(root_graph, ATTR_NAME_PARENT_GRAPH_NAME, root_graph->GetName()), true);
  auto ret = GraphUtils::DumpGEGraphByPath(root_graph, "./subgraph_0.txt", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ(ret, 0);

  // build case sub graph
  auto case_sub_builder = ut::GraphBuilder("case_sub");
  const auto &case_data = case_sub_builder.AddNode("case_data", DATA, 0, 1);
  AttrUtils::SetInt(case_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &case_squeeze = case_sub_builder.AddNode("case_squeeze", SQUEEZE, 1, 1);
  const auto &case_netoutput = case_sub_builder.AddNode("case_netoutput", NETOUTPUT, 1, 1);
  case_sub_builder.AddDataEdge(case_data, 0, case_squeeze, 0);
  case_sub_builder.AddDataEdge(case_squeeze, 0, case_netoutput, 0);
  const auto &case_sub_graph = case_sub_builder.GetGraph();
  case_sub_graph->SetParentNode(case_0);
  case_sub_graph->SetParentGraph(root_graph);
  EXPECT_EQ(AttrUtils::SetStr(case_sub_graph, ATTR_NAME_PARENT_GRAPH_NAME, case_sub_graph->GetName()), true);
  ret = GraphUtils::DumpGEGraphByPath(case_sub_graph, "./subgraph_1.txt", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ(ret, 0);

  ComputeGraphPtr com_graph0 = std::make_shared<ComputeGraph>("TestGraph0");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  ASSERT_EQ(GraphUtils::LoadGEGraph("./subgraph_0.txt", *com_graph0), true);
  ASSERT_EQ(GraphUtils::LoadGEGraph("./subgraph_1.txt", *com_graph1), true);

  std::map<int64_t, std::string> options;
  options.emplace(0, "./subgraph_0.txt");
  options.emplace(1, "./subgraph_1.txt");
  Graph g;
  EXPECT_EQ(TuningUtils::ConvertFileToGraph(options, g), SUCCESS);

  options.clear();
  EXPECT_EQ(TuningUtils::ConvertFileToGraph(options, g), FAILED);
}

TEST_F(UtestTuningUtils, MergeSubGraph) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto graph = builder.GetGraph();
    EXPECT_EQ(TuningUtils::MergeSubGraph(graph), SUCCESS);
}

TEST_F(UtestTuningUtils, MergeSubGraph_End) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("end", END, 1, 1);
    auto graph = builder.GetGraph();
    EXPECT_EQ(TuningUtils::MergeSubGraph(graph), FAILED);
}

TEST_F(UtestTuningUtils, MergeSubGraph_Valid) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("data", DATA, 1, 1);
    auto graph = builder.GetGraph();
    AttrUtils::SetStr(node0->GetOpDesc(), "_peerNodeName", "Hello world");
    EXPECT_EQ(TuningUtils::MergeSubGraph(graph), SUCCESS);
}

TEST_F(UtestTuningUtils, MergeSubGraph_Netoutput) {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("net", NETOUTPUT, 1, 1);
    auto graph = builder.GetGraph();
    std::vector<std::string> val;
    val.push_back("Hello world");
    AttrUtils::SetListStr(node0->GetOpDesc(), "_aliasName", val);
    EXPECT_EQ(TuningUtils::MergeSubGraph(graph), SUCCESS);
}

TEST_F(UtestTuningUtils, FindNode) {
    int64_t in_index;
    EXPECT_EQ(TuningUtils::FindNode("Data0", in_index), nullptr);
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto node0 = builder.AddNode("Data0", "Data", 1, 1);
    auto graph = builder.GetGraph();
    TuningUtils::netoutput_nodes_.push_back(nullptr);
    TuningUtils::netoutput_nodes_.push_back(node0);
    EXPECT_EQ(TuningUtils::FindNode("Data0", in_index), nullptr);
    AttrUtils::SetListStr(node0->GetOpDesc(), "_aliasName", {"Data0", "str2", "str3"});
    AttrUtils::SetListInt(node0->GetOpDesc(), "_aliasIndexes", {0, 1, 2});
    EXPECT_NE(TuningUtils::FindNode("Data0", in_index), nullptr);
}

TEST_F(UtestTuningUtils, ConvertConstToWeightAttr) {
    auto builder = ut::GraphBuilder("root");
    const auto &placeholder_0 = builder.AddNode("placeholder_0", PLACEHOLDER, 0, 1);
    const auto &placeholder_1 = builder.AddNode("placeholder_1", PLACEHOLDER, 1, 1);
    std::map<int, ge::GeTensorPtr> tmap;
    tmap[10] = std::make_shared<GeTensor>();
    OpDescUtils::SetWeights(*placeholder_0, tmap);
    builder.AddDataEdge(placeholder_0, 0, placeholder_1, 0);
    const auto &graph = builder.GetGraph();
    EXPECT_EQ(TuningUtils::ConvertConstToWeightAttr(graph), SUCCESS);
    EXPECT_EQ(OpDescUtils::SetWeights(placeholder_0->GetOpDesc(), std::make_shared<GeTensor>()), GRAPH_SUCCESS);
    auto weight = OpDescUtils::MutableWeights(placeholder_0);
    EXPECT_EQ(weight.empty(), false);
    EXPECT_EQ(TuningUtils::ConvertConstToWeightAttr(graph), SUCCESS);
}

TEST_F(UtestTuningUtils, DumpGraphToPath) {
    EXPECT_NO_THROW(
        auto builder = ut::GraphBuilder("root");
        const auto &placeholder_0 = builder.AddNode("placeholder_0", PLACEHOLDER, 0, 1);
        const auto &placeholder_1 = builder.AddNode("placeholder_1", PLACEHOLDER, 1, 1);
        builder.AddDataEdge(placeholder_0, 0, placeholder_1, 0);
        const auto &graph = builder.GetGraph();
        TuningUtils::DumpGraphToPath(graph, 1, true, "path");
        TuningUtils::DumpGraphToPath(graph, 1, false, "path");
    );
}

TEST_F(UtestTuningUtils, RemoveDataNetoutputEdge) {
    auto builder = ut::GraphBuilder("root");
    const auto &placeholder_0 = builder.AddNode("placeholder_0", PLACEHOLDER, 0, 1);
    const auto &placeholder_1 = builder.AddNode("placeholder_1", PLACEHOLDER, 1, 1);
    const auto netoutput_1 = builder.AddNode("netoutput_1", NETOUTPUT, 1, 1);
    const auto noopnode = builder.AddNode("netoutput_1NoOp", NOOP, 1, 1);
    const auto netoutput_2 = builder.AddNode("netoutput_2", NETOUTPUT, 1, 1);
    builder.AddDataEdge(placeholder_0, 0, placeholder_1, 0);
    auto graph = builder.GetGraph();
    TuningUtils::data_node_2_end_node_[placeholder_0] = "placeholder_0";
    TuningUtils::data_node_2_end_node_[placeholder_1] = "placeholder_1";
    EXPECT_EQ(TuningUtils::RemoveDataNetoutputEdge(graph), PARAM_INVALID);
    TuningUtils::data_node_2_end_node_.clear();
    TuningUtils::data_node_2_end_node_[netoutput_1] = "netoutput_1";
    std::vector<std::string> out_alias_name;
    out_alias_name.push_back("netoutput_1");
    AttrUtils::SetListStr(netoutput_1->GetOpDesc(), "_aliasName", out_alias_name);
    std::vector<int64_t> alias_indexes;
    alias_indexes.push_back(-1);
    AttrUtils::SetListInt(netoutput_1->GetOpDesc(), "_aliasIndexes", alias_indexes);
    TuningUtils::netoutput_nodes_.push_back(netoutput_1);
    int64_t index = 0;
    auto n = TuningUtils::FindNode("netoutput_1", index);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(netoutput_1->GetInControlAnchor()->LinkFrom(noopnode->GetOutControlAnchor()), GRAPH_SUCCESS);
    EXPECT_EQ(noopnode->GetInControlAnchor()->LinkFrom(netoutput_2->GetOutControlAnchor()), GRAPH_SUCCESS);
    EXPECT_EQ(TuningUtils::RemoveDataNetoutputEdge(graph), GRAPH_SUCCESS);
}

TEST_F(UtestTuningUtils, RemoveDataNetoutputEdge_FindNode) {
    auto builder = ut::GraphBuilder("root");
    const auto &placeholder_0 = builder.AddNode("placeholder_0", PLACEHOLDER, 0, 1);
    const auto &placeholder_1 = builder.AddNode("placeholder_1", PLACEHOLDER, 1, 1);
    builder.AddDataEdge(placeholder_0, 0, placeholder_1, 0);
    TuningUtils::data_node_2_end_node_[placeholder_0] = "placeholder_0";
    TuningUtils::netoutput_nodes_.push_back(placeholder_0);
    AttrUtils::SetListStr(placeholder_0->GetOpDesc(), "_aliasName", {"placeholder_0"});
    AttrUtils::SetListInt(placeholder_0->GetOpDesc(), "_aliasIndexes", {0});
    int64_t in_index0;
    EXPECT_NE(TuningUtils::FindNode("placeholder_0", in_index0), nullptr);
    EXPECT_EQ(placeholder_0->AddLinkFrom(placeholder_1), GRAPH_SUCCESS);
    auto graph = builder.GetGraph();
    EXPECT_EQ(TuningUtils::RemoveDataNetoutputEdge(graph), SUCCESS);
}

TEST_F(UtestTuningUtils, MergeAllSubGraph) {
    auto builder0 = ut::GraphBuilder("sub0");
    const auto &placeholder_0 = builder0.AddNode("placeholder_0", PLACEHOLDER, 0, 1);
    const auto &placeholder_1 = builder0.AddNode("placeholder_1", PLACEHOLDER, 1, 1);
    auto graph0 = builder0.GetGraph();
    auto builder1 = ut::GraphBuilder("sub1");
    const auto &placeholder_2 = builder1.AddNode("placeholder_2", PLACEHOLDER, 0, 1);
    const auto &placeholder_3 = builder1.AddNode("placeholder_3", PLACEHOLDER, 1, 1);
    auto graph1 = builder1.GetGraph();
    std::vector<ComputeGraphPtr> vec;
    vec.push_back(graph0);
    vec.push_back(graph1);
    auto output_builder = ut::GraphBuilder("output");
    auto output_graph = output_builder.GetGraph();
    TuningUtils::merged_graph_nodes_.push_back(placeholder_0);
    TuningUtils::merged_graph_nodes_.push_back(placeholder_1);
    TuningUtils::merged_graph_nodes_.push_back(placeholder_2);
    TuningUtils::merged_graph_nodes_.push_back(placeholder_3);
    EXPECT_EQ(TuningUtils::MergeAllSubGraph(vec, output_graph), GRAPH_FAILED);
    vec.clear();
    EXPECT_EQ(TuningUtils::MergeAllSubGraph(vec, output_graph), SUCCESS);
    std::vector<std::string> vals;
    vals.push_back("1");
    vals.push_back("2");
    AttrUtils::SetListStr(placeholder_0->GetOpDesc(), ATTR_NAME_NEED_RECOVER_ATTR, vals);
    EXPECT_EQ(TuningUtils::MergeAllSubGraph(vec, output_graph), SUCCESS);
}

TEST_F(UtestTuningUtils, WeightExternalizationAndRecover) {
  std::vector<ComputeGraphPtr> tuning_subgraphs;
  auto builder_tune = ut::GraphBuilder("tune_graph");
  const auto pld0 = builder_tune.AddNode("pld0", PLACEHOLDER, 0, 1);
  AttrUtils::SetStr(pld0->GetOpDesc(), "_parentNodeName", "Const0");
  AttrUtils::SetStr(pld0->GetOpDesc(), "_peerNodeName", "Const0");
  AttrUtils::SetStr(pld0->GetOpDesc(), "parentOpType", CONSTANT);
  AttrUtils::SetInt(pld0->GetOpDesc(), "anchorIndex", 0);
  const auto pld1 = builder_tune.AddNode("pld1", PLACEHOLDER, 0, 1);
  AttrUtils::SetStr(pld1->GetOpDesc(), "_parentNodeName", "Const0");
  AttrUtils::SetStr(pld1->GetOpDesc(), "_peerNodeName", "Const0");
  AttrUtils::SetStr(pld1->GetOpDesc(), "parentOpType", CONSTANT);
  AttrUtils::SetInt(pld1->GetOpDesc(), "anchorIndex", 0);
  const auto pld2 = builder_tune.AddNode("pld2", PLACEHOLDER, 0, 1);
  AttrUtils::SetStr(pld2->GetOpDesc(), "_parentNodeName", "Const1");
  AttrUtils::SetStr(pld2->GetOpDesc(), "_peerNodeName", "Const1");
  AttrUtils::SetStr(pld2->GetOpDesc(), "parentOpType", CONSTANT);
  AttrUtils::SetInt(pld2->GetOpDesc(), "anchorIndex", 0);
  const auto pld3 = builder_tune.AddNode("pld3", PLACEHOLDER, 0, 1);
  AttrUtils::SetStr(pld3->GetOpDesc(), "_parentNodeName", "Const2");
  AttrUtils::SetStr(pld3->GetOpDesc(), "_peerNodeName", "Const2");
  AttrUtils::SetStr(pld3->GetOpDesc(), "parentOpType", CONSTANT);
  AttrUtils::SetInt(pld3->GetOpDesc(), "anchorIndex", 0);
  const auto addn = builder_tune.AddNode("addn", "AddN", 4, 1);
  const auto end0 = builder_tune.AddNode("end0", END, 1, 0);
  builder_tune.AddDataEdge(pld0, 0, addn, 0);
  builder_tune.AddDataEdge(pld1, 0, addn, 1);
  builder_tune.AddDataEdge(pld2, 0, addn, 2);
  builder_tune.AddDataEdge(pld3, 0, addn, 3);
  builder_tune.AddDataEdge(addn, 0, end0, 0);
  auto tune_graph = builder_tune.GetGraph();
  tuning_subgraphs.push_back(tune_graph);

  std::vector<ComputeGraphPtr> non_tuning_subgraphs;
  auto builder_const = ut::GraphBuilder("const_graph");
  const auto const0 = builder_const.AddNode("Const0", CONSTANT, 0, 1);
  const auto const1 = builder_const.AddNode("Const1", CONSTANT, 0, 1);
  const auto const2 = builder_const.AddNode("Const2", CONSTANT, 0, 1);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  EXPECT_EQ(ge::OpDescUtils::SetWeights(const0, {tensor}), 0);
  EXPECT_EQ(ge::OpDescUtils::SetWeights(const1, {tensor}), 0);

  ge::GeTensorPtr empty_tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> empty_value;
  std::vector<int64_t> empty_shape{0};
  empty_tensor->MutableTensorDesc().SetShape(GeShape(empty_shape));
  empty_tensor->SetData(empty_value);
  empty_tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  EXPECT_EQ(ge::OpDescUtils::SetWeights(const2, {empty_tensor}), 0);

  const auto netoutput = builder_const.AddNode("netoutput0", NETOUTPUT, 3, 0);
  builder_const.AddDataEdge(const0, 0, netoutput, 0);
  builder_const.AddDataEdge(const1, 0, netoutput, 1);
  builder_const.AddDataEdge(const2, 0, netoutput, 2);
  auto nongraph = builder_const.GetGraph();
  non_tuning_subgraphs.push_back(nongraph);
  pld0->GetOpDesc()->SetExtAttr("parentNode", const0);
  pld1->GetOpDesc()->SetExtAttr("parentNode", const0);
  pld2->GetOpDesc()->SetExtAttr("parentNode", const1);
  pld3->GetOpDesc()->SetExtAttr("parentNode", const2);

  EXPECT_EQ(TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs, true, "./"), GRAPH_SUCCESS);
  EXPECT_EQ(TuningUtils::reusable_weight_files_.size(), 1U);
  EXPECT_EQ(TuningUtils::hash_to_files_.size(), 1U);
  const auto file_const1 = tune_graph->FindFirstNodeMatchType(FILECONSTANT);
  EXPECT_NE(file_const1, nullptr);
  const auto real_const1 = tune_graph->FindFirstNodeMatchType(CONSTANT);
  EXPECT_NE(real_const1, nullptr);
  const auto file_const2 = nongraph->FindFirstNodeMatchType(FILECONSTANT);
  EXPECT_NE(file_const2, nullptr);
  const auto real_const2 = nongraph->FindFirstNodeMatchType(CONSTANT);
  EXPECT_NE(real_const2, nullptr);

  ComputeGraphPtr load_aicore = std::make_shared<ComputeGraph>("TestGraph0");
  ComputeGraphPtr load_const = std::make_shared<ComputeGraph>("TestGraph1");
  EXPECT_EQ(GraphUtils::LoadGEGraph("./aicore_subgraph_0.txt", load_aicore), true);
  EXPECT_EQ(GraphUtils::LoadGEGraph("./subgraph_0.txt", load_const), true);
  const auto recover_const1 = load_aicore->FindFirstNodeMatchType(CONSTANT);
  EXPECT_NE(recover_const1, nullptr);
  const auto file_const3 = load_aicore->FindFirstNodeMatchType(FILECONSTANT);
  EXPECT_EQ(file_const3, nullptr);
  const auto recover_const2 = load_const->FindFirstNodeMatchType(CONSTANT);
  EXPECT_NE(recover_const2, nullptr);
  const auto file_const4 = load_const->FindFirstNodeMatchType(FILECONSTANT);
  EXPECT_EQ(file_const4, nullptr);

  system("rm -rf ./tmp_weight_*");
  system("rm -rf ./aicore_subgraph_*");
  system("rm -rf ./subgraph_*");
}
} // namespace ge
