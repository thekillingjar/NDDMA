/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_
#include <gtest/gtest.h>
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
namespace gert {
namespace bg {
class BgTest : public testing::Test {
 public:
  void ConnectFromInit(const ge::Node *src_node, int32_t src_index, const ge::Node *dst_node, int32_t dst_index,
                       const char *dst_graph_type) {
    ge::InDataAnchorPtr top_dst_anchor = dst_node->GetInDataAnchor(dst_index);
    ASSERT_NE(top_dst_anchor, nullptr);
    ge::OutDataAnchorPtr top_src_anchor;
    while (true) {
      top_src_anchor = top_dst_anchor->GetPeerOutAnchor();
      ASSERT_NE(top_src_anchor, nullptr);

      auto tmp_src_node = top_src_anchor->GetOwnerNode();
      ASSERT_NE(tmp_src_node, nullptr);
      if (tmp_src_node->GetType() != "InnerData") {
        break;
      }
      int32_t tmp_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(tmp_src_node->GetOpDesc(), "index", tmp_index));

      auto tmp_graph = tmp_src_node->GetOwnerComputeGraph();
      ASSERT_NE(tmp_graph, nullptr);
      auto tmp_parent_node = tmp_graph->GetParentNode();
      ASSERT_NE(tmp_parent_node, nullptr);

      top_dst_anchor = tmp_parent_node->GetInDataAnchor(tmp_index);
      ASSERT_NE(top_dst_anchor, nullptr) << "Failed to get in anchor from node " << tmp_parent_node->GetName()
                                         << ", index " << tmp_index;
    }

    auto top_src_node = top_src_anchor->GetOwnerNode();
    ASSERT_NE(top_src_node, nullptr);
    ASSERT_EQ(top_src_node->GetType(), "Init");
    auto top_dst_node = top_dst_anchor->GetOwnerNode();
    ASSERT_NE(top_dst_node, nullptr);
    ASSERT_EQ(top_dst_node->GetType(), dst_graph_type);

    auto init_graph = ge::NodeUtils::GetSubgraph(*top_src_node, 0);
    ASSERT_NE(init_graph, nullptr);
    auto init_output = init_graph->FindFirstNodeMatchType("InnerNetOutput");
    ASSERT_NE(init_output, nullptr);
    auto dst_anchor_on_init = init_output->GetInDataAnchor(top_src_anchor->GetIdx());
    ASSERT_NE(dst_anchor_on_init, nullptr);
    auto src_anchor = dst_anchor_on_init->GetPeerOutAnchor();
    ASSERT_NE(src_anchor, nullptr);
    EXPECT_EQ(src_anchor->GetIdx(), src_index);
    EXPECT_EQ(src_anchor->GetOwnerNode().get(), src_node);
    EXPECT_EQ(src_anchor->GetOwnerNode()->GetName(), src_node->GetName());
  }
  void ConnectFromInit(const ge::FastNode *src_node, int32_t src_index, const ge::FastNode *dst_node,
                       int32_t dst_index, const char *dst_graph_type) {
    auto top_edge = dst_node->GetInDataEdgeByIndex(dst_index);
    ASSERT_NE(top_edge, nullptr);
    while (true) {
      auto tmp_src_node = top_edge->src;
      ASSERT_NE(tmp_src_node, nullptr);
      if (tmp_src_node->GetType() != "InnerData") {
        break;
      }
      int32_t tmp_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(tmp_src_node->GetOpDescBarePtr(), "index", tmp_index));

      auto tmp_graph = tmp_src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      ASSERT_NE(tmp_graph, nullptr);
      auto tmp_parent_node = tmp_graph->GetParentNodeBarePtr();
      ASSERT_NE(tmp_parent_node, nullptr);

      top_edge = tmp_parent_node->GetInDataEdgeByIndex(tmp_index);
      ASSERT_NE(top_edge, nullptr) << "Failed to get in edge point from node " << tmp_parent_node->GetName()
                                   << ", index " << tmp_index;
    }

    auto top_src_node = top_edge->src;
    ASSERT_NE(top_src_node, nullptr);
    ASSERT_EQ(top_src_node->GetType(), "Init");
    auto top_dst_node = top_edge->dst;
    ASSERT_NE(top_dst_node, nullptr);
    ASSERT_EQ(top_dst_node->GetType(), dst_graph_type);

    auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(top_src_node, 0);
    ASSERT_NE(init_graph, nullptr);
    auto init_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "InnerNetOutput");
    ASSERT_NE(init_output, nullptr);
    auto edge_on_init = init_output->GetInDataEdgeByIndex(top_edge->src_output);
    ASSERT_NE(edge_on_init, nullptr);
    EXPECT_EQ(edge_on_init->src_output, src_index);
    EXPECT_EQ(edge_on_init->src, src_node);
    EXPECT_EQ(edge_on_init->src->GetName(), src_node->GetName());
  }

  void ConnectFromInitToMain(const ge::Node *init_node, int32_t src_index, const ge::Node *main_node,
                             int32_t dst_index) {
    ConnectFromInit(init_node, src_index, main_node, dst_index, "Main");
  }
  void ConnectFromInitToMain(const ge::FastNode *init_node, int32_t src_index, const ge::FastNode *main_node,
                             int32_t dst_index) {
    ConnectFromInit(init_node, src_index, main_node, dst_index, "Main");
  }

  void ConnectFromInitToDeInit(const ge::Node *init_node, int32_t src_index, const ge::Node *de_init_node,
                               int32_t dst_index) {
    ConnectFromInit(init_node, src_index, de_init_node, dst_index, "DeInit");
  }
  void ConnectFromInitToDeInit(const ge::FastNode *init_node, int32_t src_index, const ge::FastNode *main_node,
                               int32_t dst_index) {
    ConnectFromInit(init_node, src_index, main_node, dst_index, "DeInit");
  }

  void CheckOwners(const ge::ComputeGraphPtr &root_graph) {
    for (const auto &node : root_graph->GetDirectNode()) {
      EXPECT_EQ(node->GetOwnerComputeGraph(), root_graph);
      EXPECT_EQ(root_graph->GetParentNode(), nullptr);
      EXPECT_EQ(root_graph->GetParentGraph(), nullptr);
    }
    for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
      auto owner_node = subgraph->GetParentNode();
      ASSERT_NE(owner_node, nullptr);
      EXPECT_EQ(owner_node->GetOwnerComputeGraph(), subgraph->GetParentGraph());
      for (const auto &node : subgraph->GetDirectNode()) {
        EXPECT_EQ(node->GetOwnerComputeGraph(), subgraph)
            << "The node " << node->GetName() << "(" << node->GetType() << ")"
            << " owner graph name " << node->GetOwnerComputeGraph()->GetName() << ", expect " << subgraph->GetName();
      }
    }
  }
  void CheckOwners(const ge::ExecuteGraph *root_graph) {
    for (const auto node : root_graph->GetDirectNode()) {
      EXPECT_EQ(node->GetExtendInfo()->GetOwnerGraphBarePtr(), root_graph);
      EXPECT_EQ(root_graph->GetParentNodeBarePtr(), nullptr);
      EXPECT_EQ(root_graph->GetParentGraphBarePtr(), nullptr);
    }
    for (const auto subgraph : root_graph->GetAllSubgraphs()) {
      auto owner_node = subgraph->GetParentNodeBarePtr();
      ASSERT_NE(owner_node, nullptr);
      EXPECT_EQ(owner_node->GetExtendInfo()->GetOwnerGraphBarePtr(), subgraph->GetParentGraphBarePtr());
      for (const auto node : subgraph->GetDirectNode()) {
        EXPECT_EQ(node->GetExtendInfo()->GetOwnerGraphBarePtr(), subgraph)
            << "The node " << node->GetName() << "(" << node->GetType() << ")"
            << " owner graph name " << node->GetExtendInfo()->GetOwnerGraphBarePtr()->GetName() << ", expect "
            << subgraph->GetName();
      }
    }
  }

  void CheckSubgraphExists(const ge::ComputeGraphPtr &root_graph) {
    for (const auto &node : root_graph->GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
        EXPECT_NE(root_graph->GetSubgraph(subgraph_name), nullptr)
            << "Subgraph " << subgraph_name << " does not exists on root graph, node name " << node->GetName();
      }
    }
  }
  void CheckSubgraphExists(const ge::ExecuteGraph *root_graph) {
    for (const auto node : root_graph->GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDescBarePtr()->GetSubgraphInstanceNames()) {
        EXPECT_NE(root_graph->GetSubGraph(subgraph_name), nullptr)
                  << "Subgraph " << subgraph_name << " does not exists on root graph, node name " << node->GetName();
      }
    }
  }

  void CheckSubgraphAndParentNodeIoNum(const ge::ComputeGraphPtr &root_graph) {
    for (const auto &node : root_graph->GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
        auto subgraph = root_graph->GetSubgraph(subgraph_name);
        std::vector<ge::NodePtr> data_nodes;
        ge::NodePtr netoutput_node;
        for (const auto &subgraph_node : subgraph->GetDirectNode()) {
          if (subgraph_node->GetType() == "Data") {
            data_nodes.emplace_back(subgraph_node);
            continue;
          }
          if (subgraph_node->GetType() == "NetOutput") {
            ASSERT_EQ(netoutput_node, nullptr);
            netoutput_node = subgraph_node;
          }
        }
        ASSERT_NE(netoutput_node, nullptr);
        auto parent_node = subgraph->GetParentNode();
        ASSERT_NE(parent_node, nullptr);
        EXPECT_EQ(parent_node->GetAllInDataAnchorsSize(), data_nodes.size());
        EXPECT_EQ(parent_node->GetAllOutDataAnchorsSize(), netoutput_node->GetAllInDataAnchorsSize());
      }
    }
  }
  void CheckSubgraphAndParentNodeIoNum(const ge::ExecuteGraph *root_graph) {
    for (const auto node : root_graph->GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDescBarePtr()->GetSubgraphInstanceNames()) {
        auto subgraph = root_graph->GetSubGraph(subgraph_name);
        std::vector<ge::FastNode *> data_nodes;
        ge::FastNode *netoutput_node;
        for (const auto subgraph_node : subgraph->GetDirectNode()) {
          if (subgraph_node->GetType() == "Data") {
            data_nodes.emplace_back(subgraph_node);
            continue;
          }
          if (subgraph_node->GetType() == "NetOutput") {
            ASSERT_EQ(netoutput_node, nullptr);
            netoutput_node = subgraph_node;
          }
        }
        ASSERT_NE(netoutput_node, nullptr);
        auto parent_node = subgraph->GetParentNodeBarePtr();
        ASSERT_NE(parent_node, nullptr);
        EXPECT_EQ(parent_node->GetDataInNum(), data_nodes.size());
        EXPECT_EQ(parent_node->GetDataOutNum(), netoutput_node->GetDataInNum());
      }
    }
  }

  void CheckGraphGenerally(const ge::ComputeGraphPtr &graph) {
    CheckOwners(graph);
    CheckSubgraphExists(graph);
  }
  void CheckGraphGenerally(const ge::ExecuteGraph *graph) {
    CheckOwners(graph);
    CheckSubgraphExists(graph);
  }

  void GetSrcNodeFromInit(ge::FastNode *node, ge::FastNode *&src_node) {
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->GetType(), "InnerData");

    auto tmp_src_node = node;
    ge::Edge<ge::FastNode> *top_edge = nullptr;
    while (true) {
      ASSERT_NE(tmp_src_node, nullptr);
      if (tmp_src_node->GetType() != "InnerData") {
        break;
      }
      int32_t tmp_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(tmp_src_node->GetOpDescBarePtr(), "index", tmp_index));

      auto tmp_graph = tmp_src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      ASSERT_NE(tmp_graph, nullptr);
      auto tmp_parent_node = tmp_graph->GetParentNodeBarePtr();
      ASSERT_NE(tmp_parent_node, nullptr);

      top_edge = tmp_parent_node->GetInDataEdgeByIndex(tmp_index);
      ASSERT_NE(top_edge, nullptr) << "Failed to get in edge point from node " << tmp_parent_node->GetName()
                                   << ", index " << tmp_index;
      tmp_src_node = top_edge->src;
    }

    ASSERT_EQ(tmp_src_node->GetType(), "Init");
    auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(tmp_src_node, 0);
    ASSERT_NE(init_graph, nullptr);
    auto init_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "InnerNetOutput");
    ASSERT_NE(init_output, nullptr);
    auto edge_on_init = init_output->GetInDataEdgeByIndex(top_edge->src_output);
    ASSERT_NE(edge_on_init, nullptr);
    src_node = edge_on_init->src;
    ASSERT_NE(src_node, nullptr);
  }

 protected:
  void TearDown() override {
    Test::TearDown();
    while (ValueHolder::PopGraphFrame() != nullptr) {}
  }
};
class BgTestAutoCreateFrame : public BgTest {
 public:
 protected:
  void SetUp() override {
    Test::SetUp();
    ValueHolder::PushGraphFrame();
  }

  void Create3StageFrames(std::unique_ptr<GraphFrame> &init_frame, std::unique_ptr<GraphFrame> &de_init_frame) {
    bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
  }

  void Create3StageFrames() {
    std::unique_ptr<GraphFrame> init_frame, de_init_frame;
    Create3StageFrames(init_frame, de_init_frame);
  }
};
class BgTestAutoCreate3StageFrame : public BgTestAutoCreateFrame {
 protected:
  void SetUp() override {
    BgTestAutoCreateFrame::SetUp();
    Create3StageFrames(init_frame_, de_init_frame_);
  }
  std::unique_ptr<GraphFrame> init_frame_;
  std::unique_ptr<GraphFrame> de_init_frame_;
};
}

}
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_
