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
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "exe_graph/runtime/context_extend.h"
#include "graph/utils/node_utils.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
class BgTest : public testing::Test {
 protected:
  void SetUp() override {
    Test::SetUp();
    bg::ValueHolder::PushGraphFrame();
  }
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame())
      ;
  }

 public:
  void CheckGraphGenerally(const ge::ComputeGraph &graph) {
    CheckNamesUniq(graph);
    CheckOwners(graph);
    CheckSubgraphExists(graph);
    CheckDataIndexOnAllSubgraphs(graph);
  }
  void CheckGraphGenerally(const ge::ExecuteGraph *graph) {
    CheckNamesUniq(graph);
    CheckOwners(graph);
    CheckSubgraphExists(graph);
    CheckDataIndexOnAllSubgraphs(graph);
  }

  void CheckExeGraphGenerally(const ge::ComputeGraph &graph) {
    CheckGraphGenerally(graph);
    CheckKernelExtendInfoOk(graph);
  }
  void CheckExeGraphGenerally(const ge::ExecuteGraph *graph) {
    CheckGraphGenerally(graph);
    CheckKernelExtendInfoOk(graph);
  }

  void CheckComputeNodeInfoOk(const ge::ExecuteGraph &root_graph,
                              const std::map<bg::ValueHolderPtr, ge::NodePtr> &holders_to_compute_node) {
    ge::Buffer buffer;
    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kComputeNodeInfo, buffer));
    auto compute_nodes_info = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(compute_nodes_info, nullptr);

    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kBuffer, buffer));
    auto exe_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(exe_buffer, nullptr);

    std::map<const ge::FastNode *, ge::NodePtr> exe_nodes_to_compute_node;
    for (const auto &holder_to_cnode : holders_to_compute_node) {
      exe_nodes_to_compute_node[holder_to_cnode.first->GetFastNode()] = holder_to_cnode.second;
    }

    for (const auto &exe_node : root_graph.GetAllNodes()) {
      auto enode_to_cnode = exe_nodes_to_compute_node.find(exe_node);
      if (enode_to_cnode == exe_nodes_to_compute_node.end()) {
        continue;
      }
      auto cnode = enode_to_cnode->second;

      int64_t index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(exe_node->GetOpDescBarePtr(), kComputeNodeIndex, index))
          << "The node " << exe_node->GetName() << " Type " << exe_node->GetType()
          << " does not have attr ComputeNodeIndex";

      auto compute_node_info = compute_nodes_info->Get<ComputeNodeInfo>(index);
      ASSERT_NE(compute_node_info, nullptr)
          << "The node " << exe_node->GetName() << " Type " << exe_node->GetType() << " ComputeNodeIndex invalid, "
          << index << ", total num " << compute_nodes_info->GetNum();

      auto name_index = reinterpret_cast<size_t>(compute_node_info->GetNodeName());
      auto name = exe_buffer->Get<char>(name_index);
      ASSERT_NE(name, nullptr) << "The node " << exe_node->GetName() << " Type " << exe_node->GetType()
                               << " does not have a valid name index";
      EXPECT_STREQ(cnode->GetName().c_str(), name);

      auto type_index = reinterpret_cast<size_t>(compute_node_info->GetNodeType());
      auto type = exe_buffer->Get<char>(type_index);
      ASSERT_NE(type, nullptr) << "The node " << exe_node->GetName() << " Type " << exe_node->GetType()
                               << " does not have a valid type index";
      EXPECT_STREQ(cnode->GetType().c_str(), type);

      // todo check input, output, attrs...
    }
  }

  void HasControlEdge(const ge::ComputeGraph &graph, const ge::Node &src_node, const ge::Node &dst_node) {
    auto src_anchor = src_node.GetOutControlAnchor();
    auto dst_anchor = dst_node.GetInControlAnchor();
    EXPECT_TRUE(src_anchor->IsLinkedWith(dst_anchor));
  }
  void HasControlEdge(const ge::ExecuteGraph *graph, const ge::FastNode *src_node, const ge::FastNode *dst_node) {
    bool flag = false;
    for (const auto edge : src_node->GetAllOutControlEdgesRef()) {
      if (edge == nullptr) {
        continue;
      }
      if (edge->dst == dst_node) {
        flag = true;
        break;
      }
    }
    EXPECT_EQ(flag, true);
  }

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

    void ConnectFromInitToDeInit(const ge::Node *init_node, int32_t src_index, const ge::Node *main_node,
                                 int32_t dst_index) {
      ConnectFromInit(init_node, src_index, main_node, dst_index, "DeInit");
    }
    void ConnectFromInitToDeInit(const ge::FastNode *init_node, int32_t src_index, const ge::FastNode *main_node,
                                 int32_t dst_index) {
      ConnectFromInit(init_node, src_index, main_node, dst_index, "DeInit");
    }
 private:
  void CheckDataIndex(const ge::ComputeGraph &graph) {
    std::map<int32_t, std::string> indexes_to_name;
    for (const auto &node : graph.GetDirectNode()) {
      if (node->GetType() == "Data" || node->GetType() == "InnerData") {
        int32_t index;
        ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), "index", index))
            << "Can not get index attr on data " << node->GetName();
        ASSERT_TRUE(indexes_to_name.emplace(index, node->GetName()).second)
            << "Duplicated index on data " << node->GetName() << " and data " << indexes_to_name[index] << ", on graph "
            << graph.GetName();
      }
    }
  }
  void CheckDataIndex(const ge::ExecuteGraph *graph) {
    std::map<int32_t, std::string> indexes_to_name;
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() == "Data" || node->GetType() == "InnerData") {
        int32_t index;
        ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", index))
            << "Can not get index attr on data " << node->GetName();
        ASSERT_TRUE(indexes_to_name.emplace(index, node->GetName()).second)
            << "Duplicated index on data " << node->GetName() << " and data " << indexes_to_name[index] << ", on graph "
            << graph->GetName();
      }
    }
  }

  void CheckDataIndexOnAllSubgraphs(const ge::ComputeGraph &graph) {
    CheckDataIndex(graph);
    for (const auto &subgraph : graph.GetAllSubgraphs()) {
      CheckDataIndex(*subgraph);
    }
  }
  void CheckDataIndexOnAllSubgraphs(const ge::ExecuteGraph *graph) {
    CheckDataIndex(graph);
    for (const auto &subgraph : graph->GetAllSubgraphs()) {
      CheckDataIndex(subgraph);
    }
  }

  void CheckKernelExtendInfoOk(const ge::ComputeGraph &root_graph) {
    ge::Buffer buffer;
    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kKernelExtendInfo, buffer));
    auto extend_info_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(extend_info_buffer, nullptr);

    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kBuffer, buffer));
    auto exe_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(exe_buffer, nullptr);

    std::set<std::string> nodes_does_have_extend_info = {"NetOutput"};
    for (const auto &node : root_graph.GetAllNodes()) {
      if (nodes_does_have_extend_info.count(node->GetType()) > 0) {
        continue;
      }
      int64_t index;
      EXPECT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), kKernelExtendIndex, index))
          << "The node " << node->GetName() << " Type " << node->GetType() << " does not have a KernelExtendInfoIndex";
      ASSERT_LT(static_cast<size_t>(index), extend_info_buffer->GetNum())
          << "The node " << node->GetName() << " Type " << node->GetType()
          << " KernelExtendInfoIndex out of range: " << index << " > " << extend_info_buffer->GetNum();
      auto extend_info = extend_info_buffer->Get<KernelExtendInfo>(index);
      ASSERT_NE(extend_info, nullptr);

      auto name_index = reinterpret_cast<bg::BufferPool::BufId>(extend_info->GetKernelName());
      auto name = exe_buffer->Get<char>(name_index);
      ASSERT_NE(name, nullptr) << "The node " << node->GetName() << " Type " << node->GetType()
                               << " does not have a valid name index";
      auto type_index = reinterpret_cast<bg::BufferPool::BufId>(extend_info->GetKernelType());
      auto type = exe_buffer->Get<char>(type_index);
      ASSERT_NE(type, nullptr) << "The node " << node->GetName() << " Type " << node->GetType()
                               << " does not have a valid type index";

      EXPECT_STREQ(node->GetName().c_str(), name);
      EXPECT_STREQ(node->GetType().c_str(), type);
    }
  }
  void CheckKernelExtendInfoOk(const ge::ExecuteGraph *root_graph) {
    ge::Buffer buffer;
    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph->shared_from_this(), kKernelExtendInfo, buffer));
    auto extend_info_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(extend_info_buffer, nullptr);

    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph->shared_from_this(), kBuffer, buffer));
    auto exe_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(exe_buffer, nullptr);

    std::set<std::string> nodes_does_have_extend_info = {"NetOutput"};
    for (const auto &node : root_graph->GetAllNodes()) {
      if (nodes_does_have_extend_info.count(node->GetType()) > 0) {
        continue;
      }
      int64_t index;
      EXPECT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kKernelExtendIndex, index))
          << "The node " << node->GetName() << " Type " << node->GetType() << " does not have a KernelExtendInfoIndex";
      ASSERT_LT(static_cast<size_t>(index), extend_info_buffer->GetNum())
          << "The node " << node->GetName() << " Type " << node->GetType()
          << " KernelExtendInfoIndex out of range: " << index << " > " << extend_info_buffer->GetNum();
      auto extend_info = extend_info_buffer->Get<KernelExtendInfo>(index);
      ASSERT_NE(extend_info, nullptr);

      auto name_index = reinterpret_cast<bg::BufferPool::BufId>(extend_info->GetKernelName());
      auto name = exe_buffer->Get<char>(name_index);
      ASSERT_NE(name, nullptr) << "The node " << node->GetName() << " Type " << node->GetType()
                               << " does not have a valid name index";
      auto type_index = reinterpret_cast<bg::BufferPool::BufId>(extend_info->GetKernelType());
      auto type = exe_buffer->Get<char>(type_index);
      ASSERT_NE(type, nullptr) << "The node " << node->GetName() << " Type " << node->GetType()
                               << " does not have a valid type index";

      EXPECT_STREQ(node->GetName().c_str(), name);
      EXPECT_STREQ(node->GetType().c_str(), type);
    }
  }

  void CheckOwners(const ge::ComputeGraph &root_graph) {
    auto &non_const_graph = const_cast<ge::ComputeGraph &>(root_graph);
    for (const auto &node : root_graph.GetDirectNode()) {
      EXPECT_EQ(node->GetOwnerComputeGraph(), non_const_graph.shared_from_this());
      EXPECT_EQ(non_const_graph.GetParentNode(), nullptr);
      EXPECT_EQ(non_const_graph.GetParentGraph(), nullptr);
    }
    for (const auto &subgraph : root_graph.GetAllSubgraphs()) {
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

  void CheckSubgraphExists(const ge::ComputeGraph &root_graph) {
    for (const auto &node : root_graph.GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
        EXPECT_NE(root_graph.GetSubgraph(subgraph_name), nullptr)
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

  void CheckSubgraphAndParentNodeIoNum(const ge::ComputeGraph &root_graph) {
    for (const auto &node : root_graph.GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
        auto subgraph = root_graph.GetSubgraph(subgraph_name);
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

  void CheckNamesUniq(const ge::ComputeGraph &graph) {
    std::set<std::string> node_names;
    for (const auto &node : graph.GetAllNodes()) {
      node_names.emplace(node->GetName());
    }
    EXPECT_EQ(node_names.size(), graph.GetAllNodesSize());
  }
  void CheckNamesUniq(const ge::ExecuteGraph *graph) {
    std::set<std::string> node_names;
    for (const auto node : graph->GetAllNodes()) {
      node_names.emplace(node->GetName());
    }
    EXPECT_EQ(node_names.size(), graph->GetAllNodes().size());
  }
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_
