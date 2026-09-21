/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/task_node_map_faker.h"
#include <memory>
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
namespace ge {
namespace {
NodePtr CreateData(const ComputeGraphPtr &graph) {
  auto op_desc = std::make_shared<OpDesc>("data0", "Data");
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddOutputDesc(GeTensorDesc());
  return graph->AddNode(op_desc);
}
}  // namespace
TaskNodeMap TaskNodeMapFaker::Build() const {
  std::map<size_t, int64_t> task_indexes_to_node_id;
  std::map<int64_t, std::vector<size_t>> node_ids_to_task_indexes;
  auto graph = FakeGraph(task_indexes_to_node_id, node_ids_to_task_indexes);
  GE_DUMP(graph, "snut");
  TaskNodeMap tn_map;
  tn_map.Init(graph, task_indexes_to_node_id.size());
  for (const auto &task_index_and_node_id : task_indexes_to_node_id) {
    tn_map.AddRelation(task_index_and_node_id.first, task_index_and_node_id.second);
  }
  return tn_map;
}
TaskNodeMapFaker &TaskNodeMapFaker::RangeOneOneMap(size_t start_task_index, size_t count) {
  task_num_ = std::max(task_num_, start_task_index + count);
  return *this;
}
NodePtr TaskNodeMapFaker::FakeNode(const ComputeGraphPtr &graph, size_t task_index) const {
  std::string name = "FakeNode/" + std::to_string(task_index);
  std::string node_type = "FakeType";
  auto iter = task_indexes_to_node_type_.find(task_index);
  if (iter != task_indexes_to_node_type_.end()) {
    node_type = iter->second;
  }
  auto op_desc = std::make_shared<OpDesc>(name, node_type);

  for (size_t ini = 0UL; ini < task_indexes_to_param_.at(task_index).parsed_input_addrs.size(); ++ini) {
    op_desc->AddInputDesc(GeTensorDesc());
  }
  for (size_t oi = 0UL; oi < task_indexes_to_param_.at(task_index).parsed_output_addrs.size(); ++oi) {
    op_desc->AddOutputDesc(GeTensorDesc());
  }
  return graph->AddNode(op_desc);
}
/*
 *     +----------> NetOutput <-------------+
 *     |                |                   |
 *     |            Identity/n...           |
 *     |                |                   |
 * FakeNode/n....   FakeNode/n...     FakeNode/n
 *     |                |                   |
 * Identity/n.....      |                   |
 *     |                |                   |
 *     +-------------  data   --------------+
 */
ComputeGraphPtr TaskNodeMapFaker::FakeGraph(map<size_t, int64_t> &task_indexes_to_node_id,
                                            map<int64_t, std::vector<size_t>> &node_ids_to_task_indexes) const {
  auto graph = std::make_shared<ComputeGraph>("faked_graph");
  auto data_node = CreateData(graph);
  auto task_indexes_to_in_identity_nodes = CreateInIdentityNodes(graph);

  // step 1. 创建nodes，连边data
  std::vector<NodePtr> fake_nodes;
  for (size_t i = 0UL; i < task_num_; ++i) {  // task当前与node是一一映射，因此task的数量就是node的数量
    auto node = FakeNode(graph, i);
    task_indexes_to_node_id[i] = node->GetOpDesc()->GetId();
    node_ids_to_task_indexes[node->GetOpDesc()->GetId()].push_back(i);

    // 输入连边
    int32_t identity_i = 0;
    for (size_t in_i = 0UL; in_i < node->GetAllInDataAnchorsSize(); ++in_i) {
      auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(in_i));
      if (task_indexes_to_in_identity_indexes_.count(i) > 0) {
        if (task_indexes_to_in_identity_indexes_.at(i).count(in_i) > 0) {
          GraphUtils::AddEdge(task_indexes_to_in_identity_nodes.at(i).at(identity_i++)->GetOutDataAnchor(0), in_anchor);
          continue;
        }
      }
      GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), in_anchor);
    }

    fake_nodes.emplace_back(std::move(node));
  }

  // step 2. 按需创建identity，连边fake nodes
  std::vector<OutDataAnchorPtr> out_anchors;
  for (const auto &node : fake_nodes) {
    auto task_index = node_ids_to_task_indexes[node->GetOpDesc()->GetId()].at(0);
    for (const auto &src_anchor : node->GetAllOutDataAnchors()) {
      if (task_indexes_to_out_identity_indexes_.count(task_index) > 0UL) {
        if (task_indexes_to_out_identity_indexes_.at(task_index).count(src_anchor->GetIdx()) > 0UL) {
          auto identity_op_desc = std::make_shared<OpDesc>(
              "Identity/" + std::to_string(task_index) + "/out/" + std::to_string(src_anchor->GetIdx()), "Identity");
          identity_op_desc->AddInputDesc(GeTensorDesc());
          identity_op_desc->AddOutputDesc(GeTensorDesc());
          auto identity_node = graph->AddNode(identity_op_desc);
          GraphUtils::AddEdge(src_anchor, identity_node->GetInDataAnchor(0));
          out_anchors.emplace_back(identity_node->GetOutDataAnchor(0));
          continue;
        }
      }
      out_anchors.emplace_back(src_anchor);
    }
  }

  // step 3. 创建netoutput
  auto op_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  for (size_t i = 0; i < out_anchors.size(); ++i) {
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
  }
  auto out_node = graph->AddNode(op_desc);
  for (size_t i = 0UL; i < out_anchors.size(); ++i) {
    GraphUtils::AddEdge(out_anchors[i], out_node->GetInDataAnchor(static_cast<int32_t>(i)));
  }

  return graph;
}
TaskNodeMapFaker &TaskNodeMapFaker::NodeInFromIdentity(size_t task_index, int32_t in_index) {
  task_indexes_to_in_identity_indexes_[task_index].insert(in_index);
  return *this;
}
TaskNodeMapFaker &TaskNodeMapFaker::NodeOutToIdentity(size_t task_index, int32_t out_index) {
  task_indexes_to_out_identity_indexes_[task_index].insert(out_index);
  return *this;
}
std::map<size_t, std::vector<NodePtr>> TaskNodeMapFaker::CreateInIdentityNodes(const ComputeGraphPtr &graph) const {
  auto data_node = graph->FindNode("data0");
  std::map<size_t, std::vector<NodePtr>> task_indexes_to_identity_nodes;
  for (const auto &task_index_and_in_indexes : task_indexes_to_in_identity_indexes_) {
    auto task_index = task_index_and_in_indexes.first;
    for (const auto i : task_index_and_in_indexes.second) {
      auto op_desc =
          std::make_shared<OpDesc>("Identity/" + std::to_string(task_index) + "/in/" + std::to_string(i), "Identity");
      op_desc->AddInputDesc(GeTensorDesc());
      op_desc->AddOutputDesc(GeTensorDesc());
      auto node = graph->AddNode(op_desc);
      GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
      task_indexes_to_identity_nodes[task_index].emplace_back(std::move(node));
    }
  }
  return task_indexes_to_identity_nodes;
}
TaskNodeMapFaker::TaskNodeMapFaker(const vector<TaskRunParam> &params) : task_indexes_to_param_(params) {}
TaskNodeMapFaker &TaskNodeMapFaker::NodeType(size_t task_index, std::string node_type) {
  task_indexes_to_node_type_[task_index] = std::move(node_type);
  return *this;
}
}  // namespace ge
