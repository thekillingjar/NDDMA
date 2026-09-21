/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/common_subexpression_elimination_pass.h"

#include <sstream>
#include <string>
#include <set>
#include <queue>

#include "common/checker.h"
#include "common/base64.h"
#include "graph/utils/node_utils.h"
#include "graph/passes/standard_optimize/constant_folding/folding_pass.h"
#include "register/op_kernel_registry.h"
#include "common/op/transop_util.h"

namespace ge {
namespace {
struct CandidateNodes {
  std::queue<NodePtr> node_queue;
  std::unordered_set<NodePtr> candidate_nodes_set;
  void InsertCandidate(const NodePtr &node) {
    if (candidate_nodes_set.insert(node).second) {
      GELOGD("Node %s is candidate of CSE.", node->GetNamePtr());
      node_queue.emplace(node);
    }
  }
};
std::string GetCseKey(const NodePtr &node) {
  std::stringstream ss;
  ss << node->GetType() << "-data-inputs-";
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    auto src_anchor = in_anchor->GetPeerOutAnchor();
    if (src_anchor == nullptr) {
      ss << in_anchor->GetIdx() << "-null-";
    } else {
      const auto &input_desc = node->GetOpDesc()->GetInputDescPtr(in_anchor->GetIdx());
      GE_ASSERT_NOTNULL(input_desc);
      ss << in_anchor->GetIdx() << "-"
         << src_anchor->GetOwnerNode()->GetName() << "-"
         << src_anchor->GetIdx() << "-"
         << input_desc->GetShape().ToString() << "-"
         << input_desc->GetOriginShape().ToString() << "-"
         << input_desc->GetFormat() << "-"
         << input_desc->GetOriginFormat() << "-";
    }
  }

  ss << "control-inputs-";
  std::set<std::string> control_in_node_names;
  for (auto &src_node : node->GetInControlNodes()) {
    control_in_node_names.insert(src_node->GetName());
  }
  for (auto &name : control_in_node_names) {
    ss << name << "-";
  }

  ss << "attrs-" << AttrUtils::GetAllAttrsStr(node->GetOpDesc());

  return ss.str();
}

Status CollectPeerCandidateNodesWithType(
    const OutDataAnchor *const out_data_anchor, const std::unordered_map<NodePtr, size_t> &nodes_2_topo_idx,
    std::map<std::string, std::map<size_t, NodePtr>> &node_types_to_ordered_nodes) {
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    const auto &peer_in_node = peer_in_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    // this attr mean node is inserted by ge， can not be optimized
    // 这个属性后续需要逐渐演变为标记由ge的插入算子，后续的优化跳过ge插入的算子。
    if (AttrUtils::HasAttr(peer_in_node->GetOpDesc(), ATTR_NO_NEED_CONSTANT_FOLDING)) {
      continue;
    }

    /// As the operator category has not been defined, we do not know what types of node can be processed by CSE.
    /// To avoid delete wrong nodes(e.g. stateful nodes),
    /// only nodes have folding kernel will be considered for the CSE process
    const auto &node_type = peer_in_node->GetType();
    if (!OpKernelRegistry::GetInstance().IsRegistered(node_type) &&
        (folding_pass::GetKernelByType(peer_in_node) == nullptr)) {
      continue;
    }

    const auto iter = nodes_2_topo_idx.find(peer_in_node);
    GE_ASSERT_TRUE(iter != nodes_2_topo_idx.cend());
    node_types_to_ordered_nodes[node_type].emplace(iter->second, peer_in_node);
  }
  return SUCCESS;
}

Status CollectCandidate(const NodePtr &node, const std::unordered_map<NodePtr, size_t> &nodes_2_topo_idx,
                        CandidateNodes &candidate_nodes) {
  // output has same type sibling
  std::map<std::string, std::map<size_t, NodePtr>> node_type_2_nodes;
  for (const auto out_data_anchor : node->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(out_data_anchor);
    if (out_data_anchor->GetPeerInDataNodesSize() < 2U) {
      continue;
    }
    GE_ASSERT_SUCCESS(CollectPeerCandidateNodesWithType(out_data_anchor, nodes_2_topo_idx, node_type_2_nodes));

    for (const auto &type_2_nodes : node_type_2_nodes) {
      if (type_2_nodes.second.size() < 2U) {
        continue;
      }
      // sort nodes by topo idx
      GELOGD("Node %s output %d has %zu sibling of type %s.", node->GetNamePtr(), out_data_anchor->GetIdx(),
             type_2_nodes.second.size(), type_2_nodes.first.c_str());
      for (const auto &candidate : type_2_nodes.second) {
        candidate_nodes.InsertCandidate(candidate.second);
      }
    }
    node_type_2_nodes.clear();
  }
  return GRAPH_SUCCESS;
}
}  // namespace

Status CommonSubexpressionEliminationPass::Run(ComputeGraphPtr graph) {
  GELOGD("Begin to run the CSE process on the graph");
  GE_CHECK_NOTNULL(graph);
  // here mark nodes to its topo id, to make sure CSE optimize follow origin node seq
  std::unordered_map<NodePtr , size_t> nodes_2_topo_idx;
  size_t index = 0U;
  for (const auto &node : graph->GetDirectNode()) {
    nodes_2_topo_idx.emplace(node, index++);
  }

  // collect node may match CSE
  CandidateNodes candidate_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_SUCCESS(CollectCandidate(node, nodes_2_topo_idx, candidate_nodes));
  }

  std::unordered_map<std::string, NodePtr> keys_to_node;
  while (!candidate_nodes.node_queue.empty()) {
    const auto node = candidate_nodes.node_queue.front();
    candidate_nodes.node_queue.pop();
    auto key = GetCseKey(node);
    GELOGD("The node %s cse key %s", node->GetName().c_str(), ge::base64::EncodeToBase64(key).c_str());
    std::unordered_map<std::string, NodePtr>::const_iterator iter = keys_to_node.find(key);
    if (iter == keys_to_node.cend()) {
      keys_to_node[key] = node;
      continue;
    }

    if (node->GetAllOutDataAnchorsSize() != iter->second->GetAllOutDataAnchorsSize()) {
      GELOGW("The node %s and %s have the same CSE key, but different output anchor count, skip to fusion them",
             iter->second->GetName().c_str(), node->GetName().c_str());
      continue;
    }

    std::vector<int32_t> output_map(node->GetAllOutDataAnchorsSize());
    for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      output_map[i] = i;
    }

    auto ret = GraphUtils::ReplaceNodeAnchors(iter->second, node, {}, output_map);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Replace node:%s(%s)'s anchor by node:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str(),
                        iter->second->GetName().c_str(), iter->second->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Replace][Node] %s by node %s failed, ret:%u",
             node->GetName().c_str(), iter->second->GetName().c_str(), ret);
      return INTERNAL_ERROR;
    }
    NodeUtils::UnlinkAll(*node);
    ret = GraphUtils::RemoveNodeWithoutRelink(graph, node);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                        node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Remove][Node] %s from graph:%s failed", node->GetName().c_str(),
             graph->GetName().c_str());
      return INTERNAL_ERROR;
    }

    GELOGI("Remove node %s by the CSE process, replace it with node %s",
           node->GetName().c_str(), iter->second->GetName().c_str());
    GE_ASSERT_SUCCESS(CollectCandidate(iter->second, nodes_2_topo_idx, candidate_nodes));
  }
  return SUCCESS;
}

REG_PASS_OPTION("CommonSubexpressionEliminationPass").LEVELS(OoLevel::kO3);
}  // namespace ge
