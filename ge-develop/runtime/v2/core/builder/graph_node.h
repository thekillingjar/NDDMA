/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_NODE_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_NODE_H_
#include <vector>
#include <map>
#include "graph/fast_graph/fast_node.h"
#include "core/execution_data.h"
#include "common/debug/ge_log.h"
#include "graph_async_value.h"
#include "executor_builder.h"
#include "node_types.h"

namespace gert {
// 执行图上节点并不都会创建执行节点，按照职能划分为以下7类（其中1-5类在最终执行时无执行节点也不执行，6类有执行节点，需要初始化输出，但不需要执行）：
// 1、表达图输入的节点：主要包括Data、ConstData节点
// 2、表达图输出的节点：主要包括NetOutput节点
// 3、表达用户传入的输出内存的节点：主要包括OutputData节点
// 4、表达内存传递关系的节点：主要包括InnerData、InnerNetOutput节点
// 5、表达复杂激活关系的节点：主要包括含子图的节点，如If、Case、While、SubGraphCall等
// 6、用于存储常量数据的节点：主要包括Const节点
// 7、需要执行的执行图节点：除1-6类之外的其他节点

// 类型1：表达图输入的节点
inline bool IsGraphInputNode(const char *const node_type) {
  return IsFeedType(node_type) || IsConstFeedType(node_type);
}
// 类型2：表达图输出的节点
inline bool IsGraphOutputNode(const char *const node_type) {
  return IsOutputType(node_type);
}
// 类型3：表达用户传入的输出内存的节点
inline bool IsUsrOutputNode(const char *const node_type) {
  return IsTypeOutputData(node_type);
}
// 类型4：表达内存传递关系的节点
inline bool IsMemTransferNode(const char *const node_type) {
  return IsInnerDataType(node_type) || IsInnerOutput(node_type);
}
// 类型5：表达复杂激活关系的节点
inline bool IsHasSubGraphNode(const char *const node_type) {
  return IsIfOrCaseType(node_type) || IsWhileType(node_type) || IsSubgraphCall(node_type);
}
// 类型6：用于存储常量数据的节点
inline bool IsStroreConstDataNode(const char *const node_type) {
  return IsConstType(node_type);
}

inline bool IsNodeNeedExec(const char *const node_type) {
  return !(IsGraphInputNode(node_type) || IsGraphOutputNode(node_type) || IsUsrOutputNode(node_type) ||
           IsMemTransferNode(node_type) || IsHasSubGraphNode(node_type) || IsStroreConstDataNode(node_type));
}

struct GraphNode {
  std::vector<Node *> nodes; // 所有执行图上节点（不带连边关系）
  std::unordered_map<const ge::FastNode *, Node *> graph_nodes_to_executor_node; // 执行图节点到执行节点的映射
  std::vector<int64_t> node_indegrees;
  std::map<int32_t, AsyncAnyValue *> indexes_to_feed_input; // index到输入tensor的索引
  std::vector<AsyncAnyValue *> indexes_to_output; // 输出tensor
  std::vector<Node *> start_nodes;
  std::vector<ge::FastNode *> feed_nodes; // 执行图上feed node
  std::vector<Watcher *> node_watchers;
  std::unordered_map<const ge::FastNode *, std::vector<const ge::FastNode *>> additional_add_info;
  std::unordered_map<const ge::FastNode *, std::vector<const ge::FastNode *>> additional_del_info;
  std::unordered_map<const ge::FastNode *, int64_t> additional_indegree_info;

  ge::graphStatus ReadInTopoInfo(const std::pair<ge::FastNode *, Node *> &graph_node_to_exe_node, Watcher *&watcher);
  ge::graphStatus EnsureNodeExeInOrder(ge::ExecuteGraph *exe_graph);
  ge::graphStatus ReadInNodeHasSubgraph(const ge::FastNode *const node);

  ge::graphStatus ReadInFeed(ge::FastNode *const node, GraphAsyncValue &graph_async_value) {
    auto &output_values = graph_async_value.nodes_to_output_values[node];
    int32_t feed_index;
    if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", feed_index)) {
      GELOGE(ge::FAILED, "Failed to get index from data node %s", node->GetName().c_str());
      return ge::GRAPH_FAILED;
    }
    if (indexes_to_feed_input.count(feed_index) > 0U) {
      GELOGE(ge::PARAM_INVALID, "Duplicated Data index found, index %d, name %s", feed_index, node->GetName().c_str());
      return ge::PARAM_INVALID;
    }
    if (output_values.empty()) {
      GELOGE(ge::FAILED, "Can not find the output any value for node %s", node->GetName().c_str());
      return ge::GRAPH_FAILED;
    }
    indexes_to_feed_input[feed_index] = output_values[0];
    if (IsFeedType(node->GetTypePtr())) {
      feed_nodes.emplace_back(node);
    }
    GELOGD("Loaded feed input index %d, node name %s", feed_index, node->GetNamePtr());
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus ReadInNetOutput(ge::FastNode *const node, GraphAsyncValue &graph_async_value) {
    const auto &input_values = graph_async_value.nodes_to_input_values[node];
    const auto data_in_size = node->GetDataInNum();
    if (data_in_size > input_values.size()) {
      GELOGE(ge::PARAM_INVALID,
             "Failed to load NetOutput node, there are %u inputs on the exe-graph, but only %zu chains created",
             data_in_size, input_values.size());
      return ge::PARAM_INVALID;
    }
    indexes_to_output.insert(indexes_to_output.cend(), input_values.cbegin(), input_values.cend());
    return ge::GRAPH_SUCCESS;
  }

  Node *ReadInNode(ge::FastNode *const node, GraphAsyncValue &graph_async_value) {
    auto &input_values = graph_async_value.nodes_to_input_values[node];
    auto &output_values = graph_async_value.nodes_to_output_values[node];
    auto exe_node =
        CreateNode(nodes.size(), input_values.size(), input_values.data(), output_values.size(), output_values.data());
    if (exe_node == nullptr) {
      return nullptr;
    }

    nodes.emplace_back(exe_node);
    graph_nodes_to_executor_node[node] = exe_node;
    GELOGD("Execute kernel id %ld, name %s", exe_node->node_id, node->GetNamePtr());
    return exe_node;
  }

 private:
  int64_t CalcIndegree(const ge::FastNode *node) const {
    int64_t indegree = 0;
    const auto incre_indegree = [&indegree](const ge::FastEdge *const edge) -> void {
      if ((edge != nullptr) && IsNodeNeedExec(edge->src->GetTypePtr())) {
        ++indegree;
      }
    };
    std::for_each(node->GetAllInDataEdgesRef().begin(), node->GetAllInDataEdgesRef().end(), incre_indegree);
    std::for_each(node->GetAllInControlEdgesRef().begin(), node->GetAllInControlEdgesRef().end(), incre_indegree);
    const auto it = additional_indegree_info.find(node);
    if (it != additional_indegree_info.end()) {
      indegree += it->second;
    }

    // todo 新增loader机制解决此类定制问题
    if (IsWaitAnyone(node->GetTypePtr())) {
      return std::min(indegree, static_cast<int64_t>(1L)); // tiny下编译需要强转
    }
    return indegree;
  }
  bool IsStartNode(const ge::FastNode *node) const {
    // 执行节点中只有第6类节点不需要执行
    if (IsStroreConstDataNode(node->GetTypePtr())) {
      return false;
    }
    return CalcIndegree(node) == 0;
  }

  ge::graphStatus GetExeNodeId(const ge::FastNode *node, NodeIdentity &node_id) {
    const auto iter = graph_nodes_to_executor_node.find(node);
    GE_ASSERT_TRUE(iter != graph_nodes_to_executor_node.end(),
                   "Can not find the executor node from graph node %s(%s) when create watcher",
                   node->GetName().c_str(), node->GetType().c_str());
    node_id = iter->second->node_id;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus UpdateWatcherInfo(const ge::FastNode *node, std::vector<NodeIdentity> &watch_nodes) {
    NodeIdentity node_id = 0UL;
    auto add_info_it = additional_add_info.find(node);
    if (add_info_it != additional_add_info.end()) {
      for (auto watch_node : add_info_it->second) {
        GE_ASSERT_GRAPH_SUCCESS(GetExeNodeId(watch_node, node_id));
        watch_nodes.push_back(node_id);
      }
    }

    auto del_info_it = additional_del_info.find(node);
    if (del_info_it != additional_del_info.end()) {
      for(auto watch_node : del_info_it->second) {
        GE_ASSERT_GRAPH_SUCCESS(GetExeNodeId(watch_node, node_id));
        const auto it = std::find(watch_nodes.begin(), watch_nodes.end(), node_id);
        if (it != watch_nodes.end()) {
          watch_nodes.erase(it);
        }
      }
    }
    return ge::GRAPH_SUCCESS;
  }

  void ReadInIndegree(const std::pair<ge::FastNode *, Node *> &node_to_exe_node) {
    auto indegree = CalcIndegree(node_to_exe_node.first);
    node_indegrees[node_to_exe_node.second->node_id] = indegree;
  }

  void ReadInStartNode(const std::pair<ge::FastNode *, Node *> &node_to_exe_node) {
    if (IsStartNode(node_to_exe_node.first)) {
      start_nodes.emplace_back(node_to_exe_node.second);
    }
  }

  ge::graphStatus ReadInWatcher(const std::pair<ge::FastNode *, Node *> &node_to_exe_node, Watcher *&watcher);
  ge::graphStatus ReadInIfOrCase(const ge::FastNode *const node);
  ge::graphStatus ReadInWhile(const ge::FastNode *const node);
  ge::graphStatus ReadInSubgraphCall(const ge::FastNode *const node);
  ge::graphStatus GuardGraphByPivotAndDone(ge::ExecuteGraph *const graph, ge::FastNode *pivot, ge::FastNode *done);
  ge::graphStatus SetAdditionalInfo(const ge::FastNode *src_node, const ge::FastNode *dst_node, bool isAdd);
  ge::graphStatus AddAdditionalInfo(const ge::FastNode *src_node, const ge::FastNode *dst_node);
  ge::graphStatus RemoveAdditionalInfo(const ge::FastNode *src_node, const ge::FastNode *dst_node);
  ge::graphStatus EnsureNodeExeInOrderInSubgraph(const ge::ExecuteGraph *sub_exe_graph);
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_NODE_H_
