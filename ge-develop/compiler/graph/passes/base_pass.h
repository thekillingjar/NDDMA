/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_BASE_PASS_H_
#define GE_GRAPH_PASSES_BASE_PASS_H_

#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "framework/common/ge_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/debug/log.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "register/optimization_option_registry.h"
#include "base/err_msg.h"

namespace ge {
enum NodePassOption {
  // if there is a sub graph on the node, the pass on the node will do:
  // Pass(node) -> pass all sub graphs on the node -> Pass(node)
  // when pass the node for the second time, the kOptimizeAfterSubGraph will be set as a flag key
  kOptimizeAfterSubGraph,

  // add new options before kOptionEnd
  kOptionEnd
};

class BaseNodePass {
  // todo comments
 public:
  struct Perf {
    uint64_t time_cost_{0U};
    uint64_t call_num_{0U};
  };
  /// Optimize on one node. the function can add nodes to the graph, change
  /// connections between nodes while optimizing or remove nodes from the graph.
  /// @param node
  /// @return
  virtual Status Run(NodePtr &node) = 0;

  virtual ~BaseNodePass() = default;

  const std::vector<NodePtr> &GetNodesNeedRePass() { return nodes_need_re_pass_; }

  const OrderedNodeSet &GetNodesNeedRePassImmediately() { return nodes_need_re_pass_immediately_; }

  const OrderedNodeSet &GetGlobalNodesNeedRePassImmediately() {
    return global_nodes_need_repass_immediately_;
  }

  const std::unordered_set<NodePtr> &GetNodesDeleted() { return nodes_deleted_; }

  const std::unordered_set<NodePtr> &GetNodesSuspend() { return nodes_suspend_; }

  const OrderedNodeSet &GetNodesResume() { return nodes_resume_; }

  virtual Status OnSuspendNodesLeaked() { return SUCCESS; }

  virtual Status OnFinishGraph(ComputeGraphPtr &root_graph, std::vector<NodePtr> &node_to_be_repass) { return SUCCESS; }

  void SetOption(NodePassOption option, const std::string &value) { options_[option] = value; }

  void ClearOptions() { options_.clear(); }

  void Init() {
    nodes_need_re_pass_.clear();
    nodes_need_re_pass_immediately_.clear();
    nodes_deleted_.clear();
    nodes_suspend_.clear();
    nodes_resume_.clear();
    global_nodes_need_repass_immediately_.clear();
  }

  virtual void OnStartPassGraph(const ComputeGraphPtr &graph) {
    current_graph_name_ = graph->GetName();
  }
  Perf &MutablePerf() {
    return perf_;
  }

 protected:
  const std::string &GetCurrentGraphName() const {
    return current_graph_name_;
  }
  Status IsolateAndDeleteNode(NodePtr &node, const std::vector<int32_t> &io_map, bool is_repass_io_immediately = false);

  Status IsolateAndDeleteNode(NodePtr &node, const std::initializer_list<int32_t> &io_map,
                              bool is_repass_io_immediately = false) {
    return IsolateAndDeleteNode(node, std::vector<int32_t>(io_map), is_repass_io_immediately);
  }

  Status DeleteUselessConstAxisNode(NodePtr &axis_node);

  /// Add a node to be optimized again. If you add a new node to the graph, or
  /// change a node connections, and you want to make sure the node will be
  /// optimized by other passes, call this function.
  /// @param node
  void AddRePassNode(const NodePtr &node) { nodes_need_re_pass_.emplace_back(node); }

  /// Add a node to be optimized immediately again. If you add a new node to the graph, or
  /// change a node connections, and you want to make sure the node will be
  /// optimized by other passes, call this function.
  /// @param node
  void AddImmediateRePassNode(const NodePtr &node) { nodes_need_re_pass_immediately_.insert(node); }

  /// Add a node to be optimized again which belongs to other sub_graph. If you add a new node to the graph, or
  /// change a node connections, and you want to make sure the node will be
  /// optimized by other passes, call this function.
  /// @param node
  void AddGlobalImmediateRePassNode(const NodePtr &node) {
    global_nodes_need_repass_immediately_.insert(node);
  }

  /// Add a node and it's input/output data nodes to be optimized again.
  /// @param node
  void AddRePassNodesWithInOut(const NodePtr &node) {
    auto in_nodes = node->GetInNodes();
    for (auto &in_node : in_nodes) {
      AddRePassNode(in_node);
    }
    AddRePassNode(node);
    auto out_nodes = node->GetOutNodes();
    for (auto &out_node : out_nodes) {
      AddRePassNode(out_node);
    }
  }

  /// Add a node and it's input/output data nodes to be optimized immediately again.
  /// @param node
  void AddImmediateRePassNodesWithInOut(const NodePtr &node) {
    auto in_nodes = node->GetInNodes();
    for (auto &in_node : in_nodes) {
      AddImmediateRePassNode(in_node);
    }
    AddImmediateRePassNode(node);
    auto out_nodes = node->GetOutNodes();
    for (auto &out_node : out_nodes) {
      AddImmediateRePassNode(out_node);
    }
  }

  /// If you deleted a node from the graph, especially current node. The remain
  /// iterate passes will continue process on the deleted node(if it can be
  /// reached by edge connections) till the last one. Obviously it is a waste of
  /// time. You can add the deleted nodes by calling this function, to stop the
  /// next iterations.
  /// @param node
  void AddNodeDeleted(const NodePtr &node) { nodes_deleted_.insert(node); }

  /// If you postpone a node from the graph, especially following node. The remain
  /// iterate passes will stop process on the postpone node(if it can be
  /// reached by edge connections) till the last one. Obviously it is a waste of
  /// time. You can add the postpone nodes by calling this function, to stop the
  /// next iterations.
  /// @param node
  void AddNodeSuspend(const NodePtr &node) { nodes_suspend_.insert(node); }

  void AddNodeResume(const NodePtr &node) { nodes_resume_.insert(node); }

  bool OptionExists(NodePassOption option) { return options_.count(option) > 0; }

 private:
  std::vector<NodePtr> nodes_need_re_pass_;
  OrderedNodeSet nodes_need_re_pass_immediately_;
  std::unordered_set<NodePtr> nodes_deleted_;
  std::unordered_set<NodePtr> nodes_suspend_;
  OrderedNodeSet nodes_resume_;
  OrderedNodeSet global_nodes_need_repass_immediately_;
  std::map<NodePassOption, std::string> options_;
  std::string current_graph_name_;
  Perf perf_;
};

using NamesToPass = std::vector<std::pair<std::string, BaseNodePass *>>;

class GEPass {
 public:
  explicit GEPass(const ComputeGraphPtr &graph) : depth_(1), graph_(graph), root_graph_(graph) {
    GE_MAKE_SHARED(repass_nodes_on_root_graph_ = std::make_shared<RepassLevelState>(),
                   repass_nodes_on_root_graph_ = nullptr);
  }

  Status Run(const NamesToPass &names_to_passes, bool with_filter = true);

  Status AddPassAfterGraphOptimized(const NamesToPass &names_to_passes);
  /*
  * todo
  * OneGraph: nodes_deleted, nodes_seen, nodes_passed, nodes_suspended
  * RePass: nodes_re_pass
  * GraphOneTime: nodes_last
  * NodeOneTime: nodes_re_pass_immediately, nodes_resume
  */
  struct GraphLevelState {
    std::unordered_set<NodePtr> nodes_deleted;
    std::unordered_set<Node *> nodes_seen;
    std::unordered_set<NodePtr> nodes_passed;
    std::unordered_set<Node *> nodes_suspend;
    OrderedNodeSet nodes_last;
    std::deque<NodePtr> nodes;
    uint64_t passed_node_size = 0;
    uint64_t max_pass_node_size = 0;

    void AddNodeToQueueFront(NodePtr node) {
      nodes_seen.insert(node.get());
      nodes.emplace_front(std::move(node));
    }

    void AddNodeToQueue(NodePtr node) {
      nodes_seen.insert(node.get());
      nodes.emplace_back(std::move(node));
    }
    void AddNodeToQueueIfNotSeen(NodePtr node) {
      if (nodes_seen.insert(node.get()).second) {
        nodes.emplace_back(std::move(node));
      }
    }
    NodePtr PopFront() {
      NodePtr node = nodes.front();
      nodes.pop_front();
      return node;
    }
  };
  struct RepassLevelState {
    std::vector<NodePtr> nodes_re_pass;
    std::unordered_set<NodePtr> nodes_re_pass_set;
    bool AddNodeToRepass(NodePtr node) {
      if (!nodes_re_pass_set.insert(node).second) {
        return false;
      }
      nodes_re_pass.emplace_back(node);
      return true;
    }
    void EraseNodeFromRepass(NodePtr node) {
      nodes_re_pass_set.erase(node);
    }
    void ClearRepass() {
      nodes_re_pass_set.clear();
      nodes_re_pass.clear();
    }
  };
  struct GraphOneTimeLevelState {
    std::unordered_set<NodePtr> nodes_last;
  };
  struct RootGraphLevelState {
    ComputeGraphPtr root_graph;
    RepassLevelState root_graph_immediate_repass_state;
  };

 private:
  using RepassNodesPtr = std::shared_ptr<RepassLevelState>;
  GEPass(ComputeGraphPtr &graph, ComputeGraphPtr &root_graph, RepassNodesPtr &repass_on_root_graph, int32_t depth)
      : depth_(depth), graph_(graph), root_graph_(root_graph), repass_nodes_on_root_graph_(repass_on_root_graph) {}

  Status RunPassesNodeOnce(NodePtr &node, const NamesToPass &names_to_passes,
                           GraphLevelState &graph_state, RepassLevelState &rp_state);
  Status RunPassesGraphRepass(const NamesToPass &names_to_passes, GraphLevelState &graph_state);
  Status RunPassesOneGraph(const NamesToPass &names_to_passes);
  Status RunPassesOnSubGraph(const NodePtr &node, const NamesToPass &names_to_passes, bool &has_sub_graph);
  Status RunPassesOnNode(NodePtr &node, const NamesToPass &names_to_passes, GraphLevelState &graph_state,
                         RepassLevelState &rp_state);
  Status HandleLeakedSuspendNodes(const NamesToPass &names_to_passes, GraphLevelState &graph_state) const;
  Status RunPassesAfterFinishGraph(GraphLevelState &graph_state);
  void AddGlobalImmediateRepassNodeToQueueIfSeen(GraphLevelState &graph_state) const;
  bool IsCurrentPassRootGraph() const {
      return graph_ == root_graph_;
  }
  static NamesToPass FilterDisabledOptimizations(const NamesToPass &names_to_passes);

  int32_t depth_;
  ComputeGraphPtr graph_;
  ComputeGraphPtr root_graph_;
  RepassNodesPtr repass_nodes_on_root_graph_; // only root GE PASS owns it
  NamesToPass pass_after_graph_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_BASE_PASS_H_
