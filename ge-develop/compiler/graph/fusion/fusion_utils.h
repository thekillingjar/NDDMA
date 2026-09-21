/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_FUSION_UTILS_H
#define CANN_GRAPH_ENGINE_FUSION_UTILS_H
#include <queue>
#include "common/checker.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils.h"
#include "ge/fusion/subgraph_boundary.h"
#include "ge/fusion/match_result.h"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace ge {
namespace fusion {
class FusionUtils {
 public:
  static Status MarkPassNameOnReplacementNodes(const std::unique_ptr<Graph> &replacement,
                                               const std::unique_ptr<SubgraphBoundary> &subgraph,
                                               const std::string &pass_name);
  static std::string ToString(const std::unique_ptr<Graph> &graph);

  static std::string GetFusionSwitchFileFromOption();

  static std::map<string, bool> ParseFusionSwitch();

  // 使用detector前后需要重置状态，确保当前detector掌握图的最新状态
  static void ResetCycleDetectors();

  static CycleDetectorSharedPtr GetOrCreateCycleDetector(const ComputeGraphPtr &curr_graph);

  static bool WillCauseCycleIfFuse(const std::unique_ptr<MatchResult> &match_result);

  static Status UpdateToCycleDetector(const ComputeGraphPtr &curr_graph,
                                      const std::unique_ptr<MatchResult> &match_result,
                                      const std::unique_ptr<Graph> &replacement);

  static void RecordFusionStatistic(const uint64_t session_id, const std::string graph_id, const std::string pass_name,
                                              const int match_times, const int effect_times);

 private:
  static std::unordered_map<ComputeGraphPtr, CycleDetectorSharedPtr> graph_2_cycle_detectors_;
};

struct InnerSubgraphBoundary {
  Status Init(const SubgraphBoundary &subgraph_boundary, std::string &failed_reason) {
    GE_WARN_ASSERT_GRAPH_SUCCESS(ToInnerBoundary(subgraph_boundary));
    GE_WARN_ASSERT_GRAPH_SUCCESS(CollectNodesInSubgraph(failed_reason));
    return SUCCESS;
  };
  /**
   * 子图输入tensor producer的个数，若一个tensor是单输出多引用的，算一个输入
   * producer本身不属于子图边界内
   */
  [[nodiscard]] size_t GetInputNum() const {
    return producer_out_anchor_.size();
  }

  /**
   * 子图边界input anchor的个数，若一个tensor producer是单输出两引用的，算两个边界输入
   * producer本身不属于子图边界内
   */
  [[nodiscard]] size_t GetBoundaryInputNum() const {
    size_t boundary_input_num = 0;
    for (const auto &producer_2_in_anchors : producer_2_inputs_in_anchor_) {
      boundary_input_num += producer_2_in_anchors.second.size();
    }
    return boundary_input_num;
  }
  /**
   * 子图输出tensor个数
   * @return
   */
  [[nodiscard]] size_t GetOutputNum() const {
    return outputs_out_anchor_.size();
  }

  [[nodiscard]] const std::vector<NodePtr> &GetNodes() const {
    return nodes_;
  }

  /**
   * 获取输入producer的out anchor
   * @param input_index input tensor的index
   * @return
   */
  [[nodiscard]] OutDataAnchorPtr GetInputProducerOutAnchor(size_t input_index) const {
    GE_ASSERT_TRUE(input_index < producer_out_anchor_.size());
    return producer_out_anchor_[input_index];
  }

  [[nodiscard]] std::vector<InDataAnchorPtr> GetBoundaryInAnchor(size_t input_index) const {
    GE_ASSERT_TRUE(input_index < producer_out_anchor_.size());
    auto producer_out_anchor = producer_out_anchor_[input_index];
    const auto iter = producer_2_inputs_in_anchor_.find(producer_out_anchor);
    GE_ASSERT_TRUE(iter != producer_2_inputs_in_anchor_.cend());
    return iter->second;
  }

  [[nodiscard]] OutDataAnchorPtr GetOutputOutAnchor(size_t output_index) const {
    GE_ASSERT_TRUE(output_index < outputs_out_anchor_.size());
    return outputs_out_anchor_[output_index];
  }

  [[nodiscard]] ComputeGraphPtr GetOwnerGraph() const {
    if (nodes_.empty()) {
      return nullptr;
    }
    return nodes_[0]->GetOwnerComputeGraph();
  }

 private:
  Status ToInnerBoundary(const SubgraphBoundary &boundary) {
    std::vector<SubgraphInput> subgraph_inputs;
    GE_WARN_ASSERT_GRAPH_SUCCESS(boundary.GetAllInputs(subgraph_inputs));
    std::set<NodePtr> subgraph_input_producer;
    for (const auto &subgraph_input : subgraph_inputs) {
      auto node_inputs = subgraph_input.GetAllInputs();
      GE_ASSERT_TRUE(!node_inputs.empty());

      OutDataAnchorPtr producer_out_anchor = nullptr;
      for (const auto &node_input : subgraph_input.GetAllInputs()) {
        auto in_data_anchor =
            NodeAdapter::GNode2Node(node_input.node)->GetInDataAnchor(static_cast<int32_t>(node_input.index));
        if (producer_out_anchor == nullptr) {
          producer_out_anchor = in_data_anchor->GetPeerOutAnchor();
          producer_out_anchor_.emplace_back(producer_out_anchor);
        }
        producer_2_inputs_in_anchor_[producer_out_anchor].emplace_back(in_data_anchor);
      }
    }

    std::vector<SubgraphOutput> subgraph_outputs;
    GE_WARN_ASSERT_GRAPH_SUCCESS(boundary.GetAllOutputs(subgraph_outputs));
    for (const auto &subgraph_output : subgraph_outputs) {
      NodeIo node_output;
      GE_WARN_ASSERT_GRAPH_SUCCESS(subgraph_output.GetOutput(node_output));
      auto out_node = NodeAdapter::GNode2Node(node_output.node);
      GE_ASSERT_NOTNULL(out_node);
      outputs_out_anchor_.emplace_back(out_node->GetOutDataAnchor(static_cast<int32_t>(node_output.index)));
    }
    return SUCCESS;
  }

  bool IsSubgraphBoundaryValid(const std::set<NodePtr> &nodes_set, const std::set<NodePtr> &output_nodes_set,
                               std::string &failed_reason) {
    ComputeGraphPtr owner_graph = nullptr;
    // check output data in same graph
    for (const auto &out_data_anchor : outputs_out_anchor_) {
      const auto output_node = out_data_anchor->GetOwnerNode();
      if (owner_graph == nullptr) {
        owner_graph = output_node->GetOwnerComputeGraph();
      } else {
        if (output_node->GetOwnerComputeGraph() != owner_graph) {
          GELOGE(FAILED, "Node [%s][%s] as output node of boundary, is not in same graph with others.",
                 output_node->GetNamePtr(), output_node->GetTypePtr());
          return false;
        }
      }
    }

    // check subgraph is self contained
    for (const auto &node : nodes_) {
      if (output_nodes_set.count(node) > 0) {
        continue;
      }
      for (const auto &out_node : node->GetOutDataNodes()) {
        if (nodes_set.count(out_node) == 0) {
          std::stringstream ss;
          ss << "Node [" << out_node->GetNamePtr() << "][" << out_node->GetTypePtr() << "] as output of Node ["
             << node->GetNamePtr() << "][" << node->GetTypePtr()
             << "] is not inside boundary, Boundary is not self contained.";
          failed_reason = ss.str();
          return false;
        }
      }
    }
    return true;
  }

  Status CollectNodesInSubgraph(std::string &failed_reason) {
    std::set<NodePtr> subgraph_input_producer;
    for (const auto &out_anchor : producer_out_anchor_) {
      subgraph_input_producer.emplace(out_anchor->GetOwnerNode());
    }

    std::set<NodePtr> output_nodes;
    std::queue<NodePtr> node_queue;
    for (const auto &out_anchor : outputs_out_anchor_) {
      node_queue.emplace(out_anchor->GetOwnerNode());
      output_nodes.emplace(out_anchor->GetOwnerNode());
    }

    std::set<NodePtr> nodes_set;
    while(!node_queue.empty()) {
      auto cur_node = node_queue.front();
      node_queue.pop();

      if (nodes_set.emplace(cur_node).second) {
        GELOGD("Node [%s][%s][topo_id: %ld] is in subgraph boundary.", cur_node->GetNamePtr(), cur_node->GetTypePtr(),
               cur_node->GetOpDesc()->GetId());
        nodes_.emplace_back(cur_node);
      }

      for (const auto &in_node : cur_node->GetInDataNodes()) {
        if (subgraph_input_producer.count(in_node) > 0) {
          continue;
        }
        node_queue.emplace(in_node);
      }
    }
    GE_WARN_ASSERT(IsSubgraphBoundaryValid(nodes_set, output_nodes, failed_reason));
    return SUCCESS;
  }
  std::vector<OutDataAnchorPtr> producer_out_anchor_;
  std::map<OutDataAnchorPtr, std::vector<InDataAnchorPtr>> producer_2_inputs_in_anchor_;
  std::vector<OutDataAnchorPtr> outputs_out_anchor_;
  std::vector<NodePtr> nodes_;
};
}  // namespace fusion
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_FUSION_UTILS_H
