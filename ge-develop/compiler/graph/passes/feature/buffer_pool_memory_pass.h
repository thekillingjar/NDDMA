/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_BUFFER_POOL_MEMORY_PASS_H_
#define GE_GRAPH_PASSES_BUFFER_POOL_MEMORY_PASS_H_

#include <queue>
#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
class BufferPoolMemoryPass : public GraphPass {
 public:
  explicit BufferPoolMemoryPass() : logic_event_num_(0) {}

  ~BufferPoolMemoryPass() override = default;

  struct BufferPool {
    int64_t pool_id = 0;
    int64_t pool_size = 0;
    std::unordered_map<NodePtr, NodePtr> buffer_node_to_calc;
    BufferPool(int64_t id, int64_t size, const std::unordered_map<NodePtr, NodePtr> &node_map)
        : pool_id(id), pool_size(size), buffer_node_to_calc(node_map) {}
  };

  struct BufferPoolNodeItem {
    NodePtr node = nullptr;
    NodePtr out_calc_node = nullptr;
    NodePtr pre_buffer_pool_node = nullptr;
    int64_t total_size = 0;
    int64_t offset_start = 0;
    int64_t offset_end = 0;
    bool is_last_input = true;
    bool is_first_in_group = true;
    BufferPoolNodeItem(const NodePtr &buffer_n, const NodePtr &calc_n, const NodePtr &pre_buffer_n,
                       int64_t size, int64_t start, int64_t end, bool last)
        : node(std::move(buffer_n)),
          out_calc_node(std::move(calc_n)),
          pre_buffer_pool_node(std::move(pre_buffer_n)),
          total_size(size),
          offset_start(start),
          offset_end(end),
          is_last_input(last) {}

    BufferPoolNodeItem(const NodePtr &buffer_n, int64_t start, int64_t end)
        : node(std::move(buffer_n)),
          out_calc_node(nullptr),
          pre_buffer_pool_node(nullptr),
          total_size(0),
          offset_start(start),
          offset_end(end),
          is_last_input(true) {}
  };

  struct BufferPoolNodeItemGroup {
    std::vector<BufferPoolNodeItem> node_items;
  };

  Status Run(ComputeGraphPtr graph) override;

 private:
  static void ClearQueue(std::queue<std::pair<std::string, uint32_t>> &q);

  static bool IsBufferPoolMemEnable(const ComputeGraphPtr &graph);

  static Status CheckBufferPoolSize(int64_t total_size, int64_t pool_id, int64_t buffer_pool_size,
                                    std::unordered_map<int64_t, int64_t> &calc_total_size);

  static Status TryToFixNodeOrder(NodePtr &pre_node, NodePtr &curr_node, bool &not_change);

  Status InsertMemCpyNodeAfter(NodePtr &node) const;

  Status CopyOutForMultiUsedOutput(ComputeGraphPtr &graph) const;

  Status GetBufferPoolAndPeerCalcNodes(const ComputeGraphPtr &graph);

  Status SetBufferPoolSize(const std::string &batch_label, int64_t id, int64_t size);

  Status AllocateAllBufferPoolSpace();

  Status AllocateSpaceInBatch(const std::map<int64_t, std::vector<NodePtr>> &calc_nodes,
                              const std::unordered_map<int64_t, int64_t> &buffer_pool_size_map,
                              const std::unordered_map<NodePtr, NodePtr> &buffer_node_to_calc);

  Status AllocateSpaceInBufferPool(const BufferPool &buffer_pool,
                                   const std::vector<NodePtr> &calc_nodes_in_pool);

  Status AllocateSpaceForBufferPoolNode(int64_t &next_start,
                                        const BufferPool &buffer_pool,
                                        BufferPoolNodeItem &buffer_pool_node_item,
                                        const int64_t group_total_mem_size,
                                        std::queue<BufferPoolNodeItem> &node_mem_range_in_pool);

  NodePtr GetOffsetAndDependency(int64_t &next_start,
                                 int64_t total_mem_size,
                                 int64_t buffer_pool_size,
                                 const std::unordered_map<NodePtr, NodePtr> &buffer_node_to_calc,
                                 std::queue<BufferPoolNodeItem> &nodes_in_buffer) const;

  Status FixTheTimingOfDependentNodes(NodePtr &dependent_calc_node, NodePtr &curr_pool_node);

  uint32_t GenerateEventId(const std::string &node_name, std::queue<std::pair<std::string, uint32_t>> &event_queue);

  Status SetResultOfMemoryAndEvent();

  Status DisableBufferPreventingCycle(const ComputeGraphPtr &compute_graph);

  static Status DoCheckAndDisable(
      const std::vector<NodePtr> &calc_nodes,
      std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>> &calc_node_to_buffer_node_items);

  static bool HasDependency(const NodePtr &from_node, const NodePtr &to_node);
  static bool FindOrBackward(const NodePtr &target_node,
                             std::vector<NodePtr> &nodes,
                             std::unordered_set<const Node *> &visited);
  void GroupBufferPoolNodes();
  void DoGroupBufferPoolNodes(std::vector<NodePtr> &calc_nodes,
                              std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>> &buffer_node_items);
  static NodePtr GetLastOutDataNode(const NodePtr &node);
  static NodePtr BypassRefIoNodes(const NodePtr &node);

  // Use map to ensure that each visit is in the order of batch label and pool id
  std::map<std::string, std::map<int64_t, std::vector<NodePtr>>> calc_nodes_;

  std::unordered_map<std::string, std::unordered_map<NodePtr, NodePtr>> buffer_node_to_calc_;

  std::unordered_map<std::string, std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>>> peer_buffer_node_item_;
  std::unordered_map<NodePtr, std::vector<BufferPoolNodeItemGroup>> peer_buffer_node_items_;

  std::unordered_map<std::string, std::unordered_map<int64_t, int64_t>> buffer_pool_size_;

  uint32_t logic_event_num_;

  std::queue<std::pair<std::string, uint32_t>> mem_ctrl_event_;

  std::queue<std::pair<std::string, uint32_t>> stream_ctrl_event_;

  std::unordered_map<NodePtr, std::vector<std::string>> node_event_multiplexing_;

  std::unordered_map<NodePtr, std::vector<int64_t>> buffer_node_logical_offset_;

  std::unordered_set<NodePtr> buffer_pool_nodes_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_BUFFER_POOL_MEMORY_PASS_H_
