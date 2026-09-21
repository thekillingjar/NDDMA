/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_STREAM_INFO_H
#define CANN_GRAPH_ENGINE_STREAM_INFO_H
#include <string>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include "graph/graph.h"
#include "graph_metadef/graph/node.h"

namespace ge {
struct StreamNode {
 public:
  explicit StreamNode(const NodePtr node);
  size_t GetTopoId() const;
  std::string GetStreamLabel() const;
  std::vector<uint32_t> GetActivatedStreams() const;
  int64_t GetStreamId() const;
  std::string GetName() const;
  std::string GetType() const;
  const std::vector<size_t> &GetSendEventIds() const;
  const std::vector<size_t> &GetRecvEventIds() const;
  const std::vector<uint32_t> &GetLabelIndexes() const;
  size_t GetIdInStream() const;
  bool IsInBranch() const;
  void SetIdInStream(size_t id);
  void AppendSendEventId(size_t event_id);
  void SetRecvEventIds(const std::vector<size_t> &recv_event_ids);

 private:
  NodePtr origin_node_;
  size_t idx_{};
  bool in_branch_{false};
  std::vector<size_t> send_event_ids_;
  std::vector<size_t> recv_event_ids_;
  std::vector<uint32_t> label_indexes_;
};
using StreamNodePtr = std::shared_ptr<StreamNode>;

class StreamGraph {
 public:
  explicit StreamGraph(const ComputeGraphPtr &graph);
  ~StreamGraph() = default;
  std::string ToDotString() const;

 private:
  void Build();
  std::string DotHeaderStr() const;
  std::string DotStreams() const;
  std::string DotEdges() const;
  std::string DotActives() const;
  std::string DotTailStr() const;
  std::string DotId(const StreamNodePtr node) const;

 private:
  std::string name_;
  std::vector<StreamNodePtr> nodes_;
  size_t stream_num_{0UL};
  std::vector<std::vector<StreamNodePtr>> stream_nodes_;
  std::map<size_t, StreamNodePtr> send_event_node_map_;
  std::map<size_t, StreamNodePtr> recv_event_node_map_;
  std::vector<std::vector<StreamNodePtr>> stream_nodes_without_events_;
  std::vector<std::string> stream_labels_;
  std::set<size_t> stream_in_branch_;
};
using StreamGraphPtr = std::shared_ptr<StreamGraph>;
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_STREAM_INFO_H
