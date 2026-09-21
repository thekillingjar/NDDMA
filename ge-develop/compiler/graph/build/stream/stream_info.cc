/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_info.h"
#include <sstream>
#include "external/ge/ge_api_types.h"
#include "graph_metadef/graph/debug/ge_attr_define.h"
#include "graph/compute_graph.h"
#include "common/types.h"

namespace ge {
namespace {
std::string VecIntToStr(const std::vector<uint32_t> &vec) {
  std::stringstream ss;
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i];
    if (i != vec.size() - 1) {
      ss << ",";
    }
  }
  return ss.str();
}
}  // namespace

StreamGraph::StreamGraph(const ComputeGraphPtr &graph) {
  name_ = graph->GetName();
  for (const auto &node : graph->GetAllNodes()) {
    nodes_.emplace_back(std::make_shared<StreamNode>(node));
  }
  Build();
}

std::string StreamGraph::DotId(const StreamNodePtr node) const {
  std::stringstream dot_name;
  dot_name << "S" << node->GetStreamId() << "N" << node->GetIdInStream();
  return dot_name.str();
}

StreamNode::StreamNode(const NodePtr node) : origin_node_(node) {}

size_t StreamNode::GetTopoId() const {
  return static_cast<size_t>(origin_node_->GetOpDesc()->GetId());
}

std::string StreamNode::GetStreamLabel() const {
  std::string stream_label;
  (void)ge::AttrUtils::GetStr(origin_node_->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
  return stream_label;
}

std::vector<uint32_t> StreamNode::GetActivatedStreams() const {
  std::vector<uint32_t> activated_streams;
  (void)ge::AttrUtils::GetListInt(origin_node_->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, activated_streams);
  return activated_streams;
}

int64_t StreamNode::GetStreamId() const {
  return origin_node_->GetOpDesc()->GetStreamId();
}

std::string StreamNode::GetName() const {
  return origin_node_->GetOpDesc()->GetName();
}

std::string StreamNode::GetType() const {
  return origin_node_->GetOpDesc()->GetType();
}

const std::vector<size_t> &StreamNode::GetSendEventIds() const {
  return send_event_ids_;
}

const std::vector<size_t> &StreamNode::GetRecvEventIds() const {
  return recv_event_ids_;
}

const std::vector<uint32_t> &StreamNode::GetLabelIndexes() const {
  return label_indexes_;
}

size_t StreamNode::GetIdInStream() const {
  return idx_;
}

bool StreamNode::IsInBranch() const {
  return in_branch_;
}

void StreamNode::SetIdInStream(size_t id) {
  idx_ = id;
}

void StreamNode::AppendSendEventId(size_t event_id) {
  send_event_ids_.push_back(event_id);
}

void StreamNode::SetRecvEventIds(const std::vector<size_t> &recv_event_ids) {
  recv_event_ids_ = recv_event_ids;
}

std::string StreamGraph::DotStreams() const {
  std::stringstream content;
  for (size_t stream_id = 0; stream_id < stream_num_; stream_id++) {
    std::stringstream ss;
    size_t all_nodes_num = stream_nodes_[stream_id].size();
    size_t node_num_without_events = stream_nodes_without_events_[stream_id].size();
    std::vector<string> node_dot_ids;

    ss << "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">";
    ss << "<tr><td><b>Stream" << stream_id << "</b></td></tr>";
    ss << "<tr><td>" << all_nodes_num << "(" << node_num_without_events << "+"
       << (all_nodes_num - node_num_without_events) << ")nodes</td></tr>";
    if (!stream_labels_[stream_id].empty()) {
      ss << "<tr><td><i>Label:" << stream_labels_[stream_id] << "</i></td></tr>";
    }
    ss << "</table>";
    std::string color;
    if (stream_in_branch_.find(stream_id) != stream_in_branch_.end()) {
      color = ",color=\"blue\"";
    }

    content << "{\n";
    content << "rank=\"same\";\n";
    content << "edge[style=\"solid\",penwidth=2" << color << "];\n";
    content << "Stream" + std::to_string(stream_id) + "[label=<" + ss.str() + ">" + color + "];\n";
    content << "End" + std::to_string(stream_id) + "[shape=\"point\",width=0.1,height=0.1];\n";
    for (auto node : stream_nodes_without_events_[stream_id]) {
      std::string dot_id = DotId(node);
      node_dot_ids.emplace_back(dot_id);
      content << dot_id << "[label=<" << node->GetName() << "<br/>(id:" << node->GetTopoId() << ")";
      const auto &activated_streams = node->GetActivatedStreams();
      if (!activated_streams.empty()) {
        content << "<br/>(active streams:" << VecIntToStr(activated_streams) << ")";
      }
      const auto &label_indexes = node->GetLabelIndexes();
      if (!label_indexes.empty()) {
        content << "<br/>(labels:" << VecIntToStr(label_indexes) << ")";
      }
      content << ">];\n";
    }

    content << "Stream" << stream_id;
    for (auto node_dot_id : node_dot_ids) {
      content << "->" << node_dot_id;
    }
    content << "->End" << stream_id << ";\n";
    content << "}\n";
  }
  return content.str();
}
std::string StreamGraph::DotEdges() const {
  std::stringstream content;
  for (auto node : nodes_) {
    for (size_t send_event_id : node->GetSendEventIds()) {
      if (recv_event_node_map_.find(send_event_id) == recv_event_node_map_.end()) {
        continue;
      }
      auto recv_node = recv_event_node_map_.at(send_event_id);
      content << DotId(node) << "->" << DotId(recv_node);
      content << "[label=\"Event" << send_event_id << "\",arrowhead=\"normal\"];\n";
    }
  }
  return content.str();
}
std::string StreamGraph::DotActives() const {
  if (stream_in_branch_.empty()) {
    return "";
  }
  std::stringstream content;
  content << "{\n";
  for (auto node : nodes_) {
    std::string color;
    std::string font_color;
    if (node->GetType() == STREAMSWITCH) {
      color = ",color=\"blue\"";
      font_color = ",fontcolor=\"blue\"";
    }
    const auto &activated_streams = node->GetActivatedStreams();
    if (activated_streams.size() > 0) {
      content << "AN" << node->GetIdInStream() << "[label=\"" << node->GetName() << "(id: " << node->GetTopoId()
              << ",stream: " << node->GetStreamId() << ")\"];\n";
      for (uint32_t activated_stream : activated_streams) {
        content << "AS" << activated_stream << "[label=\"Stream" << activated_stream << "\"" << font_color << "];\n";
        content << "AN" << node->GetIdInStream() << "-> AS" << activated_stream
                << " [label=\"active\", arrowhead=\"normal\"" << color << font_color << "];\n";
      }
    }
  }
  content << "}\n";
  return content.str();
}

std::string StreamGraph::DotHeaderStr() const {
  std::stringstream content;
  content << "digraph " << name_
          << "{\nrankdir=\"LR\";\nnode[shape=\"plaintext\",width=0,height=0];\nedge[arrowhead=\"none\",style="
             "\"dashed\"];\n\n";
  return content.str();
}
std::string StreamGraph::DotTailStr() const {
  return "}\n";
}

std::string StreamGraph::ToDotString() const {
  std::stringstream content;
  content << DotHeaderStr();
  content << DotStreams();
  content << DotEdges();
  content << DotActives();
  content << DotTailStr();
  return content.str();
}

void StreamGraph::Build() {
  if (nodes_.empty()) {
    return;
  }

  // 1. 建立 id/name 映射，同时设置 idx
  for (size_t idx = 0; idx < nodes_.size(); ++idx) {
    nodes_[idx]->SetIdInStream(idx);
  }

  // 2. 计算最大 stream_id
  auto max_stream_node = std::max_element(
      nodes_.begin(), nodes_.end(),
      [](const StreamNodePtr &lhs, const StreamNodePtr &rhs) { return lhs->GetStreamId() < rhs->GetStreamId(); });
  if (max_stream_node != nodes_.end() && (*max_stream_node != nullptr)) {
    stream_num_ = (*max_stream_node)->GetStreamId() + 1;
  } else {
    return;
  }

  // 3. 按 stream_id 分类节点（含 Send/Recv）
  if (stream_nodes_.size() < stream_num_) {
    stream_nodes_.resize(stream_num_);
  }
  for (auto &node : nodes_) {
    if (node->GetStreamId() >= 0) {
      stream_nodes_[node->GetStreamId()].emplace_back(node);
    }
  }

  // 4. 处理 Send/Recv 事件
  for (auto &nodes_per_stream : stream_nodes_) {
    StreamNodePtr pre_node = nullptr;
    std::vector<size_t> recv_events;
    for (auto &node : nodes_per_stream) {
      if (node->GetType() == SEND) {
        if (pre_node) {
          if (auto pos = node->GetName().find("_Send_"); pos != std::string::npos) {
            int event_id = std::stoi(node->GetName().substr(pos + 6));
            pre_node->AppendSendEventId(event_id);
          }
        }
      } else if (node->GetType() == RECV) {
        if (auto pos = node->GetName().find("_Recv_"); pos != std::string::npos) {
          int event_id = std::stoi(node->GetName().substr(pos + 6));
          recv_events.push_back(event_id);
        }
      } else {
        node->SetRecvEventIds(recv_events);
        recv_events.clear();
        pre_node = node;
      }
    }
  }

  // 5. 移除 Send/Recv 节点
  nodes_.erase(
      std::remove_if(nodes_.begin(), nodes_.end(),
                     [](const StreamNodePtr &node) { return (node->GetType() == SEND) || (node->GetType() == RECV); }),
      nodes_.end());

  // 6. 建立事件映射
  for (auto &node : nodes_) {
    for (auto id : node->GetSendEventIds()) {
      send_event_node_map_[id] = node;
    }
    for (auto id : node->GetRecvEventIds()) {
      recv_event_node_map_[id] = node;
    }
  }

  // 7. 处理 stream 标签和无事件节点
  stream_nodes_without_events_.resize(stream_num_);
  stream_labels_.resize(stream_num_);
  for (auto &node : nodes_) {
    if (node->GetStreamId() < 0) {
      continue;
    }
    stream_nodes_without_events_[node->GetStreamId()].push_back(node);
    if (!node->GetStreamLabel().empty() && stream_labels_[node->GetStreamId()].empty()) {
      stream_labels_[node->GetStreamId()] = node->GetStreamLabel();
    }
    if (node->GetType() == STREAMSWITCH || (node->GetType() == STREAMACTIVE && node->IsInBranch())) {
      for (uint32_t activated_stream : node->GetActivatedStreams()) {
        stream_in_branch_.insert(activated_stream);
      }
    }
  }
}
}  // namespace ge
