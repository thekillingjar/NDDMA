/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/graph_fusion/connection_matrix.h"
#include "framework/common/debug/ge_log.h"

namespace fe {
ConnectionMatrix::ConnectionMatrix() : enable_data_flow_(false) {}
ConnectionMatrix::ConnectionMatrix(bool enable_data_flow) : enable_data_flow_(enable_data_flow) {}
ConnectionMatrix::ConnectionMatrix(const ge::ComputeGraph &graph) : enable_data_flow_(false) {
  (void)graph;
}

ConnectionMatrix::~ConnectionMatrix() {
  bit_maps.clear();
  name_to_index_.clear();
  bit_maps_back_up_.clear();
  data_bit_maps_.clear();
  data_bit_maps_back_up_.clear();
}

void ConnectionMatrix::Generate(const ge::ComputeGraph &graph) {
  bit_maps.clear();
  data_bit_maps_.clear();
  name_to_index_.clear();
  const ge::ComputeGraph::Vistor<ge::NodePtr> direct_nodes = graph.GetDirectNode();
  size_ = direct_nodes.size();
  bit_maps.reserve(size_);
  int64_t index_loop = 0;
  for (const ge::NodePtr &node : direct_nodes) {
    name_to_index_[node->GetName()] = index_loop;
    (void)bit_maps.emplace_back(size_);
    index_loop++;
  }

  if (enable_data_flow_) {
    data_bit_maps_ = bit_maps;
  }

  for (const ge::NodePtr &node : direct_nodes) {
    const ge::Node::Vistor<ge::NodePtr> inputs = node->GetInAllNodes();
    SetConnectivity(inputs, node);
    if (enable_data_flow_) {
      const ge::Node::Vistor<ge::NodePtr> data_inputs = node->GetInDataNodes();
      SetDataConnectivity(data_inputs, node);
    }
  }
}

void ConnectionMatrix::Update(const ge::ComputeGraph &graph, const std::vector<ge::NodePtr> &fusion_nodes) {
  ge::LargeBitmap new_bit_vector(graph.GetDirectNodesSize());
  new_bit_vector.SetValues(0U);
  std::vector<uint64_t> fusion_indexs(fusion_nodes.size(), 0);
  for (size_t i = 0U; i < fusion_nodes.size(); ++i) {
    const uint64_t index = static_cast<uint64_t>(GetIndex(fusion_nodes[i]));
    new_bit_vector.Or(GetBitMap(index));
    fusion_indexs[i] = index;
  }

  for (ge::LargeBitmap &node_map: bit_maps) {
    for (size_t i = 0U; i < fusion_nodes.size(); ++i) {
      if (node_map.GetBit(fusion_indexs[i])) {
        node_map.Or(new_bit_vector);
        break;
      }
    }
  }

  if (enable_data_flow_) {
    new_bit_vector.SetValues(0U);
    for (size_t i = 0U; i < fusion_nodes.size(); ++i) {
      const uint64_t index = static_cast<uint64_t>(GetIndex(fusion_nodes[i]));
      new_bit_vector.Or(GetDataBitMap(index));
    }
    for (ge::LargeBitmap &node_map: data_bit_maps_) {
      for (size_t i = 0U; i < fusion_nodes.size(); ++i) {
        if (node_map.GetBit(fusion_indexs[i])) {
          node_map.Or(new_bit_vector);
          break;
        }
      }
    }
  }
}

void ConnectionMatrix::BackupBitMap() {
  bit_maps_back_up_ = bit_maps;
  data_bit_maps_back_up_ = data_bit_maps_;
}

void ConnectionMatrix::RestoreBitMap() {
  bit_maps = bit_maps_back_up_;
  data_bit_maps_ = data_bit_maps_back_up_;
}

void ConnectionMatrix::SetConnectivity(const ge::Node::Vistor<ge::NodePtr> &inputs, const ge::NodePtr &node) {
  ge::LargeBitmap &bitmap = GetBitMap(node);
  if (std::find(inputs.begin(), inputs.end(), node) == inputs.end()) {
    bitmap.SetValues(0);
  }

  bitmap.SetBit(static_cast<size_t>(GetIndex(node)));
  for (const ge::NodePtr &input : inputs) {
    if (input != node) {
      bitmap.Or(GetBitMap(input));
    }
  }
}

void ConnectionMatrix::SetDataConnectivity(const ge::Node::Vistor<ge::NodePtr> &inputs, const ge::NodePtr &node) {
  ge::LargeBitmap &bitmap = GetDataBitMap(node);
  if (std::find(inputs.begin(), inputs.end(), node) == inputs.end()) {
    bitmap.SetValues(0);
  }

  bitmap.SetBit(static_cast<size_t>(GetIndex(node)));
  for (const ge::NodePtr &input : inputs) {
    if (input != node) {
      bitmap.Or(GetDataBitMap(input));
    }
  }
}

int64_t ConnectionMatrix::GetIndex(const ge::NodePtr &node) const {
  const auto iter = name_to_index_.find(node->GetName());
  if (iter != name_to_index_.end()) {
    return iter->second;
  } else {
    GELOGW("Node %s was not found in name_to_index_.", node->GetName().c_str());
    return 0;
  }
}

bool ConnectionMatrix::IsConnected(const ge::NodePtr &a, const ge::NodePtr &b) const {
  return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

bool ConnectionMatrix::IsDataConnected(const ge::NodePtr &a, const ge::NodePtr &b) const {
  return GetDataBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

const ge::LargeBitmap &ConnectionMatrix::GetBitMap(const ge::NodePtr &node) const {
  return bit_maps[static_cast<uint64_t>(GetIndex(node))];
}

ge::LargeBitmap &ConnectionMatrix::GetBitMap(const ge::NodePtr &node) {
  return bit_maps[static_cast<uint64_t>(GetIndex(node))];
}

ge::LargeBitmap &ConnectionMatrix::GetBitMap(uint64_t index) {
  return bit_maps[index];
}

const ge::LargeBitmap &ConnectionMatrix::GetDataBitMap(const ge::NodePtr &node) const {
  return data_bit_maps_[static_cast<uint64_t>(GetIndex(node))];
}

ge::LargeBitmap &ConnectionMatrix::GetDataBitMap(const ge::NodePtr &node) {
  return data_bit_maps_[static_cast<uint64_t>(GetIndex(node))];
}

ge::LargeBitmap &ConnectionMatrix::GetDataBitMap(uint64_t index) {
  return data_bit_maps_[index];
}
}
