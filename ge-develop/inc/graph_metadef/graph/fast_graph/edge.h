/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_EDGE_H
#define INC_GRAPH_EDGE_H

#include "graph/anchor.h"

namespace ge {
constexpr int32_t kInvalidEdgeIndex = -2;
constexpr int32_t kControlEdgeIndex = -1;

enum class DirectionType {
  kDirectionInType,
  kDirectionOutType,
};

template <class T>
struct Edge {
  T *src = nullptr;                  // src node or output node
  T *dst = nullptr;                  // dst node or input node
  int32_t src_output = -1;           // the output index of output node
  int32_t dst_input = -1;            // the input index of input node
  int32_t in_edge_index = -1;        // the record index of input node, it used to quickly find the edge in node.
  int32_t out_edge_index = -1;       // the record index of output node, it used to quickly find the edge in node.
  Anchor *src_anchor_ptr = nullptr;  // the reserved information.
  Anchor *dst_anchor_ptr = nullptr;  // the reserved information.
};

class FastNode;

struct EdgeEndpoint {
  FastNode *node;
  int32_t index;
  DirectionType type;
};
struct EdgeEndpointWithDirection {
  EdgeEndpointWithDirection() : node(nullptr), index(kInvalidEdgeIndex) {}
  EdgeEndpointWithDirection(FastNode *n, int32_t i) : node(n), index(i) {}
  bool operator<(const EdgeEndpointWithDirection &rhs) const {
    if (node < rhs.node) {
      return true;
    }
    if (node > rhs.node) {
      return false;
    }
    return index < rhs.index;
  }
  bool operator==(const EdgeEndpointWithDirection &rhs) const {
    return (node == rhs.node) && (index == rhs.index);
  }
  FastNode *node;
  int32_t index;
};
using EdgeDstEndpoint = EdgeEndpointWithDirection;
using EdgeSrcEndpoint = EdgeEndpointWithDirection;
}  // namespace ge
#endif  // INC_GRAPH_EDGE_H
