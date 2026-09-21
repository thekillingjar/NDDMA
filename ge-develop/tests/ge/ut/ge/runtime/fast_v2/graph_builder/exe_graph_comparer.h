/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_GRAPH_BUILDER_EXE_GRAPH_COMPARER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_GRAPH_BUILDER_EXE_GRAPH_COMPARER_H_
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
namespace gert {
class ExeGraphComparer {
 public:
  static bool ExpectConnectTo(const ge::NodePtr &node, const std::string &peer_node_type);
  static bool ExpectConnectTo(const ge::NodePtr &node, const std::string &peer_node_type, int32_t out_index, int32_t in_index);

  static bool ExpectConnectFrom(const ge::NodePtr &node, const std::vector<std::string> &from_nodes);
  static bool ExpectConnectFrom(const ge::NodePtr &node, const std::string &from_node, int32_t out_index, int32_t in_index);

  static bool GetAttr(const ge::NodePtr &const_node, ge::Buffer &buffer);

  // for fastnode
  static bool ExpectConnectTo(const ge::FastNode *node, const std::string &peer_node_type);
  static bool ExpectConnectTo(const ge::FastNode *node, const std::string &peer_node_type, int32_t out_index, int32_t in_index);

  static bool ExpectConnectFrom(const ge::FastNode *node, const std::vector<std::string> &from_nodes);
  static bool ExpectConnectFrom(const ge::FastNode *node, const std::string &from_node, int32_t out_index, int32_t in_index);

  static bool GetAttr(const ge::FastNode *const_node, ge::Buffer &buffer);
};
}

#endif  // AIR_CXX_TESTS_UT_GE_RUNTIME_V2_GRAPH_BUILDER_EXE_GRAPH_COMPARER_H_
