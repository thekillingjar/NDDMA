/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MEMCPY_ADDR_ASYNC_PASS_H_
#define GE_GRAPH_PASSES_MEMCPY_ADDR_ASYNC_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {

class MemcpyAddrAsyncPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status AddMemcpyAddrAsyncNode(const ComputeGraphPtr &graph, const NodePtr &node);
  Status AddMemcpyAsyncNode(const NodePtr &node);
  Status AddMemcpyAsyncNodeForDsa(const NodePtr &node);
  void FindUserData(const NodePtr &parent_node, uint32_t &parent_index);
  void FindUserDataForKnown(const NodePtr &parent_node, const uint32_t &parent_index);
  void FindUserDataForNonDynamic(const ge::NodePtr &parent_node, uint32_t &parent_index);
  bool IsEmptyTenor(const GeShape &shape) const;

  NodePtr CreateMemcpyAddrAsyncNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor,
                                    const NodePtr &out_of_user_data) const;
  Status InsertMemcpyAddrAsyncNode(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &in_anchor,
                                   const NodePtr &node) const;
  bool NeedInsertMemAddrAsyncNodeAfterData(const ComputeGraphPtr &graph) const;
  Status InsertMemAddrAsyncNodeBeforeNetoutput(const ComputeGraphPtr &graph, const NodePtr &node) const;
  Status InsertMemAddrAsyncNodeBetweenHcclAndRefdata(const ComputeGraphPtr &graph, const NodePtr &node) const;
  bool IsFeatureMapRefreshable(const ComputeGraphPtr &graph, const ComputeGraphPtr &sub_graph) const;
  bool IsFeatureMapRefreshableInStaticGraph(const ComputeGraphPtr &graph) const;

  NodePtr user_data_;
  NodePtr out_of_user_data_;
  OutDataAnchorPtr peer_out_anchor_;
  InDataAnchorPtr in_anchor_;
  bool find_user_data_ = false;
  NodePtr user_data_for_known_;
  NodePtr out_of_user_data_for_known_;
  OutDataAnchorPtr peer_out_anchor_for_known_;
  InDataAnchorPtr in_anchor_for_known_;
  bool find_user_data_for_known_ = false;
  bool known_sub_graph_ = false;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MEMCPY_ADDR_ASYNC_PASS_H_
