/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_MERGING_TRANS_NODE_MERGING_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_MERGING_TRANS_NODE_MERGING_H_

#include <stack>
#include "common/fe_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace fe {

template <class T>
using Vistor = RangeVistor<T, std::shared_ptr<ge::ConstAnchor>>;

struct BasicInfoForRemovingNode {
  ge::NodePtr src_node;
  ge::OpDescPtr src_op_desc;
  ge::NodePtr dst_node;
  ge::OpDescPtr dst_op_desc;
  ge::NodePtr node;
  ge::InDataAnchorPtr dst_in_anchor;
  ge::OutDataAnchorPtr src_out_anchor;
  ge::InDataAnchorPtr in_anchor_of_node;
  ge::OutDataAnchorPtr out_anchor_of_node;
  size_t dst_in_anchor_size_of_node;
};

/** @brief Class of merging all reversed trans-nodes
* pairs. */
class TransNodeMerging {
 public:
  explicit TransNodeMerging();

  ~TransNodeMerging();

  TransNodeMerging(const TransNodeMerging &) = delete;

  TransNodeMerging &operator=(const TransNodeMerging &) = delete;

  Status MergeAllTransOps(ge::ComputeGraph &fused_graph, const std::vector<ge::NodePtr> &origin_cast_list,
                          const bool check_list_flag);

 private:
  /* Remove edge between src_anchor and dst Anchor. Then, add two edges which are
   * frome src_anchor to new_node and from new_node to dst_anchor. */
  Status AddEdgesForNewNode(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor, ge::NodePtr new_node);

  Status MergeOneNode(ge::ComputeGraph &fused_graph, ge::NodePtr node, const uint32_t &current_in_anchor_index,
                            std::stack<uint32_t> &in_anchor_index_stack);

  bool CheckTwoTransOpDescMergable(const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                                   const std::vector<ge::NodePtr> &origin_cast_list,
                                   const bool check_list_flag) const;

  Status MergeTransOpBetweenTwoNormalOp(ge::ComputeGraph &fused_graph, ge::NodePtr src_node,
                                        const ge::InDataAnchorPtr &in_anchor,
                                        const std::vector<ge::NodePtr> &origin_cast_list,
                                        const bool check_list_flag);

  uint32_t FindIndexOfCurrentNode(const Vistor<ge::InDataAnchorPtr> &in_data_anchor_ptr_vec,
                                  const ge::InDataAnchorPtr &in_anchor) const;

  bool IsOptimizableCast(const ge::NodePtr &node, const std::vector<ge::NodePtr> &origin_cast_list) const;

  bool IsFloatingCast(const ge::DataType &in_data_type, const ge::DataType &out_data_type) const;

  bool IsFloatingType(const ge::DataType &data_type) const;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_MERGING_TRANS_NODE_MERGING_H_
