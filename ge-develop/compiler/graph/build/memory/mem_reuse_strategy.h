/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEM_REUSE_STRATEGY_H
#define GE_GRAPH_BUILD_MEM_REUSE_STRATEGY_H

#include <string>
#include <vector>
#include <list>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"

namespace ge {
constexpr int32_t kInvalidThreadScopeId = -1;
constexpr int64_t kInvalidStreamId = -1;

class MemReuseUtils {
 public:
  static int32_t GetThreadScopeId(const ge::OpDesc *const desc);
  static int64_t GetStreamId(const ge::OpDesc *const desc);
  static void SetStreamId(ge::OpDesc *const desc, int64_t stream_id);
  static bool IsMergeNode(const NodePtr &node);
  static bool IsMergeNode(const Node *node);
  static bool IsNoPaddingContinuousInput(const Node *node);
  static Status GetNoAlignSize(const GeTensorDesc &tensor, size_t &size);
  static Status GetOutputNoAlignSize(const ge::OpDesc &desc, uint32_t index, size_t &size);
  static bool IsAllOutRefAllInput(const NodePtr &node);
  static Status GetTensorSize(const GeTensorDesc &tensor_desc, int64_t &size, const bool need_split_size = false);
  static bool IsNeedSplitSize(const ge::NodePtr &node, const uint32_t index);
  static bool IsNeedSplitSize(const ge::Node *const node, const uint32_t index);
  static void AlignMemOffset(size_t &mem_align_size);
  static Status GetSrcNodeThroughRefNode(const Node *const node, const int32_t input_index, Node *&src_node,
                                         int32_t &out_index);
  static Status GetDstNodeThroughRefNode(const Node *const node, const int32_t out_index,
                                         std::vector<Node *> &dst_nodes, std::vector<int32_t> &in_indexes);
  static std::string GetGraphNameId(const ge::ComputeGraph *const graph);
};

class MemReuseStrategy {
 public:
  static bool GetDiffStreamPrior(const Node *const node);
  static void OptimizeDiffStream(const Node *const node);
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEM_REUSE_STRATEGY_H_
