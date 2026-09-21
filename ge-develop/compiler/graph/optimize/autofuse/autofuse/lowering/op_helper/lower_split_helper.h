/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef AUTOFUSE_LOWER_SPLIT_HELPER_H
#define AUTOFUSE_LOWER_SPLIT_HELPER_H
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "lowering/asc_lowerer/loop_api.h"

namespace ge {
class LowerSplitHelper {
 public:
  explicit LowerSplitHelper(NodePtr asc_backend_node);
  graphStatus NeedLifting(bool &need_lifting);
 private:
  enum class SplitCase : int32_t {
    kFirstDim = 0,
    kAllAligned = 1,
    kOther = 2,
    kNoLifting = 3,
  };
  graphStatus CheckFuseRatio(bool &need_lifting);
  graphStatus CheckFuseOtherNodes(bool &need_lifting);
  graphStatus InitAndCheck();
  static void FindSplitNodes(const AscGraph &asc_graph, std::vector<AscNodePtr> &split_nodes);

  graphStatus ParseSplitNode(bool &found);
  graphStatus ParseSplitCase();

  ComputeGraphPtr graph_;
  NodePtr asc_backend_node_;
  std::vector<AscNodePtr> split_nodes_;
  std::vector<std::vector<ge::Expression>> output_shapes_;
  std::vector<ge::Expression> input_shape_;
  SplitCase case_ = SplitCase::kNoLifting;
  size_t split_dim_ = 0;
  int64_t total_fused_dim_size_ = 0L;
  int64_t input_dim_size_ = -1L;
};
}  // namespace ge

#endif  // ASCGEN_DEV_AUTOFUSE_LOWERING_LOWER_SPLIT_HELPER_H_
