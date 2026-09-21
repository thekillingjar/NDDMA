/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_AUTOFUSE_LOWERING_LOWER_CONCAT_HELPER_H_
#define ASCGEN_DEV_AUTOFUSE_LOWERING_LOWER_CONCAT_HELPER_H_

#include "graph/node.h"
#include "graph/compute_graph.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "ascendc_ir.h"

namespace ge {
class LowerConcatHelper {
 public:
  explicit LowerConcatHelper(NodePtr fused_asc_backend_node);
  static graphStatus LiftingPoorPerfFusedAscBackendOps(const ComputeGraphPtr &graph);
 private:
  enum class ConcatCase : int32_t {
    kFirstDim = 0,
    kAllAligned = 1,
    kOther = 2,
    kNoLifting = 3,
  };
  graphStatus LiftingPoorPerfFusedAscBackendOp();
  bool FindConcatAscBackendNode();
  static AscNodePtr FindConcatNode(const AscGraph &asc_graph);

  graphStatus ParseConcatNode();
  graphStatus ParseConcatCase();
  graphStatus NeedLifting(bool &need_lifting);
  graphStatus UnfoldFusedAscBackend() const;
  bool CheckGraph() const;
  bool HasBackwardFusion() const;
  bool IsTile() const;

  ComputeGraphPtr graph_;
  NodePtr fused_asc_backend_node_;
  NodePtr concat_asc_backend_node_;
  AscNodePtr concat_node_;
  std::vector<ge::Expression> output_shape_;
  std::vector<std::vector<ge::Expression>> input_shapes_;
  ConcatCase case_ = ConcatCase::kNoLifting;
  size_t concat_dim_ = 0;
  int64_t total_fused_dim_size_ = 0L;
  int64_t output_dim_size_ = -1L;
};
}  // namespace ge

#endif  // ASCGEN_DEV_AUTOFUSE_LOWERING_LOWER_CONCAT_HELPER_H_
