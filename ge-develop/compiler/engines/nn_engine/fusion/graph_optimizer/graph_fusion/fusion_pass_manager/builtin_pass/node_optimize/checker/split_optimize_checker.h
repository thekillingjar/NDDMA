/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_CHECKER_SPLIT_OPTIMIZE_CHECKER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_CHECKER_SPLIT_OPTIMIZE_CHECKER_H_

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/node_optimize_checker_base.h"
namespace fe {
class SplitOptimizeChecker : public NodeOptimizeCheckerBase {
 public:
  bool Check(const ge::NodePtr &node_ptr) const;

 private:
  /**
   * Check if ther ara multiple data outputs.
   * @param node_ptr node
   * @return true or false
   */
  bool IsMultiDataOutputs(const ge::NodePtr &node_ptr) const;

  /**
   * Check if the dim C is aligned by 16(dtype=float16) 32(dtype=int8) or 64(dtype=int4).
   * @param node_ptr node
   * @return true or false
   */
  bool IsDimCAligned(const ge::NodePtr &node_ptr) const;
};
}  // namespace fe

#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_CHECKER_SPLIT_OPTIMIZE_CHECKER_H_
