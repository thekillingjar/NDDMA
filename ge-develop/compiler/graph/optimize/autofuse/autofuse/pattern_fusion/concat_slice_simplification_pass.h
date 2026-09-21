/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CONCAT_SLICE_OPTIMIZE_PASS_H
#define AUTOFUSE_CONCAT_SLICE_OPTIMIZE_PASS_H

#include "operator_factory.h"
#include "graph/node.h"
#include "utils/autofuse_utils.h"

namespace ge {
class ConcatSliceSimplificationPass {
 public:
  graphStatus Run(const ComputeGraphPtr &graph, const GraphPasses &graph_passes, bool &changed);
 private:
  graphStatus HandleSlice(const NodePtr &node);
  static bool FindInput(const NodePtr &concat_node, size_t concat_dim, const std::vector<int64_t> &sizes,
                        std::vector<int64_t> &offsets, size_t &input_index);
  static NodePtr AddNewSliceNode(const ComputeGraphPtr &graph, const std::string &name,
                                 const std::vector<int64_t> &offsets, const std::vector<int64_t> &sizes);

  bool need_prune_ = false;
  bool need_constant_folding_ = false;
};
}  // namespace ge

#endif  // AUTOFUSE_CONCAT_SLICE_OPTIMIZE_PASS_H
