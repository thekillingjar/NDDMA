/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"

#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "host_kernels/kernel.h"

namespace ge {
namespace {
const std::string kPassName = "DimensionComputePass";
}
bool DimensionComputePass::NeedIgnorePass(const NodePtr &node) {
  if (AreAllOutputsEmptyShape(node->GetOpDesc())) {
    GELOGI("Cur node %s is potential empty const, ignore pass.", node->GetName().c_str());
    return true;
  }
  return false;
}

bool DimensionComputePass::NeedFold() const {
  return need_fold_;
}

 /// Compute node dimension by host kernel
 /// @param node
 /// @param outputs  SUCCESS: compute success.
 ///                 UNSUPPORTED: not support compute, do nothing
 ///                 NOT_CHANGED: dynamic_shape or param invalid
 ///                 INTERNAL_ERROR: after compute, output weight is empty
 /// @return
Status DimensionComputePass::ComputePotentialWeight(NodePtr &node, std::vector<GeTensorPtr> &outputs) {
  auto op_kernel = folding_pass::GetKernelByType(node);
  if (op_kernel == nullptr || folding_pass::IsNoNeedConstantFolding(node)) {
    return UNSUPPORTED;
  }

  return op_kernel->Compute(node, outputs);
}

string DimensionComputePass::GetPassName() const {
  return kPassName;
}

REG_PASS_OPTION("DimensionComputePass").LEVELS(OoLevel::kO1).SWITCH_OPT(ge::OO_CONSTANT_FOLDING);
}  // namespace ge
