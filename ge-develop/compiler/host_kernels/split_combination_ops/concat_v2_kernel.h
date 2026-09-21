/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_CONCAT_V2_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_CONCAT_V2_KERNEL_H_

#include <vector>

#include "host_kernels/kernel.h"

namespace ge {
class ConcatV2Kernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status ConcatV2PreCompute(const std::vector<ConstGeTensorPtr> &input, int32_t &tidx, ConstGeTensorPtr &tensor) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_CONCAT_V2_KERNEL_H_
