/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_DYNAMIC_STITCH_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_DYNAMIC_STITCH_KERNEL_H_

#include <map>
#include <vector>

#include "host_kernels/kernel.h"

namespace ge {
class DynamicStitchKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status ValidateParams(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input);
  void ComputeMergedShape(const std::vector<ConstGeTensorPtr> &input, GeShape &merged_shape) const;
  Status GenData(const std::vector<ConstGeTensorPtr> &input, GeTensorPtr &output_ptr) const;
  Status StitchDataFollowIndices(int64_t data_unit, const std::vector<ConstGeTensorPtr> &input, int64_t allowance,
                                 std::unique_ptr<uint8_t[]> &buf) const;
  int32_t n_;  // data input number
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_DYNAMIC_STITCH_KERNEL_H_
