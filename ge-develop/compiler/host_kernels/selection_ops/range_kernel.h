/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_RANGE_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_RANGE_KERNEL_H_

#include <vector>

#include "graph/ge_tensor.h"
#include "host_kernels/kernel.h"

namespace ge {
class RangeKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status RangeCheck(const std::vector<ConstGeTensorPtr> &input) const;

  template <typename T>
  Status GetRange(const T start, const T limit, const T delta, GeTensorPtr &output) const;

  template <typename T>
  bool IsZero(const T &element) const;
};

template <>
bool RangeKernel::IsZero<float32_t>(const float32_t &element) const;

template <>
bool RangeKernel::IsZero<float64_t>(const float64_t &element) const;
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_RANGE_KERNEL_H_
