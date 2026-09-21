/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_FLOORDIV_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_FLOORDIV_KERNEL_H_

#include <vector>

#include "host_kernels/kernel.h"

namespace ge {
class FloorDivKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status FloorDivCheck(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) const;
  void ShapeCal(const std::vector<ConstGeTensorPtr> &input, GeTensorPtr output_ptr) const;
  template <typename T>
  T DivCal(const T &x_i, const T &y_i);

  template <typename T>
  bool ZeroCheck(const T &element, DataType data_type) const;

  template <typename T>
  Status DataCalBroadcast(const T &x, const T &y, size_t num_x, size_t num_y, DataType data_type,
                          GeTensorPtr output_ptr);
  template <typename T>
  Status DataCal(const std::vector<ConstGeTensorPtr> &input, ge::GeTensorPtr output_ptr);
  Status ComputeByDataType(DataType data_type, const std::vector<ConstGeTensorPtr> &input, GeTensorPtr output_ptr);
};

template <>
bool FloorDivKernel::ZeroCheck<float32_t>(const float32_t &element, DataType data_type) const;

template <>
bool FloorDivKernel::ZeroCheck<float64_t>(const float64_t &element, DataType data_type) const;
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_FLOORDIV_KERNEL_H_
