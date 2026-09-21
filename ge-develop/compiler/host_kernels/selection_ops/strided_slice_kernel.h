/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_

#include <vector>
#include "host_kernels/kernel.h"

namespace ge {
class StridedSliceKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr attr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status CheckAndGetAttr(const OpDescPtr &attr);
  static Status CheckInputParam(const std::vector<ConstGeTensorPtr> &input);
  Status InitParamWithAttrs(const std::vector<ConstGeTensorPtr> &input, std::vector<int64_t> &input_dims,
                            std::vector<int64_t> &begin_vec, std::vector<int64_t> &output_dims,
                            std::vector<int64_t> &stride_vec);
  Status MaskCal(const size_t i, int64_t &begin_i, int64_t &end_i, const int64_t &dim_i) const;
  static Status StrideCal(const int64_t x_dims_i, int64_t &begin_i, int64_t &end_i, int64_t &stride_i,
                          int64_t &dim_final);
  void ExpandDimsWithNewAxis(const ConstGeTensorPtr &begin_tensor, const size_t x_dims_num,
                             std::vector<int64_t> &x_dims);
  void ExpandStrideWithEllipsisMask(const size_t x_dims_num, const std::vector<int64_t> &x_dims,
                                    std::vector<int64_t> &orig_begin_vec, std::vector<int64_t> &orig_end_vec,
                                    std::vector<int64_t> &orig_stride_vec);

  void GetOutputDims(uint32_t dims_size, const std::vector<int64_t> &output_dims, std::vector<int64_t> &v_dims);

  std::map<std::string, uint32_t> attr_value_map_ = {{STRIDE_SLICE_ATTR_BEGIN_MASK, 0},
                                                     {STRIDE_SLICE_ATTR_END_MASK, 0},
                                                     {STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0},
                                                     {STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0},
                                                     {STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0}};
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_
