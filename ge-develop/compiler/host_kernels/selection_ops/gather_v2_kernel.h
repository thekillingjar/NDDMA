/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_GATHER_V2_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_GATHER_V2_KERNEL_H_

#include <vector>

#include "host_kernels/kernel.h"

namespace ge {
class GatherV2Kernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  template <typename T>
  Status ProcessAxis0(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status ProcessAxis1(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status ProcessAxis2(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status ProcessAxis3(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status GenData(const int64_t data_num, ConstGeTensorPtr tensor_x, int64_t axis, GeTensorPtr output);
  Status Check(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
               const std::vector<GeTensorPtr> &v_output) const;
  Status CalcStride(std::vector<int64_t> &stride, std::vector<int64_t> dims) const;
  Status SaveIndicesByDataType(ConstGeTensorPtr indices_tensor_ptr, const GeShape &x_shape, GeShape &indices_shape,
                               DataType indices_data_type, size_t axis);
  Status Process(int64_t axis, DataType data_type, ConstGeTensorPtr input_tensor_ptr, GeTensorPtr output_ptr);
  void DebugPrint(int64_t axis, const GeShape &x_shape, const GeShape &indices_shape,
                  const std::vector<int64_t> &y_shape);

 private:
  std::vector<int64_t> indicates_;
  std::vector<int64_t> xstride_;
  std::vector<int64_t> ystride_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_GATHER_V2_KERNEL_H_
