/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_GREATER_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_GREATER_KERNEL_H_

#include <set>
#include <vector>

#include "graph/ge_tensor.h"
#include "host_kernels/kernel.h"

namespace ge {
class GreaterKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status ComputeOutData(ConstGeTensorPtr input_x1, ConstGeTensorPtr input_x2, std::vector<int64_t> &x1_indexes,
                        std::vector<int64_t> &x2_indexes, std::vector<uint8_t> &y_data);

  Status GreaterCheck(const std::vector<ConstGeTensorPtr> &input);

  const std::set<DataType> greater_supported_type = {
      DT_FLOAT, DT_FLOAT16, DT_INT8,   DT_INT16,  DT_UINT16, DT_UINT8,
      DT_INT32, DT_INT64,   DT_UINT32, DT_UINT64, DT_BOOL,   DT_DOUBLE,
  };
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_GREATER_KERNEL_H_
