/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_INC_PARAM_CALCULATE_H_
#define FFTS_ENGINE_INC_PARAM_CALCULATE_H_

#include "inc/ffts_error_codes.h"
#include "common/sgt_slice_type.h"
#include "graph/node.h"

namespace ffts {
class ParamCalculator {
 public:
  static Status CalcAutoThreadInput(const ge::NodePtr &node,
                                    vector<vector<vector<int64_t>>> &tensor_slice,
                                    vector<uint64_t> &task_addr_offset,
                                    std::vector<uint32_t> &input_tensor_indexes);
  static Status CalcAutoThreadOutput(const ge::NodePtr &node,
                                     vector<vector<vector<int64_t>>> &tensor_slice,
                                     vector<uint64_t> &task_addr_offset,
                                     std::vector<uint32_t> &output_tensor_indexes);
  static void ConvertTensorSlice(const std::vector<std::vector<std::vector<DimRange>>> &in_tensor_slice,
                                 std::vector<std::vector<std::vector<int64_t>>> &out_tensor_slice);
};
}  // namespace ffts
#endif  // FFTS_ENGINE_INC_PARAM_CALCULATE_H_
