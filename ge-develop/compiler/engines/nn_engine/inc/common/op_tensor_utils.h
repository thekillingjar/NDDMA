/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_OP_TENSOR_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_OP_TENSOR_UTILS_H_

#include <vector>
#include "ops_store/op_kernel_info.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class OpTensorUtils {
 public:
  static Status CalcTensorSize(const std::vector<int64_t> &dims, const ge::DataType &data_type,
                               int32_t output_real_calc_flag, int64_t &tensor_size);

  static Status CalcTensorSize(const ge::GeTensorDesc &tensor_desc,
                               const int32_t output_real_calc_flag, int64_t &tensor_size);

  static bool IsStaticReuseBinaryOp(const ge::OpDescPtr &op_desc);

  static bool IsFuzzBuildOp(const ge::OpDesc &op_desc);

  static Status CalibrateTensorSize(int64_t &tensor_size);
 private:
  /**
   * verify the data type, format and shape of this tensor
   * @param tensor_desc Tensor object
   * @return Status
   */
  static Status VerifyTensor(const std::vector<int64_t> &dims, const ge::DataType &data_type);

  /**
   * Check whether the multiplication of a set of int64_t number a,b may exceed
   * the maximum of int64_t
   * And do the multiplication at the same time
   * @param nums a set of int64_t number
   * @param result multiplication result
   * @return Status
   */
  static Status ArrayMultiplyInt64WithVerify(const std::vector<int64_t> &dims, int64_t &result);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_OP_TENSOR_UTILS_H_
