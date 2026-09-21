/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_SPACESIZE_CALCULATOR_TENSOR_COMPUTE_UTIL_H_
#define FUSION_ENGINE_UTILS_COMMON_SPACESIZE_CALCULATOR_TENSOR_COMPUTE_UTIL_H_

#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "graph/ge_tensor.h"
#include "graph/types.h"

namespace fe {
class TensorComputeUtil {
 public:
  /**
   * verify the data type, format and shape of this tensor
   * @param tensor_desc Tensor object
   * @return Status
   */
  static Status VerifyTensor(const ge::GeTensorDesc &tensor_desc);

  /**
   * calculate the tensor element count by multiply all the dim of shape
   * @param tensor_desc Tensor object
   * @param element_cnt output value
   * @return Status
   */
  static Status GetElementCountByMultiply(const ge::GeTensorDesc &tensor_desc, int64_t &element_cnt);

  /**
   * calculate the tensor element count by multiply all the dim of shape
   * @param shape shape object
   * @param element_cnt output value
   * @return Status
   */
  static Status GetElementCountByMultiply(const ge::GeShape &shape, int64_t &element_cnt);

  /**
   *
   * @param element_cnt
   * @param data_type
   * @param tensor_size
   * @return
   */
  static Status GetTensorSizeByDataType(int64_t &element_cnt, ge::DataType &data_type, int64_t &tensor_size,
                                        int32_t &output_real_calc_flag);

  static Status CalcTensorSize(ge::GeTensorDesc &tensor_desc, int64_t &tensor_size, int32_t &output_real_calc_flag);

 private:
  /**
   * Check whether the multiplication of a set of int64_t number a,b may exceed
   * the maximum of int64_t
   * And do the multiplication at the same time
   * @param nums a set of int64_t number
   * @param result multiplication result
   * @return Status
   */
  static Status ArrayMultiplyInt64WithVerify(const std::vector<int64_t> &nums, int64_t &result);
};
}  // namespace fe

#endif  // FUSION_ENGINE_UTILS_COMMON_SPACESIZE_CALCULATOR_TENSOR_COMPUTE_UTIL_H_
