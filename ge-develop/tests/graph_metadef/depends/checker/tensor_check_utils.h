/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_GRAPH_MATEDEF_UT_GRAPH_COMMON_TENSOR_CHECK_UTILS_H
#define TESTS_GRAPH_MATEDEF_UT_GRAPH_COMMON_TENSOR_CHECK_UTILS_H

#include "ge_tensor.h"
#include "ge_common/ge_api_types.h"
#include "runtime/tensor.h"

namespace ge {
class TensorCheckUtils {
 public:
  static std::string ShapeStr(const gert::Shape &shape);
  static std::string ShapeStr(const ge::Shape &shape);
  static std::string ShapeStr(const ge::GeShape &shape);
  static void ConstructGertTensor(gert::Tensor &gert_tensor, const std::initializer_list<int64_t> &dims = {1, 2, 3},
                                  const DataType &data_type = DataType::DT_INT32,
                                  const Format &format = Format::FORMAT_ND,
                                  const gert::TensorPlacement placement = gert::TensorPlacement::kOnDeviceHbm);
  static void ConstructGeTensor(ge::GeTensor &ge_tensor, const std::initializer_list<int64_t> &dims = {1, 2, 3},
                                const DataType &data_type = DataType::DT_INT32,
                                const Format &format = Format::FORMAT_ND,
                                const ge::Placement placement = ge::Placement::kPlacementHost);
  // 校验地址
  static Status CheckGeTensorEqGertTensor(const GeTensor &ge_tensor, const gert::Tensor &gert_tensor);
  // 校验数据
  static Status CheckGeTensorEqGertTensorWithData(const GeTensor &ge_tensor, const gert::Tensor &gert_tensor);
  static Status CheckTensorEqGertTensor(const Tensor &tensor, const gert::Tensor &gert_tensor);
};
}  // namespace ge
#endif  // TESTS_GRAPH_MATEDEF_UT_GRAPH_COMMON_TENSOR_CHECK_UTILS_H
