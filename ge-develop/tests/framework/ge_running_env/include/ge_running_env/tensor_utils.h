/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_TENSOR_UTILS_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_TENSOR_UTILS_H_
#include "fake_ns.h"
#include "graph/ge_tensor.h"
#include <vector>
#include <memory>
FAKE_NS_BEGIN
inline GeTensorPtr GenerateTensor(DataType dt, const std::vector<int64_t> &shape) {
  size_t total_num = 1;
  for (auto dim : shape) {
    total_num *= dim; // 未考虑溢出
  }

  auto size = GetSizeInBytes(static_cast<int64_t>(total_num), dt);
  auto data_value = std::unique_ptr<uint8_t[]>(new uint8_t[size]());
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_ND, dt);
  return std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)(data_value.get()),size);
}
inline GeTensorPtr GenerateTensor(const std::vector<int64_t> &shape) {
  return GenerateTensor(DT_FLOAT, shape);
}
inline GeTensorPtr GenerateTensor(DataType dt, std::initializer_list<int64_t> shape) {
  return GenerateTensor(dt, std::vector<int64_t>(shape));
}
inline GeTensorPtr GenerateTensor(std::initializer_list<int64_t> shape) {
  return GenerateTensor(std::vector<int64_t>(shape));
}


FAKE_NS_END
#endif //AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_TENSOR_UTILS_H_
