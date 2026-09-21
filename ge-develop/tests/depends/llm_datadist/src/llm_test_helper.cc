/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_test_helper.h"

namespace llm {
constexpr ge::float32_t kDefaultFloatValue = 0.0f;

void LLMTestUtils::BuildInput(ge::Tensor &input_tensor, const std::vector<int64_t> &shape, const size_t input_size) {
  std::vector<domi::float32_t> input_tensor_data(input_size, kDefaultFloatValue);
  ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
  input_tensor.SetTensorDesc(desc);
  input_tensor.SetData(reinterpret_cast<uint8_t *>(input_tensor_data.data()), input_size * 4U);
}

} // namespace llm
