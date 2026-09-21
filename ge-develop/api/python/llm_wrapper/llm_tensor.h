/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_LLM_TENSOR_H
#define AIR_CXX_LLM_TENSOR_H

#include <mutex>
#include <map>
#include <vector>
#include "pybind11/pybind11.h"
#include "ge/ge_api.h"

namespace llm {
using TensorIdAndDesc = std::tuple<uintptr_t, int32_t, std::vector<int64_t>>;

class LLMTensor {
 public:
  static uintptr_t BuildTensor(uintptr_t data_ptr, size_t data_size, int32_t data_type, std::vector<int64_t> &dims);
  static void DestroyTensor(uintptr_t tensor_id);
  static uintptr_t CloneTensor(uintptr_t tensor_id);
  static std::vector<std::string> GetStringTensor(uintptr_t tensor_id);
  static std::pair<ge::Status, pybind11::memoryview> GetBuffer(uintptr_t tensor_id);
  static int64_t CalcTensorSize(const std::vector<int64_t> &shape, int32_t data_type);
  static std::vector<uintptr_t> BuildNpuTensors(const std::vector<int64_t> &shape,
                                                int32_t data_type,
                                                size_t tensor_size,
                                                const std::vector<uintptr_t> &addresses);
  static std::shared_ptr<ge::Tensor> GetTensor(uintptr_t tensor_id);
  static uintptr_t AddTensor(const ge::Tensor &tensor);
  static ge::Status TensorIdsToTensors(const std::vector<uintptr_t> &tensor_ids, std::vector<ge::Tensor> &tensors);
  static std::vector<TensorIdAndDesc> TensorsToTensorIdAndDescs(const std::vector<ge::Tensor> &tensors);

 private:
  static ge::Status Init(uintptr_t data_ptr, size_t data_size,
                         ge::DataType data_type,
                         std::vector<int64_t> &dims,
                         ge::Tensor &tensor);
  static void ComputeStrides(ssize_t item_size, const std::vector<int64_t> &dims, std::vector<ssize_t> &strides);
  static pybind11::memoryview ToReadonlyMemoryView(ge::Tensor &tensor);

  static std::map<uintptr_t, std::shared_ptr<ge::Tensor>> ge_tensors;
  static std::mutex mu_tensors;
};
}  // namespace ge
#endif  // AIR_CXX_LLM_TENSOR_H
