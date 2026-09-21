/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_tensor.h"
#include "common/mem_utils.h"
#include "common/llm_checker.h"
#include "common/def_types.h"
#include "common/llm_utils.h"

namespace py = pybind11;

namespace llm {
std::map<uintptr_t, std::shared_ptr<ge::Tensor>> LLMTensor::ge_tensors;
std::mutex LLMTensor::mu_tensors;

namespace {
std::vector<ge::AscendString> SplitToStrVector(const char *data_ptr, const size_t &data_size, const size_t &ele_num) {
  std::vector<ge::AscendString> res;
  const size_t byte_num_per_ele = data_size / ele_num;
  for (size_t i = 0UL; i < ele_num; ++i) {
    (void) res.emplace_back(data_ptr + i * byte_num_per_ele);
  }
  return res;
}

bool CheckMulOverflow(int64_t a, int64_t b) {
  if ((a == 0) || (b == 0)) {
    return false;
  }
  const auto aAbs = std::abs(a);
  const auto bAbs = std::abs(b);
  if (INT64_MAX / aAbs < bAbs) {
    return true;
  }
  return false;
}
}  // namespace

ge::Status LLMTensor::Init(uintptr_t data_ptr,
                           size_t data_size,
                           ge::DataType data_type,
                           std::vector<int64_t> &dims,
                           ge::Tensor &tensor) {
  ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, data_type);
  (void) tensor.SetTensorDesc(desc);
  if (data_type == ge::DataType::DT_STRING) {
    const int64_t shape_size = desc.GetShape().GetShapeSize();
    const size_t ele_num = shape_size < 0L ? 1UL : static_cast<size_t>(shape_size);
    const auto &string_vec = SplitToStrVector(reinterpret_cast<const char *>(data_ptr), data_size, ele_num);
    (void) tensor.SetData(string_vec);
  } else {
    (void) tensor.SetData(reinterpret_cast<const uint8_t *>(data_ptr), data_size);
  }
  return ge::SUCCESS;
}

void LLMTensor::ComputeStrides(ssize_t item_size, const std::vector<int64_t> &dims, std::vector<ssize_t> &strides) {
  if (!dims.empty()) {
    (void) strides.emplace_back(item_size);
    auto stride = static_cast<int64_t>(item_size);
    for (auto it = dims.crbegin(); it != (dims.crend() - 1); it++) {
      if (CheckMulOverflow(stride, *it)) {
        throw std::overflow_error("Compute stride overflow.");
      }
      stride *= *it;
      (void) strides.emplace_back(static_cast<ssize_t>(stride));
    }
    std::reverse(strides.begin(), strides.end());
  }
}

int64_t LLMTensor::CalcTensorSize(const std::vector<int64_t> &shape, int32_t data_type) {
  int64_t tensor_size = -1;
  (void) llm::LLMUtils::CalcTensorMemSize(shape,
                                            static_cast<ge::DataType>(data_type),
                                            tensor_size);
  return tensor_size;
}

std::vector<uintptr_t> LLMTensor::BuildNpuTensors(const std::vector<int64_t> &shape,
                                                  int32_t data_type,
                                                  size_t tensor_size,
                                                  const std::vector<uintptr_t> &addresses) {
  ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, static_cast<ge::DataType>(data_type));
  tensor_desc.SetPlacement(ge::Placement::kPlacementDevice);
  std::vector<uintptr_t> tensor_ids;
  tensor_ids.reserve(addresses.size());
  for (const auto address : addresses) {
    ge::Tensor tensor(tensor_desc);
    const auto npu_addr = llm::PtrToPtr<void, uint8_t>(llm::ValueToPtr(address));
    tensor.SetData(npu_addr, static_cast<size_t>(tensor_size), [](uint8_t *) {});
    auto tensor_id = AddTensor(tensor);
    tensor_ids.emplace_back(tensor_id);
  }
  return tensor_ids;
}

py::memoryview LLMTensor::ToReadonlyMemoryView(ge::Tensor &tensor) {
  const auto tensor_desc = tensor.GetTensorDesc();
  const auto data_type = tensor_desc.GetDataType();
  LLMLOGD("Transfer tensor data type:%d to numpy.", static_cast<int32_t>(data_type));
  const auto item_size = static_cast<ssize_t>(ge::GetSizeByDataType(data_type));
  const auto dims = tensor_desc.GetShape().GetDims();
  std::vector<ssize_t> strides;
  LLMTensor::ComputeStrides(item_size, dims, strides);

  switch (data_type) {
    case ge::DT_BOOL: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<bool>::value,
        dims, strides, true);
    }
    case ge::DT_INT8: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<int8_t>::value,
        dims, strides, true);
    }
    case ge::DT_UINT8: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<uint8_t>::value,
        dims, strides, true);
    }
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
    case ge::DT_INT16: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<int16_t>::value,
        dims, strides, true);
    }
    case ge::DT_UINT16: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<uint16_t>::value,
        dims, strides, true);
    }
    case ge::DT_INT32: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<int32_t>::value,
        dims, strides, true);
    }
    case ge::DT_UINT32: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<uint32_t>::value,
        dims, strides, true);
    }
    case ge::DT_INT64: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<int64_t>::value,
        dims, strides, true);
    }
    case ge::DT_UINT64: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<uint64_t>::value,
        dims, strides, true);
    }
    case ge::DT_FLOAT: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<float>::value,
        dims, strides, true);
    }
    case ge::DT_DOUBLE: {
      return py::memoryview::from_buffer(tensor.GetData(), item_size, py::format_descriptor<double>::value,
        dims, strides, true);
    }
    default: {
      return py::memoryview::from_memory(tensor.GetData(), tensor.GetSize(), true);
    }
  }
}

uintptr_t LLMTensor::BuildTensor(uintptr_t data_ptr, size_t data_size, int32_t data_type, std::vector<int64_t> &dims) {
  ge::Tensor tensor;
  (void) LLMTensor::Init(data_ptr, data_size, static_cast<ge::DataType>(data_type), dims, tensor);
  return AddTensor(tensor);
}

std::pair<ge::Status, py::memoryview> LLMTensor::GetBuffer(uintptr_t tensor_id) {
  const auto tensor = GetTensor(tensor_id);
  std::pair<ge::Status, py::memoryview> result{ge::FAILED, py::memoryview::from_memory(nullptr, 0, true)};
  LLM_LOGE_IF(tensor == nullptr, "Failed to find tensor, id = %lu", tensor_id);
  if (tensor != nullptr) {
    result.second = LLMTensor::ToReadonlyMemoryView(*tensor);
    result.first = ge::SUCCESS;
  }
  return result;
}

std::vector<std::string> LLMTensor::GetStringTensor(uintptr_t tensor_id) {
  const auto tensor = GetTensor(tensor_id);
  LLM_ASSERT_NOTNULL(tensor, "Failed to find tensor, id = %lu", tensor_id);
  const auto tensor_desc = tensor->GetTensorDesc();
  const int64_t shape_size = tensor_desc.GetShape().GetShapeSize();
  const size_t ele_num = shape_size < 0L ? 1UL : static_cast<size_t>(shape_size);
  std::vector<std::string> tensor_strs;
  if (tensor->GetData() != nullptr) {
    for (size_t i = 0UL; i < ele_num; ++i) {
      auto head = reinterpret_cast<const ge::StringHead *>(tensor->GetData()) + i;
      (void) tensor_strs.emplace_back(reinterpret_cast<const char *>(tensor->GetData() + head->addr));
    }
  }
  return tensor_strs;
}

uintptr_t LLMTensor::CloneTensor(uintptr_t tensor_id) {
  std::shared_ptr<ge::Tensor> new_tensor;
  const auto tensor = GetTensor(tensor_id);
  LLM_LOGE_IF(tensor == nullptr, "Failed to find tensor, id = %lu", tensor_id);
  if (tensor != nullptr) {
    new_tensor = llm::MakeShared<ge::Tensor>(tensor->Clone());
  }
  const auto key = reinterpret_cast<uintptr_t>(new_tensor.get());
  if (new_tensor != nullptr) {
    std::lock_guard<std::mutex> lk(mu_tensors);
    ge_tensors[key] = std::move(new_tensor);
  }
  return key;
}

std::shared_ptr<ge::Tensor> LLMTensor::GetTensor(uintptr_t tensor_id) {
  std::lock_guard<std::mutex> lk(mu_tensors);
  const auto it = ge_tensors.find(tensor_id);
  return it != ge_tensors.cend() ? it->second : nullptr;
}

void LLMTensor::DestroyTensor(uintptr_t tensor_id) {
  std::lock_guard<std::mutex> lk(mu_tensors);
  ge_tensors.erase(tensor_id);
  LLMLOGD("DestroyTensor %lu", tensor_id);
}

uintptr_t LLMTensor::AddTensor(const ge::Tensor &tensor) {
  auto shared_tensor = llm::MakeShared<ge::Tensor>(tensor);
  LLM_LOGE_IF(shared_tensor == nullptr, "Failed to create tensor");
  const auto tensor_id = reinterpret_cast<uintptr_t>(shared_tensor.get());
  if (shared_tensor != nullptr) {
    std::lock_guard<std::mutex> lk(mu_tensors);
    ge_tensors[tensor_id] = std::move(shared_tensor);
    LLMLOGD("AddTensor %lu", tensor_id);
  }
  return tensor_id;
}

ge::Status LLMTensor::TensorIdsToTensors(const std::vector<uintptr_t> &tensor_ids, std::vector<ge::Tensor> &tensors) {
  tensors.reserve(tensor_ids.size());
  for (const auto tensor_id : tensor_ids) {
    auto tensor = LLMTensor::GetTensor(tensor_id);
    LLM_CHECK_NOTNULL(tensor, "Failed to get tensor by id: %lu", tensor_id);
    tensors.emplace_back(*tensor);
  }
  return ge::SUCCESS;
}

std::vector<TensorIdAndDesc> LLMTensor::TensorsToTensorIdAndDescs(const std::vector<ge::Tensor> &tensors) {
  std::vector<TensorIdAndDesc> results;
  for (auto &tensor : tensors) {
    auto tensor_id = LLMTensor::AddTensor(tensor);
    auto tensor_tuple = std::make_tuple(tensor_id,
                                        static_cast<int32_t>(tensor.GetDataType()),
                                        tensor.GetTensorDesc().GetShape().GetDims());
    results.emplace_back(std::move(tensor_tuple));
  }
  return results;
}
}  // namespace ge
