/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cinttypes>
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/types.h"
#include "graph/utils/math_util.h"
#include "graph/utils/type_utils.h"
#include "common/checker.h"
#include "exe_graph/lowering/shape_utils.h"

namespace gert {
namespace kernel {
namespace {
void ShapeToStringStream(std::stringstream &ss, const Shape &shape) {
  ss << "[";
  for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
    ss << shape.GetDim(j);
    if (j + 1U < shape.GetDimNum()) {
      ss << ", ";
    }
  }
  ss << "]";
}

std::vector<std::string> PrintShapeAndSize(ge::DataType data_type, const Shape &shape, uint64_t tensor_size) {
  std::stringstream ss;
  ss << "Data type: " << static_cast<size_t>(data_type) << ", Shape: ";
  ShapeToStringStream(ss, shape);
  ss << ", Tensor size: " << tensor_size;
  return {ss.str()};
}

std::vector<std::string> CalcTensorSizeFromShapeTracePrinter(const KernelContext *const context) {
  auto data_type = context->GetInputValue<ge::DataType>(0);
  auto shape = context->GetInputPointer<Shape>(1);
  auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  if ((shape == nullptr) || (tensor_size_ptr == nullptr)) {
    return {"shape or tensor size is null"};
  }
  return PrintShapeAndSize(data_type, *shape, *tensor_size_ptr);
}

std::vector<std::string> FromStorageTracePrinter(const KernelContext *const context) {
  auto data_type = context->GetInputValue<ge::DataType>(0);
  auto storage_shape = context->GetInputPointer<StorageShape>(1);
  auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  if ((storage_shape == nullptr) || (tensor_size_ptr == nullptr)) {
    return {"storage shape or tensor size is null"};
  }
  return PrintShapeAndSize(data_type, storage_shape->GetStorageShape(), *tensor_size_ptr);
}
}  // namespace

ge::graphStatus CalcTensorSizeFromShape(KernelContext *context) {
  auto datatype = context->GetInputValue<ge::DataType>(0);
  auto shape = context->GetInputPointer<Shape>(1);
  auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  if (shape == nullptr || tensor_size_ptr == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  return CalcAlignedSizeByShape(*shape, datatype, *tensor_size_ptr);
}

REGISTER_KERNEL(CalcTensorSizeFromShape)
    .RunFunc(CalcTensorSizeFromShape)
    .TracePrinter(CalcTensorSizeFromShapeTracePrinter);

ge::graphStatus CalcTensorSizeFromStorage(KernelContext *context) {
  auto datatype = context->GetInputValue<ge::DataType>(0);
  auto storage_shape = context->GetInputPointer<StorageShape>(1);
  auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  if (storage_shape == nullptr || tensor_size_ptr == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  return CalcAlignedSizeByShape(storage_shape->GetStorageShape(), datatype, *tensor_size_ptr);
}

ge::graphStatus CalcUnalignedTensorSizeByShape(const Shape &shape, ge::DataType data_type, uint64_t &ret_tensor_size) {
  const auto shape_size = shape.GetShapeSize();
  const auto ret = ge::GetSizeInBytes(shape_size, data_type);
  if (ret < 0) {
    GELOGE(ge::GRAPH_FAILED, "[Calc][TensorSizeByShape] shape_size[%" PRId64 "], data_type[%s]", shape_size,
           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ge::GRAPH_FAILED;
  }
  ret_tensor_size = ret;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcUnalignedTensorSizeFromStorage(KernelContext *context) {
  auto datatype = context->GetInputValue<ge::DataType>(0);
  auto storage_shape = context->GetInputPointer<StorageShape>(1);
  auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  if (storage_shape == nullptr || tensor_size_ptr == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  return CalcUnalignedTensorSizeByShape(storage_shape->GetStorageShape(), datatype, *tensor_size_ptr);
}

REGISTER_KERNEL(CalcTensorSizeFromStorage).RunFunc(CalcTensorSizeFromStorage).TracePrinter(FromStorageTracePrinter);
REGISTER_KERNEL(CalcUnalignedTensorSizeFromStorage)
    .RunFunc(CalcUnalignedTensorSizeFromStorage)
    .TracePrinter(FromStorageTracePrinter);
}  // namespace kernel
}  // namespace gert
