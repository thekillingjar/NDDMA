/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_v2_utils.h"
#include <sstream>

namespace ge {
namespace hybrid {
std::string DebugString(const gert::Shape &shape) {
  if (shape.IsScalar()) {
    return "()";
  }
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0U; i < shape.GetDimNum(); i++) {
    ss << shape.GetDim(i) << ((i + 1) == shape.GetDimNum() ? ")" : ",");
  }
  return ss.str();
}

std::string DebugString(const gert::TensorPlacement &placement) {
  switch (placement) {
    case gert::kOnDeviceHbm:
      return "kOnDeviceHbm";
    case gert::kOnHost:
      return "kOnHost";
    case gert::kFollowing:
      return "kFollowing";
    case gert::kOnDeviceP2p:
      return "kOnDeviceP2p";
    default:
      return "invalid";
  }
}

std::string DebugString(const ge::Placement &placement) {
  switch (placement) {
    case ge::Placement::kPlacementHost:
      return "kPlacementHost";
    case ge::Placement::kPlacementDevice:
      return "kPlacementDevice";
    default:
      return "invalid";
  }
}

std::string DebugString(const gert::Tensor &tensor, bool show_rt_attr) {
  std::stringstream ss;
  ss << "type: " << ge::TypeUtils::DataTypeToSerialString(tensor.GetDataType()) << " ";
  ss << "origin format: " << ge::TypeUtils::FormatToSerialString(tensor.GetOriginFormat()) << " ";
  ss << "storage format: " << ge::TypeUtils::FormatToSerialString(tensor.GetStorageFormat()) << " ";
  if (show_rt_attr) {
    ss << "origin shape: " << DebugString(tensor.GetOriginShape()) << " ";
    ss << "storage shape: " << DebugString(tensor.GetStorageShape()) << " ";
    ss << "placement: " << DebugString(tensor.GetPlacement()) << " ";
    ss << "data ptr: " << ge::PtrToValue(tensor.GetAddr());
  }
  return ss.str();
}

void DimsAsShape(const std::vector<int64_t> &dims, gert::Shape &shape) {
  shape.SetDimNum(dims.size());
  for (size_t i = 0U; i < dims.size(); i++) {
    shape.SetDim(i, dims[i]);
  }
}

void SmallVecDimsAsRtShape(const SmallVector<int64_t, kDefaultDimsNum> &dims, gert::Shape &shape) {
  shape.SetDimNum(dims.size());
  for (size_t i = 0U; i < dims.size(); i++) {
    shape.SetDim(i, dims[i]);
  }
}

void RtShapeAsGeShape(const gert::Shape &shape, ge::GeShape &ge_shape) {
  ge_shape.SetDimNum(shape.GetDimNum());
  for (size_t i = 0U; i < ge_shape.GetMutableDims().size(); i++) {
    ge_shape.SetDim(i, shape.GetDim(i));
  }
}

void GeShapeAsRtShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) {
  gert_shape.SetDimNum(ge_shape.GetDims().size());
  for (size_t i = 0U; i < ge_shape.GetMutableDims().size(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
}

std::string DebugString(const GeShape &shape) {
  if (shape.IsScalar()) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[";
  size_t dim_index = 0U;
  for (; dim_index < shape.GetDims().size() - 1U; dim_index++) {
    ss << shape.GetDim(dim_index) << ",";
  }
  ss << shape.GetDim(dim_index) << "]";
  return ss.str();
}

std::string DebugString(const GeTensorDesc &desc) {
  std::stringstream ss;
  ss << "type: " << TypeUtils::DataTypeToSerialString(desc.GetDataType()) << ", ";
  ss << "format: " << TypeUtils::FormatToSerialString(desc.GetFormat()) << ", ";
  ss << "storage shape: " << DebugString(desc.GetShape()) << ", ";
  ss << "origin shape: " << DebugString(desc.GetOriginShape());
  return ss.str();
}
}  // namespace hybrid
}  // namespace ge
