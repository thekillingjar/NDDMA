/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/gert_tensor_data_faker.h"
#include "graph/utils/math_util.h"

namespace gert {
GertTensorDataFaker &GertTensorDataFaker::Shape(std::initializer_list<int64_t> shape) {
  return OriginShape(shape).StorageShape(shape);
}
GertTensorDataFaker &GertTensorDataFaker::OriginShape(std::initializer_list<int64_t> shape) {
  gtd_desc_.ss.MutableOriginShape() = shape;
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::StorageShape(std::initializer_list<int64_t> shape) {
  gtd_desc_.ss.MutableStorageShape() = shape;
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::Format(ge::Format format) {
  return OriginFormat(format).StorageFormat(format);
}
GertTensorDataFaker &GertTensorDataFaker::OriginFormat(ge::Format format) {
  gtd_desc_.sf.SetOriginFormat(format);
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::StorageFormat(ge::Format format) {
  gtd_desc_.sf.SetStorageFormat(format);
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::ExpandDimsType(gert::ExpandDimsType edt) {
  gtd_desc_.sf.SetExpandDimsType(edt);
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::DataType(ge::DataType dt) {
  gtd_desc_.dt = dt;
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::Placement(TensorPlacement placement) {
  gtd_desc_.placement = placement;
  return *this;
}
GertTensorDataFaker &GertTensorDataFaker::StreamId(int64_t stream_id) {
  gtd_desc_.stream_id = stream_id;
  return *this;
}
GtdHolder GertTensorDataFaker::Build() const {
  if (gtd_desc_.placement == kFollowing) {
    throw std::invalid_argument("Do not support following placement in GertTensorDataFaker");
  }

  auto tensor_size = ge::GetSizeInBytes(gtd_desc_.ss.GetStorageShape().GetShapeSize(), gtd_desc_.dt);
  tensor_size = static_cast<int64_t>(ge::RoundUp(tensor_size, 32) + 32);

  GtdHolder holder{gtd_desc_.ss,
                   gtd_desc_.sf,
                   gtd_desc_.dt,
                   gtd_desc_.placement,
                   gtd_desc_.stream_id,
                   memory::MultiStreamAllocatorFaker().StreamNum(gtd_desc_.stream_id + 1).Build(),
                   nullptr};
  holder.gtd = std::make_unique<GertTensorData>(tensor_size, gtd_desc_.placement, gtd_desc_.stream_id,
                                                holder.allocator_holder.at(gtd_desc_.stream_id)->Malloc(tensor_size));
  return holder;
}
}  // namespace gert
