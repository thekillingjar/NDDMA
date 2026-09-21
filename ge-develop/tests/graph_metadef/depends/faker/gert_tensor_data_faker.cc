/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/gert_tensor_data.h"
#include "common/checker.h"
#include "allocator_faker.h"

namespace gert {
GertTensorData::GertTensorData(size_t size, TensorPlacement placement, int64_t stream_id, GertMemBlock *block)
    : tensor_data_{}, gert_mem_block_{block}, stream_id_{stream_id} {
  tensor_data_.SetSize(size);
  tensor_data_.SetPlacement(placement);
  if (gert_mem_block_ == nullptr) {
    return;
  }
  tensor_data_.SetAddr(gert_mem_block_->GetAddr(), nullptr);
}
GertTensorData::GertTensorData(void *addr, size_t size, TensorPlacement placement, int64_t stream_id)
    : tensor_data_(addr, nullptr, size, placement), gert_mem_block_(nullptr), stream_id_(stream_id) {}

bool GertTensorData::IsSharedWith(const GertTensorData &other) const {
  return (tensor_data_.IsSharedWith(other.tensor_data_) && gert_mem_block_ == other.gert_mem_block_ &&
          stream_id_ == other.stream_id_);
}

ge::graphStatus GertTensorData::ShareFrom(const GertTensorData &other) {
  if (IsSharedWith(other)) {
    return ge::GRAPH_SUCCESS;
  }
  // multi stream
  CopyFromWithoutStream(other);
  stream_id_ = other.stream_id_;
  if (gert_mem_block_ != nullptr) {
    reinterpret_cast<GertMemBlockFaker *>(gert_mem_block_)->AddCount();
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus GertTensorData::WanderFrom(const GertTensorData &other, int64_t dst_stream_id) {
  return ge::GRAPH_FAILED;
}
void GertTensorData::CopyFromWithoutStream(const GertTensorData &other) {
  if (NeedFree()) {
    Free();
  }
  tensor_data_.SetAddr(other.tensor_data_.GetAddr(), nullptr);
  SetPlacement(other.GetPlacement());
  SetSize(other.GetSize());
  gert_mem_block_ = other.gert_mem_block_;
}
}  // namespace gert
