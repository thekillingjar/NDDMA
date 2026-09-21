/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/common/tensor_value.h"
#include <sstream>
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "hybrid/common/npu_memory_allocator.h"

namespace ge {
namespace hybrid {
TensorBuffer::TensorBuffer(NpuMemoryAllocator *const allocator, void *const buffer, const size_t size,
                           const MemStorageType mem_type)
    : allocator_(allocator), buffer_(buffer), size_(size), mem_type_(mem_type) {}

std::unique_ptr<TensorBuffer> TensorBuffer::Create(NpuMemoryAllocator * const allocator, const size_t size,
                                                   const AllocationAttr * const attr) {
  void *buffer = nullptr;
  if (size == 0U) {
    GELOGD("size is 0");
    return Create(buffer, 0U);
  }

  if (allocator == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Check][Param:NpuMemoryAllocator] allocator is NULL.");
    REPORT_INNER_ERR_MSG("E19999", "input allocator is NULL.");
    return nullptr;
  }

  MemStorageType mem_type = MemStorageType::HBM;
  if (attr != nullptr) {
    mem_type = attr->GetMemType();
  }
  buffer = allocator->Allocate(size, attr);
  if (buffer == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Allocate][Memory] Failed. size = %zu.", size);
    REPORT_INNER_ERR_MSG("E19999", "allocate failed, size = %zu.", size);
    return nullptr;
  }

  GELOGD("Tensor created. addr = %p, size = %zu, mem_type = %d", buffer, size, static_cast<int32_t>(mem_type));
  auto addr = MakeUnique<TensorBuffer>(allocator, buffer, size, mem_type);
  if (addr == nullptr) {
    allocator->Deallocate(buffer, mem_type);
  }
  return addr;
}

std::unique_ptr<TensorBuffer> TensorBuffer::Create(void *const buffer, const size_t size) {
  GELOGD("Tensor created. addr = %p, size = %zu", buffer, size);
  return MakeUnique<TensorBuffer>(nullptr, buffer, size);
}

TensorBuffer::~TensorBuffer() {
  if (allocator_ != nullptr) {
    allocator_->Deallocate(buffer_, mem_type_);
    buffer_ = nullptr;
  }
}

TensorValue::TensorValue(const std::shared_ptr<TensorBuffer> buffer) : buffer_(std::move(buffer)) {
  if (buffer_ != nullptr) {
    mem_type_ = buffer_->GetMemType();
  }
}

TensorValue::TensorValue(void *const buffer, const size_t size, const MemStorageType mem_type) : ref_buffer_(buffer),
    ref_size_(size), mem_type_(mem_type) {
}

TensorValue::~TensorValue() { Destroy(); }

void TensorValue::Destroy() {
  if (buffer_ != nullptr) {
    GELOGD("Unref tensor: %s", DebugString().c_str());
    buffer_.reset();
  }
  ref_buffer_ = nullptr;
  ref_size_ = 0U;
}

size_t TensorValue::GetSize() const {
  if (ref_buffer_ != nullptr) {
    return ref_size_;
  }

  if (buffer_ == nullptr) {
    GELOGD("TensorValue[%s] is empty", name_.c_str());
    return 0U;
  }

  return buffer_->GetSize();
}

const void *TensorValue::GetData() const {
  if (ref_buffer_ != nullptr) {
    return ref_buffer_;
  }

  if (buffer_ == nullptr) {
    GELOGD("TensorValue[%s] is empty", name_.c_str());
    return nullptr;
  }
  return buffer_->GetData();
}

void *TensorValue::MutableData() {
  if (ref_buffer_ != nullptr) {
    return ref_buffer_;
  }

  if (buffer_ == nullptr) {
    GELOGD("TensorValue[%s] is empty", name_.c_str());
    return nullptr;
  }

  return buffer_->GetData();
}

std::string TensorValue::DebugString() const {
  std::stringstream ss;
  ss << "TensorValue[";
  if (name_.empty()) {
    ss << "unnamed] ";
  } else {
    ss << name_ << "] ";
  }

  if (ref_buffer_ != nullptr) {
    ss << "ref_addr = " << ref_buffer_ << ", size = " << ref_size_;
  } else if (buffer_ != nullptr) {
    ss << "addr = " << buffer_->GetData() << ", size = " << buffer_->GetSize();
  } else {
    ss << "addr = (nil)";
  }

  return ss.str();
}
}  // namespace hybrid
}  // namespace ge
