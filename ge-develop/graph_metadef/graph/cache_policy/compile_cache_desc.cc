/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/cache_policy/compile_cache_desc.h"
#include <securec.h>
#include "common/checker.h"
#include "debug/ge_util.h"

namespace ge {
constexpr int32_t kAllShape = -2;

BinaryHolder::BinaryHolder(const BinaryHolder &other) {
  if ((other.GetDataPtr() != nullptr) && (other.GetDataLen() != 0UL)) {
    data_len_ = other.GetDataLen();
    holder_ = ComGraphMakeUnique<uint8_t[]>(data_len_);
    GE_CHECK_NOTNULL_JUST_RETURN(holder_);
    const auto mem_ret = memcpy_s(holder_.get(), data_len_,
                                  ge::PtrToPtr<const uint8_t, const void>(other.GetDataPtr()), data_len_);
    if (mem_ret != EOK) {
      data_len_ = 0U;
      holder_ = nullptr;
      GELOGE(ge::GRAPH_FAILED, "[BinaryHolder] memcpy failed.");
    }
  }
}

BinaryHolder::BinaryHolder(const uint8_t *const data, const size_t data_len) {
  if ((data != nullptr) && (data_len != 0UL)) {
    data_len_ = data_len;
    holder_ = ComGraphMakeUnique<uint8_t[]>(data_len_);
    GE_CHECK_NOTNULL_JUST_RETURN(holder_);
    const auto mem_ret = memcpy_s(holder_.get(), data_len_,
                                  ge::PtrToPtr<const uint8_t, const void>(data), data_len_);
    if (mem_ret != EOK) {
      data_len_ = 0U;
      holder_ = nullptr;
      GELOGE(ge::GRAPH_FAILED, "[BinaryHolder] memcpy failed.");
    }
  }
}

BinaryHolder &BinaryHolder::operator=(const BinaryHolder &other) {
  if ((other.GetDataPtr() != nullptr) && (other.GetDataLen() != 0UL)) {
    data_len_ = other.GetDataLen();
    holder_ = ComGraphMakeUnique<uint8_t[]>(data_len_);
    if (holder_ == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "[BinaryHolder] make unique failed.");
      return *this;
    }
    const auto mem_ret = memcpy_s(holder_.get(), data_len_,
                                  ge::PtrToPtr<const uint8_t, const void>(other.GetDataPtr()), data_len_);
    if (mem_ret != EOK) {
      data_len_ = 0U;
      holder_ = nullptr;
      GELOGE(ge::GRAPH_FAILED, "[BinaryHolder] memcpy failed.");
    }
  }
  return *this;
}

BinaryHolder::BinaryHolder(BinaryHolder &&other) {
  data_len_ = other.data_len_;
  holder_ = std::move(other.holder_);
  other.data_len_ = 0;
}

BinaryHolder &BinaryHolder::operator=(BinaryHolder &&other) {
  data_len_ = other.data_len_;
  holder_ = std::move(other.holder_);
  other.data_len_ = 0;
  return *this;
}

std::unique_ptr<BinaryHolder> BinaryHolder::createFrom(std::unique_ptr<uint8_t[]> &&ptr, size_t length) {
  auto holder = ComGraphMakeUnique<BinaryHolder>();
  if ((ptr != nullptr) && (holder != nullptr) && (length != 0UL)) {
    holder->data_len_ = length;
    holder->holder_ = std::move(ptr);
  }
  return holder;
}

const uint8_t *BinaryHolder::GetDataPtr() const noexcept {
  if (holder_.get() != nullptr) {
    return holder_.get();
  }
  return nullptr;
}

const size_t &BinaryHolder::GetDataLen() const noexcept {
  return data_len_;
}

bool BinaryHolder::operator!=(const BinaryHolder &second) const {
  if (this->GetDataLen() != second.GetDataLen()) {
    return true;
  }
  const auto this_data = this->GetDataPtr();
  const auto second_data = second.GetDataPtr();
  if (((this_data == nullptr) && (second_data != nullptr)) ||
      ((this_data != nullptr) && (second_data == nullptr))) {
    return true;
  }
  if ((this_data == nullptr) && (second_data == nullptr)) {
    return false;
  }
  if (memcmp(this_data, second_data, this->GetDataLen()) != 0) {
    return true;
  }
  return false;
}

Format TensorInfoArgs::GetFormat() const {
  return format_;
}

Format TensorInfoArgs::GetOriginFormat() const {
  return origin_format_;
}

DataType TensorInfoArgs::GetDataType() const {
  return data_type_;
}

void TensorInfoArgs::SetShape(const std::vector<int64_t> &shape) {
  shape_.clear();
  for (const auto dim : shape) {
    shape_.emplace_back(dim);
  }
}

void TensorInfoArgs::SetShape(const SmallVector<int64_t, kDefaultDimsNum> &shape) {
  shape_.clear();
  shape_ = shape;
}

void TensorInfoArgs::SetOriginShape(const std::vector<int64_t> &origin_shape) {
  origin_shape_.clear();
  for (const auto dim : origin_shape) {
    origin_shape_.emplace_back(dim);
  }
}

void TensorInfoArgs::SetOriginShape(const SmallVector<int64_t, kDefaultDimsNum> &origin_shape) {
  origin_shape_.clear();
  origin_shape_ = origin_shape;
}

void TensorInfoArgs::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  shape_range_.clear();
  for (const auto &range : ranges) {
    shape_range_.emplace_back(range);
  }
}

bool TensorInfoArgs::IsUnknownShape() const {
  return std::any_of(shape_.begin(), shape_.end(), [](const int64_t &dim) {
      return (dim == UNKNOWN_DIM) || (dim == UNKNOWN_DIM_NUM);
      });
}

bool TensorInfoArgs::operator!=(const TensorInfoArgs &second) const {
  const bool ret = (this->format_ != second.format_) || (this->origin_format_ != second.origin_format_) ||
      (this->data_type_ != second.data_type_) || (this->shape_ != second.shape_) ||
          (this->origin_shape_ != second.origin_shape_) || (this->shape_range_ != second.shape_range_);
  return ret;
}

bool TensorInfoArgs::IsTensorInfoMatch(const TensorInfoArgs &other) const {
  const bool is_same = (this->format_ == other.format_) && (this->origin_format_ == other.origin_format_) &&
      (this->data_type_ == other.data_type_);
  if (!is_same) {
    GELOGD("format or origin format or datatype is not matched");
    return false;
  }
  return IsShapeInRange(other);
}

bool TensorInfoArgs::IsShapeInRange(const TensorInfoArgs &other) const {
  if ((this->shape_.size() == 1U) && (this->shape_[0] == kAllShape)) {
    // -2 is all shape, need to judge first
    GELOGD("current shape is -2");
    return true;
  }
  // check rank
  const bool is_same_rank = (this->shape_.size() == other.shape_.size()) &&
      (this->origin_shape_.size() == other.origin_shape_.size());
  if (!is_same_rank) {
    GELOGD("shape or origin shape is not same rank");
    return false;
  }
  // check shape range when shape is dynamic
  if (this->IsUnknownShape()) {
    if (this->shape_.size() != this->shape_range_.size()) {
      GELOGD("shape size %zu is not match shape rang size %zu", this->shape_.size(), this->shape_range_.size());
      return false;
    }
    for (size_t i = 0U; i < this->shape_range_.size(); ++i) {
      if (this->shape_range_[i].first > other.shape_[i]) {
        GELOGD("shape range is not match, first is %" PRId64 ", other is %" PRId64 ", index is %zu",
            this->shape_range_[i].first, other.shape_[i], i);
        return false;
      }
      // -1 means infinity great
      if (this->shape_range_[i].second == UNKNOWN_DIM) {
        GELOGD("shape second is -1, index is %zu", i);
        continue;
      }
      if (this->shape_range_[i].second < other.shape_[i]) {
        GELOGD("shape range is not match, second is %" PRId64 ", other is %" PRId64 ", index is %zu",
            this->shape_range_[i].second, other.shape_[i], i);
        return false;
      }
    }
  } else {
    GELOGD("this is exact shape");
    if ((this->shape_ != other.shape_) || (this->origin_shape_ != other.origin_shape_)) {
      GELOGD("exact shape or origin shape is not matched");
      return false;
    }
  }
  return true;
}

size_t CompileCacheDesc::GetTensorInfoSize() {
  return tensor_info_args_vec_.size();
}

TensorInfoArgs *CompileCacheDesc::MutableTensorInfo(size_t index) {
  if (index >= tensor_info_args_vec_.size()) {
    return nullptr;
  }
  return &tensor_info_args_vec_[index];
}

void CompileCacheDesc::AddBinary(const BinaryHolder &holder) {
  other_desc_.emplace_back(holder);
}

void CompileCacheDesc::AddBinary(BinaryHolder &&holder) {
  other_desc_.emplace_back(holder);
}

void CompileCacheDesc::SetOpType(const std::string &op_type) {
  op_type_ = op_type;
  return;
}

void CompileCacheDesc::AddTensorInfo(const TensorInfoArgs &tensor_info) {
  tensor_info_args_vec_.emplace_back(tensor_info);
  return;
}

void CompileCacheDesc::SetScopeId(const std::initializer_list<uint64_t> scope_id) {
  scope_id_= scope_id;
  return;
}

bool CompileCacheDesc::CheckWithoutTensorInfo(const CompileCacheDesc *first, const CompileCacheDesc *second) const {
  if ((first->op_type_ != second->op_type_) ||
      (first->tensor_info_args_vec_.size() != second->tensor_info_args_vec_.size())) {
    GELOGD("op_type_ %s, %s is not match or size %zu %zu is not match",
           first->op_type_.c_str(), second->op_type_.c_str(),
           first->tensor_info_args_vec_.size(), second->tensor_info_args_vec_.size());
    return false;
  }
  if (first->scope_id_ != second->scope_id_) {
    GELOGD("scope id is not match");
    return false;
  }
  if (first->other_desc_.size() != second->other_desc_.size()) {
    GELOGD("other_desc_ size %zu, %zu is not match ", first->other_desc_.size(), second->other_desc_.size());
    return false;
  }
  for (size_t i = 0U; i < first->other_desc_.size(); ++i) {
    if (first->other_desc_[i].GetDataLen() != second->other_desc_[i].GetDataLen()) {
      GELOGD("other_desc_ mem size %zu, %zu is not match ",
          first->other_desc_[i].GetDataLen(), second->other_desc_[i].GetDataLen());
      return false;
    }
    if ((first->other_desc_[i].GetDataPtr() == nullptr) || (second->other_desc_[i].GetDataPtr() == nullptr)) {
      return false;
    }
    const auto cmp_ret = memcmp(first->other_desc_[i].GetDataPtr(),
        second->other_desc_[i].GetDataPtr(), second->other_desc_[i].GetDataLen());
    if (cmp_ret != 0) {
      GELOGD("mem compare fail");
      return false;
    }
  }
  return true;
}

bool CompileCacheDesc::IsMatch(const CacheDescPtr &other) const {
  const auto *second = dynamic_cast<const CompileCacheDesc *>(other.get());
  GE_ASSERT_NOTNULL(second, "dynamic cast failed");
  if (!CheckWithoutTensorInfo(this, second)) {
    return false;
  }

  for (size_t i = 0U; i < this->tensor_info_args_vec_.size(); ++i) {
    const auto &first_args = this->tensor_info_args_vec_[i];
    const auto &second_args = second->tensor_info_args_vec_[i];
    if (!first_args.IsTensorInfoMatch(second_args)) {
      GELOGD("shape is not matched");
      return false;
    }
  }
  return true;
}

bool CompileCacheDesc::IsEqual(const CacheDescPtr &other) const {
  const auto *second = dynamic_cast<const CompileCacheDesc *>(other.get());
  GE_ASSERT_NOTNULL(second, "dynamic cast failed");
  if (!CheckWithoutTensorInfo(this, second)) {
    return false;
  }

  for (size_t i = 0U; i < this->tensor_info_args_vec_.size(); ++i) {
    const auto &first_args = this->tensor_info_args_vec_[i];
    const auto &second_args = second->tensor_info_args_vec_[i];
    if (first_args != second_args) {
      GELOGD("tensor info is not matched");
      return false;
    }
  }
  return true;
}

CacheHashKey CompileCacheDesc::GetCacheDescHash() const {
  CacheHashKey hash_key = 0U;
  for (const auto &arg : tensor_info_args_vec_) {
    hash_key = HashUtils::MultiHash(hash_key, arg.GetFormat(), arg.GetOriginFormat(), arg.GetDataType());
  }
  hash_key = HashUtils::MultiHash(op_type_, hash_key);
  return hash_key;
}
}  // namespace ge
