/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_tiling_info.h"
#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "graph/def_types.h"

namespace optiling {
using std::make_shared;

namespace utils {
class OpRunInfoImpl {
public:
  OpRunInfoImpl() = default;
  ~OpRunInfoImpl() = default;

  OpRunInfoImpl(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key)
          : block_dim_(block_dim),
            clear_atomic_(clear_atomic),
            tiling_key_(tiling_key),
            addr_base_(nullptr),
            max_size_(0),
            offset_(0),
            tiling_cond_(-1),
            schedule_mode_(0U) {}

  void SetBlockDim(const uint32_t &block_dim) { block_dim_ = block_dim; }

  uint32_t GetBlockDim() const { return block_dim_; }

  void SetAicpuBlockDim(uint32_t block_dim) { aicpu_block_dim_ = block_dim; }

  uint32_t GetAicpuBlockDim() const { return aicpu_block_dim_; }

  void SetScheduleMode(const uint32_t schedule_mode) {
    schedule_mode_ = schedule_mode;
  }

  uint32_t GetScheduleMode() const {
    return schedule_mode_;
  }

  void AddWorkspace(const int64_t &workspace) { workspaces_.push_back(workspace); }

  size_t GetWorkspaceNum() const { return workspaces_.size(); }

  ge::graphStatus GetWorkspace(const size_t &idx, int64_t &workspace) const {
    if ((!workspaces_.empty()) && (idx < workspaces_.size())) {
      workspace = workspaces_[idx];
      return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
  }

  void GetAllWorkspaces(std::vector<int64_t> &workspaces) const { workspaces = workspaces_; }

  const std::vector<int64_t> &GetAllWorkspaces() const { return workspaces_; }

  void SetWorkspaces(const std::vector<int64_t> &workspaces) { workspaces_ = workspaces; }

  void AddTilingData(const char *value, const size_t size) {
    if (addr_base_ == nullptr) {
      (void)tiling_data_.write(value, static_cast<std::streamsize>(size));
      (void)tiling_data_.flush();
    } else {
      auto addr = ::ge::ValueToPtr(::ge::PtrToValue(addr_base_) + offset_);
      if (memcpy_s(addr, static_cast<size_t>(max_size_ - offset_), value, size) != EOK) {
        GELOGE(ge::GRAPH_FAILED, "[Add][TilingData] Memcpy tiling data failed, "
               "dst size = %zu, src size = %zu.", static_cast<size_t>(max_size_ - offset_), size);
        REPORT_INNER_ERR_MSG("E19999", "[Add][TilingData] Memcpy tiling data failed, dst size = %zu, src size = %zu.",
                             static_cast<size_t>(max_size_ - offset_), size);
        return;
      }
      offset_ += size;
    }
  }

  void AlignOffsetWith64() {
    const uint64_t offset = (offset_ + sizeof(uint64_t) - 1U) / sizeof(uint64_t);
    offset_ = offset * sizeof(uint64_t);
  }

  bool SetMemCheckBaseOffset(uint64_t offset) {
    GELOGD("When max size is %lu, set a new offset [%lu] to replace the original offset [%lu].", max_size_, offset, offset_);
    if (offset >= std::numeric_limits<uint64_t>::max() - sizeof(uint64_t)) {
      GELOGE(ge::GRAPH_FAILED, "Offset overflow.");
      return false;
    }
    uint64_t new_offset = (offset + sizeof(uint64_t) - 1U) / sizeof(uint64_t);
    new_offset  = new_offset * sizeof(uint64_t);
    if (new_offset < offset_ || new_offset >= max_size_) {
      return false;
    }
    offset_ = new_offset;
    return true;
  }

  void* GetAddrBase(uint64_t& max_size) const {
    max_size = max_size_;
    return addr_base_;
  }

  void SetAddrBaseOffset(const uint64_t size) {
    offset_ = size;
  }

  const ByteBuffer &GetAllTilingData() const { return tiling_data_; }

  ByteBuffer &GetAllTilingData() { return tiling_data_; }

  uint64_t GetTilingDataSize() const { return offset_; }
  void SetAllTilingData(const ByteBuffer &value) {
    tiling_data_.clear();
    offset_ = 0;
    AddTilingData(value.str().c_str(), value.str().size());
  }

  void SetClearAtomic(const bool clear_atomic) { clear_atomic_ = clear_atomic; }

  bool GetClearAtomic() const { return clear_atomic_; }

  void SetTilingKey(const uint64_t &tiling_key) { tiling_key_ = tiling_key; }

  uint64_t GetTilingKey() const { return tiling_key_; }

  void ResetWorkspace() {
    workspaces_.clear();
  }

  void ResetAddrBase(void *const addr_base, const uint64_t max_size) {
    addr_base_ = addr_base;
    max_size_ = max_size;
    offset_ = 0;
  }

  void SetTilingCond(const int32_t tiling_cond) { tiling_cond_ = tiling_cond; }

  int32_t GetTilingCond() const { return tiling_cond_; }

  void SetLocalMemorySize(const uint32_t local_memory_size) {
    local_memory_size_ = local_memory_size;
  }

  uint32_t GetLocalMemorySize() const {
    return local_memory_size_;
  }

private:
  uint32_t block_dim_;
  bool clear_atomic_;
  uint64_t tiling_key_;
  ByteBuffer tiling_data_;
  std::vector<int64_t> workspaces_;
  void *addr_base_;
  uint64_t max_size_;
  uint64_t offset_;
  int32_t tiling_cond_;
  uint32_t schedule_mode_;
  uint32_t local_memory_size_ = 0U;
  uint32_t aicpu_block_dim_ = 0U;
};

OpRunInfo::OpRunInfo() {
  impl_ = make_shared<OpRunInfoImpl>();
}

OpRunInfo::OpRunInfo(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key) {
  impl_ = make_shared<OpRunInfoImpl>(block_dim, clear_atomic, tiling_key);
}

OpRunInfo::OpRunInfo(const OpRunInfo &runinfo) {
  impl_ = make_shared<OpRunInfoImpl>(runinfo.GetBlockDim(), runinfo.GetClearAtomic(), runinfo.GetTilingKey());
  std::vector<int64_t> workspaces;
  runinfo.GetAllWorkspaces(workspaces);
  impl_->SetWorkspaces(workspaces);
  impl_->SetAllTilingData(runinfo.GetAllTilingData());
  impl_->SetLocalMemorySize(runinfo.GetLocalMemorySize());
}

OpRunInfo::OpRunInfo(OpRunInfo &&runinfo) {
  impl_ = std::move(runinfo.impl_);
}

OpRunInfo &OpRunInfo::operator=(const OpRunInfo &runinfo) {
  if (&runinfo != this) {
    impl_ = make_shared<OpRunInfoImpl>(runinfo.GetBlockDim(), runinfo.GetClearAtomic(), runinfo.GetTilingKey());
    std::vector<int64_t> workspaces;
    runinfo.GetAllWorkspaces(workspaces);
    impl_->SetWorkspaces(workspaces);
    impl_->SetAllTilingData(runinfo.GetAllTilingData());
    impl_->SetLocalMemorySize(runinfo.GetLocalMemorySize());
  }
  return *this;
}

OpRunInfo &OpRunInfo::operator=(OpRunInfo &&runinfo) {
  if (&runinfo != this) {
    impl_ = std::move(runinfo.impl_);
  }
  return *this;
}

void OpRunInfo::SetBlockDim(const uint32_t &block_dim) {
  impl_->SetBlockDim(block_dim);
}

void OpRunInfo::SetAicpuBlockDim(uint32_t block_dim) const {
  impl_->SetAicpuBlockDim(block_dim);
}

void OpRunInfo::SetScheduleMode(const uint32_t schedule_mode) {
  impl_->SetScheduleMode(schedule_mode);
}

uint32_t OpRunInfo::GetScheduleMode() const {
  return impl_->GetScheduleMode();
}

uint32_t OpRunInfo::GetBlockDim() const {
  return impl_->GetBlockDim();
}

uint32_t OpRunInfo::GetAicpuBlockDim() const {
  return impl_->GetAicpuBlockDim();
}

void OpRunInfo::AddWorkspace(const int64_t &workspace) {
  impl_->AddWorkspace(workspace);
}

size_t OpRunInfo::GetWorkspaceNum() const {
  return impl_->GetWorkspaceNum();
}

ge::graphStatus OpRunInfo::GetWorkspace(const size_t &idx, int64_t &workspace) const {
  return impl_->GetWorkspace(idx, workspace);
}

void OpRunInfo::GetAllWorkspaces(std::vector<int64_t> &workspaces) const {
  impl_->GetAllWorkspaces(workspaces);
}

const std::vector<int64_t> &OpRunInfo::GetAllWorkspaces() const {
  return impl_->GetAllWorkspaces();
}

void OpRunInfo::SetWorkspaces(const std::vector<int64_t> &workspaces) {
  impl_->SetWorkspaces(workspaces);
}

void OpRunInfo::InternelSetTiling(const ByteBuffer &value) {
  impl_->SetAllTilingData(value);
}

void OpRunInfo::AddTilingData(const ge::char_t *value, const size_t size) {
  impl_->AddTilingData(value, size);
}

void OpRunInfo::AlignOffsetWith64() {
  return impl_->AlignOffsetWith64();
}

bool OpRunInfo::SetMemCheckBaseOffset(const uint64_t &offset) {
  return impl_->SetMemCheckBaseOffset(offset);
}

void* OpRunInfo::GetAddrBase(uint64_t& max_size) const {
  return impl_->GetAddrBase(max_size);
}

void OpRunInfo::SetAddrBaseOffset(const uint64_t size) {
  impl_->SetAddrBaseOffset(size);
}

ByteBuffer &OpRunInfo::GetAllTilingData() {
  return impl_->GetAllTilingData();
}

const ByteBuffer &OpRunInfo::GetAllTilingData() const {
  return impl_->GetAllTilingData();
}
uint64_t OpRunInfo::GetTilingDataSize() const {
  return impl_->GetTilingDataSize();
}
void OpRunInfo::SetClearAtomic(const bool clear_atomic) {
  impl_->SetClearAtomic(clear_atomic);
}

bool OpRunInfo::GetClearAtomic() const {
  return impl_->GetClearAtomic();
}

void OpRunInfo::SetTilingKey(const uint64_t &new_tiling_key) {
  impl_->SetTilingKey(new_tiling_key);
}

uint64_t OpRunInfo::GetTilingKey() const {
  return impl_->GetTilingKey();
}

void OpRunInfo::ResetWorkspace() {
  impl_->ResetWorkspace();
}

void OpRunInfo::ResetAddrBase(void *const addr_base, const uint64_t max_size) {
  impl_->ResetAddrBase(addr_base, max_size);
}

void OpRunInfo::SetTilingCond(const int32_t tiling_cond) {
  impl_->SetTilingCond(tiling_cond);
}

int32_t OpRunInfo::GetTilingCond() const {
  return impl_->GetTilingCond();
}

void OpRunInfo::SetLocalMemorySize(const uint32_t local_memory_size) {
  impl_->SetLocalMemorySize(local_memory_size);
}

uint32_t OpRunInfo::GetLocalMemorySize() const {
  return impl_->GetLocalMemorySize();
}

class OpCompileInfoImpl {
public:
  OpCompileInfoImpl() : key_(), value_() {}
  ~OpCompileInfoImpl() = default;
  OpCompileInfoImpl(const ge::AscendString &key, const ge::AscendString &value) : key_(key), value_(value) {}
  OpCompileInfoImpl(const std::string &key, const std::string &value) : key_(key.c_str()), value_(value.c_str()) {}

  void SetKey(const ge::AscendString &key) { key_ = key; }

  void SetValue(const ge::AscendString &value) { value_ = value; }

  const ge::AscendString &GetKey() const { return key_; }

  const ge::AscendString &GetValue() const { return value_; }

private:
  ge::AscendString key_;
  ge::AscendString value_;
};

OpCompileInfo::OpCompileInfo() {
  impl_ = make_shared<OpCompileInfoImpl>();
}

OpCompileInfo::OpCompileInfo(const ge::AscendString &key, const ge::AscendString &value) {
  impl_ = make_shared<OpCompileInfoImpl>(key, value);
}

OpCompileInfo::OpCompileInfo(const std::string &key, const std::string &value) {
  impl_ = make_shared<OpCompileInfoImpl>(key, value);
}

OpCompileInfo::OpCompileInfo(const OpCompileInfo &compileinfo) {
  impl_ = make_shared<OpCompileInfoImpl>();
  *impl_ = *compileinfo.impl_;
}

OpCompileInfo::OpCompileInfo(OpCompileInfo &&compileinfo) {
  impl_ = std::move(compileinfo.impl_);
}

OpCompileInfo &OpCompileInfo::operator=(const OpCompileInfo &compileinfo) {
  if (&compileinfo != this) {
    impl_ = make_shared<OpCompileInfoImpl>();
    *impl_ = *compileinfo.impl_;
  }
  return *this;
}

OpCompileInfo &OpCompileInfo::operator=(OpCompileInfo &&compileinfo) {
  if (&compileinfo != this) {
    impl_ = std::move(compileinfo.impl_);
  }
  return *this;
}

void OpCompileInfo::SetKey(const ge::AscendString &key) {
  impl_->SetKey(key);
}

void OpCompileInfo::SetValue(const ge::AscendString &value) {
  impl_->SetValue(value);
}

const ge::AscendString &OpCompileInfo::GetKey() const {
  return impl_->GetKey();
}

const ge::AscendString &OpCompileInfo::GetValue() const {
  return impl_->GetValue();
}
}  // namespace utils
}  // namespace optiling
