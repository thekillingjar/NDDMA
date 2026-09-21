/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include "base/err_msg.h"
#include "common/preload/elf/elf_data.h"
#include "graph/utils/math_util.h"
#include "common/tbe_handle_store/kernel_store.h"

namespace {
constexpr uint32_t kKernelItemMagic = 0x5D776EFDU;
constexpr size_t kAlignBy4B = 4U;
}

namespace ge {
void KernelStore::AddKernel(const KernelBinPtr &kernel) {
  if (kernel != nullptr) {
    kernels_[kernel->GetName()] = kernel;
  }
}

bool KernelStore::Build() {
  buffer_.clear();
  size_t total_len = 0U;
  for (const auto &item : kernels_) {
    const auto kernel = item.second;
    total_len += sizeof(KernelStoreItemHead);
    total_len += kernel->GetName().length();
    total_len += kernel->GetBinDataSize();
  }

  try {
    buffer_.resize(total_len);
  } catch (std::bad_alloc &e) {
    GELOGE(ge::MEMALLOC_FAILED, "All build memory failed, memory size %zu", total_len);
    GELOGE(ge::MEMALLOC_FAILED, "[Malloc][Memmory]Resize buffer failed, memory size %zu, "
           "exception %s", total_len, e.what());
    REPORT_INNER_ERR_MSG("E19999", "Resize buffer failed, memory size %zu, exception %s",
                      total_len, e.what());
    return false;
  }

  uint8_t *next_buffer = buffer_.data();
  size_t remain_len = total_len;
  for (const auto &item : kernels_) {
    const auto kernel = item.second;
    KernelStoreItemHead kernel_head{};
    kernel_head.magic = kKernelItemMagic;
    kernel_head.name_len = static_cast<uint32_t>(kernel->GetName().length());
    kernel_head.bin_len = static_cast<uint32_t>(kernel->GetBinDataSize());

    GELOGD("get kernel bin name %s, addr %p, size %zu, remain_len %zu", kernel->GetName().c_str(), kernel->GetBinData(),
           kernel->GetBinDataSize(), remain_len);
    GE_ASSERT_SUCCESS(GeMemcpy(next_buffer, remain_len, PtrToPtr<KernelStoreItemHead, uint8_t>(&kernel_head), sizeof(kernel_head)));

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(kernel_head)));
    GE_ASSERT_SUCCESS(GeMemcpy(next_buffer, remain_len - sizeof(kernel_head),
                               PtrToPtr<char_t, uint8_t>(kernel->GetName().data()),
                               static_cast<size_t>(kernel_head.name_len)));

    next_buffer =
        PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + static_cast<size_t>(kernel_head.name_len)));
    GE_ASSERT_SUCCESS(GeMemcpy(next_buffer, remain_len - sizeof(kernel_head) - kernel_head.name_len, kernel->GetBinData(),
                               static_cast<size_t>(kernel_head.bin_len)));

    next_buffer =
        PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + static_cast<size_t>(kernel_head.bin_len)));
    remain_len = remain_len - sizeof(kernel_head) - kernel_head.name_len - kernel_head.bin_len;
  }
  return true;
}

bool KernelStore::PreBuild() {
  GELOGD("PreBuild set soc version and context");
  pre_buffer_.clear();
  size_t buff_size = 0U;
  for (const auto &item : kernels_) {
    const auto kernel = item.second;
    buff_size += kernel->GetBinDataSize();
    ALIGN_MEM(buff_size, kAlignBy4B);
  }

  pre_buffer_.resize(buff_size);

  uint8_t * const buffer = pre_buffer_.data();
  uint32_t buff_offset = 0U;
  size_t remain_len = buff_size;
  for (const auto &item : kernels_) {
    const auto kernel = item.second;
    const auto bin_len = kernel->GetBinDataSize();

    const auto next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(buffer) + static_cast<size_t>(buff_offset)));
    // copy bin data to buff
    const errno_t mem_ret = memcpy_s(next_buffer, remain_len, kernel->GetBinData(), bin_len);
    GE_ASSERT_EOK(mem_ret, "[PreBuild][memcpy] failed");
    // get bin data offset in kernel
    uint32_t bin_data_offset = 0U;
    GE_ASSERT_SUCCESS(Elf::GetElfOffset(const_cast<uint8_t *>(kernel->GetBinData()), static_cast<uint32_t>(bin_len),
        bin_data_offset), "[PreBuild][GetElfOffset] failed get bin data offset");

    // calc kernel bin data offset in buff
    const uint32_t total_offset = buff_offset + bin_data_offset;
    pre_buffer_offset_[kernel->GetName()] = total_offset;
    GELOGD("kernel_name:%s, bin_len:%zu, total_offset:%u, kernel offset:%u, bin_data_offset:%u.",
           kernel->GetName().c_str(), bin_len, total_offset, buff_offset, bin_data_offset);
    buff_offset += static_cast<uint32_t>(bin_len);
    remain_len -= bin_len;
  }
  kernels_.clear();
  return true;
}

const uint8_t *KernelStore::Data() const {
  return buffer_.data();
}

size_t KernelStore::DataSize() const {
  return buffer_.size();
}

const uint8_t *KernelStore::PreData() const {
  return pre_buffer_.data();
}

size_t KernelStore::PreDataSize() const {
  return pre_buffer_.size();
}

bool KernelStore::Load(const uint8_t *const data, const size_t len) {
  if ((data == nullptr) || (len == 0U)) {
    return false;
  }
  size_t buffer_len = len;
  while (buffer_len > sizeof(KernelStoreItemHead)) {
    const uint8_t *next_buffer =
        PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(len - buffer_len)));

    const auto *const kernel_head = PtrToPtr<const uint8_t, const KernelStoreItemHead>(next_buffer);
    if (buffer_len < (static_cast<size_t>(kernel_head->name_len) + static_cast<size_t>(kernel_head->bin_len) +
                      sizeof(KernelStoreItemHead))) {
      GELOGW("Invalid kernel block remain buffer len %zu, name len %u, bin len %u", buffer_len, kernel_head->name_len,
             kernel_head->bin_len);
      break;
    }

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(KernelStoreItemHead)));
    std::string name(PtrToPtr<const uint8_t, const char_t>(next_buffer), static_cast<size_t>(kernel_head->name_len));

    next_buffer =
        PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + static_cast<size_t>(kernel_head->name_len)));
    GELOGD("Load kernel from om:%s,%u,%u", name.c_str(), kernel_head->name_len, kernel_head->bin_len);
    std::vector<char> kernel_bin(
        next_buffer,
        PtrToPtr<void, const uint8_t>(ValueToPtr(PtrToValue(next_buffer) + static_cast<size_t>(kernel_head->bin_len))));
    KernelBinPtr teb_kernel_ptr = ge::MakeShared<KernelBin>(name, std::move(kernel_bin));
    if (teb_kernel_ptr != nullptr) {
      (void)kernels_.emplace(name, teb_kernel_ptr);
    }
    buffer_len -= sizeof(KernelStoreItemHead) + kernel_head->name_len + kernel_head->bin_len;
  }

  return true;
}

KernelBinPtr KernelStore::FindKernel(const std::string &name) const {
  const auto it = kernels_.find(name);
  if (it != kernels_.end()) {
    return it->second;
  }
  return nullptr;
}

bool KernelStore::IsEmpty() const {
  return kernels_.empty() && buffer_.empty();
}
}  // namespace ge
