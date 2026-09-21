/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_so_store.h"
#include "common/checker.h"

namespace {
constexpr uint32_t kKernelItemMagic = 0x5D776EFDU;
}

namespace ge {
void OpSoStore::AddKernel(const OpSoBinPtr &so_bin_ptr) {
  if (so_bin_ptr != nullptr) {
    kernels_.push_back(so_bin_ptr);
  }

  return;
}

std::vector<OpSoBinPtr> OpSoStore::GetSoBin() const { return kernels_; }

const uint8_t *OpSoStore::Data() const { return buffer_.get(); }

size_t OpSoStore::DataSize() const { return static_cast<size_t>(buffer_size_); }

uint32_t OpSoStore::GetKernelNum() const {
  return static_cast<uint32_t>(kernels_.size());
};

bool OpSoStore::CalculateAndAllocMem() {
  size_t total_len = sizeof(SoStoreHead);
  for (const auto &item : kernels_) {
    total_len += sizeof(SoStoreItemHead);
    total_len += item->GetSoName().length();
    total_len += item->GetVendorName().length();
    total_len += item->GetBinDataSize();
  }
  buffer_size_ = static_cast<uint32_t>(total_len);
  buffer_ = std::shared_ptr<uint8_t> (new (std::nothrow) uint8_t[total_len],
      std::default_delete<uint8_t[]>());
  GE_ASSERT_NOTNULL(buffer_, "[Malloc][Memmory]Resize buffer failed, memory size %zu", total_len);
  return true;
}

bool OpSoStore::Build() {
  GE_ASSERT_TRUE(CalculateAndAllocMem());

  size_t remain_len = buffer_size_;

  auto next_buffer = buffer_.get();
  SoStoreHead so_store_head;
  so_store_head.so_num = static_cast<uint32_t>(kernels_.size());
  GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len, &so_store_head, sizeof(SoStoreHead)));
  remain_len -= sizeof(SoStoreHead);
  next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(SoStoreHead)));
  for (const auto &item : kernels_) {
    SoStoreItemHead so_bin_head{};
    so_bin_head.magic = kKernelItemMagic;
    so_bin_head.so_name_len = static_cast<uint16_t>(item->GetSoName().length());
    so_bin_head.so_bin_type = static_cast<uint16_t>(item->GetSoBinType());
    so_bin_head.vendor_name_len = static_cast<uint32_t>(item->GetVendorName().length());
    so_bin_head.bin_len = static_cast<uint32_t>(item->GetBinDataSize());

    GELOGD("get so name %s, vendor name:%s, size %zu",
           item->GetSoName().c_str(), item->GetVendorName().c_str(), item->GetBinDataSize());

    GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len, &so_bin_head, sizeof(SoStoreItemHead)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(SoStoreItemHead)));

    GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len - sizeof(SoStoreItemHead),
        item->GetSoName().data(), static_cast<size_t>(so_bin_head.so_name_len)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head.so_name_len));

    GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len - sizeof(SoStoreItemHead) - so_bin_head.so_name_len,
        item->GetVendorName().data(), static_cast<size_t>(so_bin_head.vendor_name_len)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head.vendor_name_len));

    GE_ASSERT_EOK(memcpy_s(next_buffer,
        remain_len - sizeof(SoStoreItemHead) - so_bin_head.so_name_len - so_bin_head.vendor_name_len,
        item->GetBinData(), static_cast<size_t>(so_bin_head.bin_len)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) +
        static_cast<size_t>(so_bin_head.bin_len)));

    remain_len = remain_len - sizeof(SoStoreItemHead) - so_bin_head.so_name_len -
        so_bin_head.vendor_name_len - so_bin_head.bin_len;
  }
  GELOGI("[OpSoBins][Build]So bin num:%zu.", kernels_.size());
  return true;
}

bool OpSoStore::Load(const uint8_t *const data, const size_t len) {
  if ((data == nullptr) || (len == 0U)) {
    return false;
  }

  size_t buffer_len = len;
  if (buffer_len > sizeof(SoStoreHead)) {
    SoStoreHead so_store_head;
    GE_ASSERT_EOK(memcpy_s(&so_store_head, sizeof(SoStoreHead), data, sizeof(SoStoreHead)));
    so_num_ = so_store_head.so_num;
    buffer_len -= sizeof(SoStoreHead);
  }
  while (buffer_len > sizeof(SoStoreItemHead)) {
    const uint8_t *next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(data) + (len - buffer_len)));

    const auto *const so_bin_head = PtrToPtr<const uint8_t, const SoStoreItemHead>(next_buffer);
    if (buffer_len < (static_cast<size_t>(so_bin_head->so_name_len) + static_cast<size_t>(so_bin_head->bin_len) +
        static_cast<size_t>(so_bin_head->vendor_name_len) + sizeof(SoStoreItemHead))) {
      GELOGW("Invalid so block remain buffer len %zu, so name len %u, vendor name len:%u, bin len %u, so_bin_type %u",
             buffer_len, so_bin_head->so_name_len, so_bin_head->vendor_name_len, so_bin_head->bin_len,
             so_bin_head->so_bin_type);
      break;
    }

    GELOGI("Get so block remain buffer len %zu, so name len %u, vendor name len:%u, bin len %u, so_bin_type %u",
           buffer_len, so_bin_head->so_name_len, so_bin_head->vendor_name_len, so_bin_head->bin_len,
           so_bin_head->so_bin_type);
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(SoStoreItemHead)));
    std::string so_name(PtrToPtr<const uint8_t, const char_t>(next_buffer),
        static_cast<size_t>(so_bin_head->so_name_len));

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head->so_name_len));
    std::string vendor_name(PtrToPtr<const uint8_t, const char_t>(next_buffer),
        static_cast<size_t>(so_bin_head->vendor_name_len));

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head->vendor_name_len));
    GELOGD("Load vendor:%s so:%s from om, bin len:%u", vendor_name.c_str(), so_name.c_str(), so_bin_head->bin_len);

    GE_ASSERT_TRUE((so_bin_head->bin_len != 0U), "bin len from SoStoreItemHead is %u", so_bin_head->bin_len);
    auto so_bin = MakeUnique<char[]>(static_cast<size_t>(so_bin_head->bin_len));
    GE_ASSERT_NOTNULL(so_bin);
    GE_ASSERT_EOK(memcpy_s(so_bin.get(), static_cast<size_t>(so_bin_head->bin_len), next_buffer,
                           static_cast<size_t>(so_bin_head->bin_len)));

    const OpSoBinPtr so_bin_ptr = ge::MakeShared<OpSoBin>(so_name, vendor_name, std::move(so_bin), so_bin_head->bin_len,
                                                          static_cast<SoBinType>(so_bin_head->so_bin_type));
    if (so_bin_ptr != nullptr) {
      (void) kernels_.push_back(so_bin_ptr);
    }
    buffer_len -= sizeof(SoStoreItemHead) + so_bin_head->so_name_len + so_bin_head->vendor_name_len +
        so_bin_head->bin_len;
  }
  GE_ASSERT_TRUE((so_num_ == kernels_.size()), "read so num:%zu so bin num:%u from om are not equal", so_num_,
                 kernels_.size());
  GELOGI("[OpSoBins][Load]So num:%zu so bin num:%u from om", so_num_, kernels_.size());
  return true;
}
}  // namespace ge
