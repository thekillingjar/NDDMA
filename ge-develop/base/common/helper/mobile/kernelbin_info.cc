/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernelbin_info.h"

#include <climits>

#include "common/checker.h"

namespace {

constexpr uint32_t MEM_ALIGN_SIZE = 256;

struct Tlv {
    uint32_t type;
    uint32_t len;
};

} // namespace

namespace ge {
namespace mobile {

void KernelBinManager::AddKernelBin(const KernelBin& kernelbin)
{
    kernelbin_list_.push_back(kernelbin);
}

uint32_t KernelBinManager::GetBinSectionSize()
{
    uint32_t size = 0;
    uint32_t kernel_binary_total_size = 0;
    for (auto kernelbin: kernelbin_list_) {
        GE_ASSERT_TRUE(kernelbin.stub_name.size() <= UINT32_MAX, "[Mobile] overflow, failed.");
        const uint32_t charLen = static_cast<uint32_t>(kernelbin.stub_name.size()) + 1U;
        GE_ASSERT_TRUE((sizeof(Tlv) <= UINT32_MAX) && (sizeof(KernelInfo) <= UINT32_MAX), "[Mobile] overflow, failed.");
        size += static_cast<uint32_t>(sizeof(Tlv)) + static_cast<uint32_t>(sizeof(KernelInfo)) + charLen;
        const uint32_t div_value = (kernelbin.kernel_info.kernel_size - 1U) / MEM_ALIGN_SIZE + 1U;
        if (div_value > (UINT32_MAX / MEM_ALIGN_SIZE)) {
            GELOGW("[Mobile] div_value is overflow(>UINT32_MAX), failed.");
            return UINT32_MAX;
        }
        kernel_binary_total_size += div_value * MEM_ALIGN_SIZE;
    }
    if (sizeof(Tlv) > UINT32_MAX) {
        GELOGW("[Mobile] tlv size is overflow(>UINT32_MAX), failed.");
        return UINT32_MAX;
    }
    size = size + static_cast<uint32_t>(sizeof(Tlv)) + kernel_binary_total_size;
    return size;
}

uint32_t KernelBinManager::GetKernelInfoTlvSize()
{
    uint32_t size = 0;
    for (auto kernelbin: kernelbin_list_) {
        GE_ASSERT_TRUE(kernelbin.stub_name.size() <= UINT32_MAX, "[Mobile] overflow, failed.");
        const uint32_t char_len = static_cast<uint32_t>(kernelbin.stub_name.size()) + 1U;
        GE_ASSERT_TRUE((sizeof(Tlv) <= UINT32_MAX) && (sizeof(KernelInfo) <= UINT32_MAX),
            "[Mobile] overflow, failed.");
        size += sizeof(Tlv) + sizeof(KernelInfo) + char_len;
    }
    return size;
}

ge::Status KernelBinManager::SerializeKernelInfo(uint8_t* start_addr, size_t addr_len)
{
    uint32_t accumulated_len = 0U;
    GE_ASSERT_NOTNULL(start_addr, "[Mobile] start_addr is empty.");
    for (auto kernelbin: kernelbin_list_) {
        GE_ASSERT_TRUE(accumulated_len < addr_len,
            "[Mobile] accumulated_len is overflow, failed. accumulated_len: %lu, len: %lu", accumulated_len, addr_len);
        Tlv* tlv = reinterpret_cast<Tlv*>(&start_addr[accumulated_len]);
        GE_ASSERT_TRUE(kernelbin.stub_name.size() <= UINT32_MAX, "[Mobile] overflow, failed.");
        const uint32_t charLen = static_cast<uint32_t>(kernelbin.stub_name.size()) + 1U;
        const uint32_t len = sizeof(Tlv) + sizeof(KernelInfo) + charLen;
        tlv->type = static_cast<uint32_t>(BinSectionType::BIN_SECTION_TYPE_KERNEL_INFO);
        tlv->len = len;
        const size_t kernel_info_start_index = accumulated_len + sizeof(Tlv);
        GE_ASSERT_TRUE(kernel_info_start_index < addr_len,
            "[Mobile] kernel_info_start_index is overflow, failed. kernel_info_start_index: %lu, len: %lu", kernel_info_start_index, addr_len);
        KernelInfo* kernel_info = reinterpret_cast<KernelInfo*>(&start_addr[kernel_info_start_index]);
        kernel_info->func_mode = kernelbin.kernel_info.func_mode;
        kernel_info->magic = kernelbin.kernel_info.magic;
        const uint32_t div_value = (kernelbin.kernel_info.kernel_size - 1U) / MEM_ALIGN_SIZE + 1U;
        GE_ASSERT_TRUE(div_value <= (UINT32_MAX / MEM_ALIGN_SIZE), "[Mobile] div_value is overflow(>UINT32_MAX), failed.");
        kernel_info->kernel_size = div_value * MEM_ALIGN_SIZE;
        kernel_info->kernel_offset = kernelbin.kernel_info.kernel_offset;
        const size_t stub_name_start_index = accumulated_len + sizeof(Tlv) + sizeof(KernelInfo);
        GE_ASSERT_TRUE(stub_name_start_index < addr_len,
            "[Mobile] stub_name_start_index is overflow, failed. stub_name_start_index: %lu, len: %lu", stub_name_start_index, addr_len);
        const auto ret = memcpy_s(&start_addr[stub_name_start_index], addr_len - stub_name_start_index, kernelbin.stub_name.c_str(),
            kernelbin.stub_name.size());
        GE_ASSERT_TRUE(ret == EOK, "[Mobile] memcpy_s failed.");
        accumulated_len += len;
        start_addr[accumulated_len - 1U] = 0;
    }
    return ge::SUCCESS;
}

ge::Status KernelBinManager::SerializeBinary(uint8_t* start_addr, size_t len)
{
    uint32_t accumulated_len = 0;
    Tlv* tlv = reinterpret_cast<Tlv*>(start_addr);
    tlv->type = static_cast<uint32_t>(BinSectionType::BIN_SECTION_TYPE_BINARY);
    accumulated_len += sizeof(Tlv);
    for (auto kernelbin: kernelbin_list_) {
        GE_ASSERT_NOTNULL(kernelbin.stub_func, "[Mobile] kernel bin stub func is null.");
        GE_ASSERT_TRUE(accumulated_len < len,
            "[Mobile] accumulated_len is overflow, failed. accumulated_len: %lu, len: %lu", accumulated_len, len);
        void* binary = static_cast<void*>(&start_addr[accumulated_len]);
        const size_t size = static_cast<size_t>(kernelbin.kernel_info.kernel_size);
        const auto ret = memmove_s(binary, static_cast<size_t>(len - accumulated_len), kernelbin.stub_func, size);
        GE_ASSERT_TRUE(ret == EOK, "[Mobile] memcpy_s failed.");
        const uint32_t div_value = (kernelbin.kernel_info.kernel_size - 1U) / MEM_ALIGN_SIZE + 1U;
        GE_ASSERT_TRUE(div_value <= (UINT32_MAX / MEM_ALIGN_SIZE), "[Mobile] div_value is overflow(>UINT32_MAX), failed.");
        accumulated_len += div_value * MEM_ALIGN_SIZE;
    }
    tlv->len = accumulated_len;
    return ge::SUCCESS;
}

ge::Status KernelBinManager::SaveKernelBinToBuffer(ge::BaseBuffer& buffer)
{
    GE_ASSERT_NOTNULL(buffer.GetData(), "[Mobile] buffer is nullptr.");
    uint32_t size = GetBinSectionSize();
    GE_ASSERT_TRUE((buffer.GetSize() >= size) && (buffer.GetSize() >= GetKernelInfoTlvSize()),
        "[Mobile] buffer length = %zu, not enough for size = %u", buffer.GetSize(), size);
    std::uint8_t* data = buffer.GetData();
    auto ret = SerializeKernelInfo(data, buffer.GetSize());
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] SerializeKernelInfo failed.");
    ret = SerializeBinary(&data[GetKernelInfoTlvSize()], buffer.GetSize() - GetKernelInfoTlvSize());
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] SerializeBinary failed.");
    return ge::SUCCESS;
}

} // namespace mobile
} // namespace ge
