/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_KERNELBIN_INFO_H
#define BASE_COMMON_HELPER_MOBILE_KERNELBIN_INFO_H

#include <string>
#include <vector>
#include "base_buffer.h"
#include "compiled_target.h"
#include "ge_common/ge_api_error_codes.h"

namespace ge {
namespace mobile {

enum class BinSectionType : std::uint8_t {
    BIN_SECTION_TYPE_INVAILD = 0,
    BIN_SECTION_TYPE_KERNEL_INFO = 1,
    BIN_SECTION_TYPE_BINARY = 2,
    BIN_SECTION_TYPE_MAX = 255,
};

struct KernelInfo {
    uint32_t func_mode;
    uint32_t magic;
    uint32_t kernel_size;
    uint32_t kernel_offset;
};

struct ReCompileInfo {
    // not use
    void* kernel_def = nullptr;
};

struct KernelBin {
    KernelInfo kernel_info;
    const void* stub_func;
    std::string stub_name;
    ReCompileInfo re_compile_info;
};

class KernelBinManager {
public:
    KernelBinManager() = default;

    ~KernelBinManager() = default;

    ge::Status SaveKernelBinToBuffer(ge::BaseBuffer& buffer);

    uint32_t GetBinSectionSize();

    void AddKernelBin(const KernelBin& kernelbin);

private:
    uint32_t GetKernelInfoTlvSize();

    ge::Status SerializeKernelInfo(uint8_t* start_addr, size_t addr_len);

    ge::Status SerializeBinary(uint8_t* start_addr, size_t len);

private:
    std::vector<KernelBin> kernelbin_list_;
};

} // namespace mobile
} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_KERNELBIN_INFO_H
