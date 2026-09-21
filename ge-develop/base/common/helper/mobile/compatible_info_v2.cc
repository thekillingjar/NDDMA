/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compatible_info_v2.h"

#include <iostream>
#include "securec.h"
#include "common/checker.h"
#include "ge_context.h"

namespace {

enum class Platform : uint32_t {
    PLATFORM_KIRINX90 = 10,
    PLATFORM_KIRIN9030 = 16,
};

uint32_t GetPlatform()
{
    const std::map<std::string, uint32_t> soc_version_to_platform_map = {
        {"KirinX90", static_cast<uint32_t>(Platform::PLATFORM_KIRINX90)},
        {"Kirin9030", static_cast<uint32_t>(Platform::PLATFORM_KIRIN9030)},
    };
    std::string soc_version;
    (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);
    auto it = soc_version_to_platform_map.find(soc_version);
    if (it != soc_version_to_platform_map.end()) {
        return it->second;
    }
    // default is KirinX90
    return static_cast<uint32_t>(Platform::PLATFORM_KIRINX90);
}

} // namespace

namespace ge {

constexpr uint32_t MAGIC_NUM = 0x5B5B5B5BU;

struct CompatibleSerialV2 {
    uint32_t magic_num;
    uint32_t reserved;
    uint32_t platform;
    uint32_t ids_num;
    uint32_t ids[0];
};

std::size_t CompatibleInfoV2::GetSize() const
{
    return sizeof(CompatibleSerialV2);
}

ge::Status CompatibleInfoV2::GetCompatibleInfo(ge::BaseBuffer& buffer) const
{
    constexpr size_t size = sizeof(CompatibleSerialV2);
    GE_ASSERT_TRUE(buffer.GetSize() >= size,
        "[Mobile] buffer length = %zu, is not need = %zu", buffer.GetSize(), size);
    GE_ASSERT_NOTNULL(buffer.GetData(), "[Mobile] buffer is null.");

    CompatibleSerialV2* head = reinterpret_cast<CompatibleSerialV2*>(buffer.GetData());
    head->magic_num = MAGIC_NUM;
    head->reserved = static_cast<uint32_t>(0);
    head->platform = static_cast<uint32_t>(GetPlatform());
    head->ids_num = static_cast<uint32_t>(0);
    return ge::SUCCESS;
}

} // namespace ge
