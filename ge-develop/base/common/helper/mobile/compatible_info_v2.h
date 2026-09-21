/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_COMPATIBLE_INFO_V2_H
#define BASE_COMMON_HELPER_MOBILE_COMPATIBLE_INFO_V2_H

#include <algorithm>
#include <string>
#include "base_buffer.h"
#include "ge_common/ge_api_error_codes.h"

namespace ge {

class CompatibleInfoV2 {
public:
    CompatibleInfoV2() = default;

    ~CompatibleInfoV2() = default;

    std::size_t GetSize() const;

    ge::Status GetCompatibleInfo(ge::BaseBuffer& buffer) const;
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_COMPATIBLE_INFO_V2_H
