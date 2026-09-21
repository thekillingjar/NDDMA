/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "string_utils.h"

namespace acl {
void StringUtils::Split(const std::string &str, const char_t delim, std::vector<std::string> &elems)
{
    elems.clear();
    if (str.empty()) {
        elems.emplace_back("");
        return;
    }

    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }

    const auto strSize = str.size();
    if ((strSize > 0U) && (str[strSize - 1U] == delim)) {
        elems.emplace_back("");
    }
}

void StringUtils::Trim(std::string &str)
{
    if (str.empty()) {
        return;
    }
    (void)str.erase(0U, str.find_first_not_of(" "));
    (void)str.erase(str.find_last_not_of(" ") + 1U);
}

void StringUtils::Strip(std::string &str, const std::string &sep)
{
    if (str.empty()) {
        return;
    }
    (void) str.erase(0U, str.find_first_not_of(sep));
    (void) str.erase(str.find_last_not_of(sep) + 1U);
}

bool StringUtils::IsDigit(const std::string &str)
{
    if (str.empty()) {
        return false;
    }

    for (const char_t &c : str) {
        if (isdigit(static_cast<int32_t>(c)) == 0) {
            return false;
        }
    }
    return true;
}
} // namespace acl