/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <vector>
#include <string>
#include <sstream>
#include "graph/ascend_limits.h"
#include "graph/small_vector.h"
#include "common/log_inner.h"

namespace acl {
class StringUtils {
public:
    template<typename T>
    static std::string VectorToString(const std::vector<T> &values)
    {
        std::stringstream ss;
        ss << '[';
        const auto size = values.size();
        for (size_t i = 0U; i < size; ++i) {
            ss << values[i];
            if (i != (size - 1U)) {
                ss << ", ";
            }
        }
        ss << ']';
        return ss.str();
    }

    template<typename T>
    static std::string VectorToString(const ge::SmallVector<T, static_cast<size_t>(ge::kDefaultMaxInputNum)> &values)
    {
        std::stringstream ss;
        ss << '[';
        const auto size = values.size();
        for (size_t i = 0U; i < size; ++i) {
            ss << values[i];
            if (i != (size - 1U)) {
                ss << ", ";
            }
        }
        ss << ']';
        return ss.str();
    }

    template<typename T>
    static std::string VectorToString(const std::vector<std::vector<T>> &values)
    {
        if (values.empty()) {
            return "[]";
        }

        const auto size = values.size();
        std::stringstream ss;
        ss << '[';
        for (size_t i = 0U; i < size; ++i) {
            ss << '[';
            const auto &subVec = values[i];
            const auto subSize = subVec.size();
            for (size_t j = 0U; j < subSize; ++j) {
                ss << subVec[j];
                if (j != (subSize - 1U)) {
                    ss << ", ";
                }
            }
            ss << "]";
            if (i != (size - 1U)) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    template<typename T>
    static std::string VectorToString(const std::vector<std::pair<T, T>> &values)
    {
        if (values.empty()) {
            return "[]";
        }

        const auto size = values.size();
        std::stringstream ss;
        ss << '[';
        for (size_t i = 0U; i < size; ++i) {
            ss << '[';
            ss << values[i].first;
            ss << ", ";
            ss << values[i].second;
            ss << "]";
            if (i != (size - 1U)) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    static void Split(const std::string &str, const char_t delim, std::vector<std::string> &elems);

    static void Trim(std::string &str);

    static void Strip(std::string &str, const std::string &sep);

    static bool IsDigit(const std::string &str);
};
} // namespace acl

#endif // STRING_UTILS_H
