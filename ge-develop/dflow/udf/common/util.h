/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTIL_H
#define COMMON_UTIL_H

#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <string.h>

namespace FlowFunc {
/**
 * @brief converts Vector to a string.
 * @param [in] v  Vector of type T
 * @return string
 */
template<typename T>
std::string ToString(const std::vector<T> &v) {
    bool first = true;
    std::stringstream ss;
    ss << "[";
    for (const T &x : v) {
        if (first) {
            first = false;
            ss << x;
        } else {
            ss << ", " << x;
        }
    }
    ss << "]";
    return ss.str();
}

template<typename Tp, typename... Args>
inline std::shared_ptr<Tp> MakeShared(Args &&... args) {
    using TpNc = typename std::remove_const<Tp>::type;
    const std::shared_ptr<Tp> ret(new(std::nothrow) TpNc(std::forward<Args>(args)...));
    return ret;
}

inline std::string GetErrorNoStr(int32_t error_no) {
    char err_buf[128U] = {};
    return strerror_r(error_no, err_buf, sizeof(err_buf));
}
}
#endif // COMMON_UTIL_H
