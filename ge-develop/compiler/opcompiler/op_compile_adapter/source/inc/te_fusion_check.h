/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_CHECK_H
#define TE_FUSION_CHECK_H

#include "inc/te_fusion_log.h"

namespace te {
namespace fusion {
/**
 * @ingroup TE_FUSION
 * @brief check cond and execute exec_expr
 */
#define TE_FUSION_CHECK(cond, exec_expr) \
    if (cond) {                          \
        exec_expr;                       \
    }

/**
 * @ingroup TE_FUSION
 * @brief check cond, print python error and execute exec_expr
 */
#define TE_FUSION_CHECK_WITH_DUMP_PYERR(cond, exec_expr) \
    if (cond) {                                          \
        HandleManager::Instance().TE_PyErr_Print();      \
        exec_expr;                                       \
    }

#define TE_FUSION_NOTNULL(val)                                  \
    do {                                                        \
        if ((val) == nullptr) {                                 \
            TE_ERRLOG("Parameter[%s] must not be null.", #val); \
            return false;                                       \
        }                                                       \
    } while (0)

// if failed to make_shared, then print log and execute the statement.
#define TE_FUSION_MAKE_SHARED(exec_expr0, exec_expr1) \
    do {\
        try {\
            exec_expr0;\
        }\
        catch(...) {\
            TE_ERRLOG("Make shared failed");\
            exec_expr1;\
        }\
    } while (0)

inline bool CheckUint64MulOverflow(uint64_t a, uint64_t b)
{
    if (a == 0 || b == 0) {
        return true;
    }

    if (a > (UINT64_MAX / b)) {
        return false;
    }

    return true;
}

#define TE_FUSION_UINT64_MULCHECK(a, b)                                        \
    if (CheckUint64MulOverflow((a), (b)) != true) {                            \
        TE_ERRLOG("UINT64 %lu and %lu multiplication can result in overflow!", \
            static_cast<uint64_t>(a),                                          \
            static_cast<uint64_t>(b));                                         \
        return false;                                                          \
    }

inline bool CheckInt64MulOverflow(int64_t m, int64_t n)
{
    if (m > 0) {
        if (n > 0) {
            if (m > (static_cast<int64_t>(INT64_MAX) / n)) {
                return false;
            }
        } else {
            if (n < (static_cast<int64_t>(INT64_MIN) / m)) {
                return false;
            }
        }
    } else {
        if (n > 0) {
            if (m < (static_cast<int64_t>(INT64_MIN) / n)) {
                return false;
            }
        } else {
            if ((m != 0) && (n < (static_cast<int64_t>(INT64_MAX) / m))) {
                return false;
            }
        }
    }
    return true;
}

#define TE_FUSION_INT64_MULCHECK(a, b)                                                                     \
    if (!CheckInt64MulOverflow((a), (b))) {                                                                \
        TE_ERRLOG("INT64 %ld and %ld multiplication can result in overflow!", (int64_t)(a), (int64_t)(b)); \
        return false;                                                                                      \
    }

inline bool CheckInt64AddOverflow(int64_t a, int64_t b)
{
    if (((b > 0) && (a > (static_cast<int64_t>(INT64_MAX) - b))) ||
        ((b < 0) && (a < (static_cast<int64_t>(INT64_MIN) - b)))) {
        return false;
    }
    return true;
}

#define TE_FUSION_INT64_ADDCHECK(a, b)                                              \
    if (!CheckInt64AddOverflow((a), (b))) {                                         \
        TE_ERRLOG("Int64 %ld and %ld addition can result in overflow!", (a), (b));  \
        return false;                                                               \
    }
} // namespace fusion
} // namespace te
#endif /* TE_FUSION_CHECK_H */
