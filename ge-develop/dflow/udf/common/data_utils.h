/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_COMMON_DATA_UTILS_H
#define UDF_COMMON_DATA_UTILS_H

#include <cstdint>
#include <string>
#include <vector>
#include "flow_func/tensor_data_type.h"
#include "flow_func/flow_func_defines.h"

namespace FlowFunc {
FLOW_FUNC_VISIBILITY int32_t GetSizeByDataType(TensorDataType data_type);

/**
 * @brief check whether uint64 addition can result in overflow
 * @param a addend
 * @param b addend
 * @return true: overflow; false: not overflow
 */
bool CheckAddOverflowUint64(const uint64_t &a, const uint64_t &b);

/**
 * @brief Check if a * b overflow.
 * @param a multiplier
 * @param b Multiplicand
 * @return true: overflow; false: not overflow
 */
FLOW_FUNC_VISIBILITY bool CheckMultiplyOverflowInt64(const int64_t &a, const int64_t &b);

/**
 * @brief calc tensor element size.
 * only support ND and not support unknown shape
 * @param shape tensor shape
 * @return negative:failed , other element count
 */
int64_t CalcElementCnt(const std::vector<int64_t> &shape);

/**
 * @brief calc data size.
 * only support ND and not support unknown shape
 * @param shape tensor shape
 * @param data_type data type
 * @return negative: failed, other:data size.
 */
FLOW_FUNC_VISIBILITY int64_t CalcDataSize(const std::vector<int64_t> &shape, TensorDataType data_type);

/**
 * @brief convert string to int32.
 * @param str a numeric string
 * @param val int32 value converted by str
 * @return true: convert success, false:convert failed.
 */
FLOW_FUNC_VISIBILITY bool ConvertToInt32(const std::string &str, int32_t &val);

/**
 * @brief convert string to int64.
 * @param str a numeric string
 * @param val int64 value converted by str
 * @return true: convert success, false:convert failed.
 */
FLOW_FUNC_VISIBILITY bool ConvertToInt64(const std::string &str, int64_t &val);
}

#endif // UDF_COMMON_DATA_UTILS_H
