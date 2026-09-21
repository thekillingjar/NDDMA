/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_utils.h"
#include <map>
#include <limits>
#include <stdexcept>
#include "udf_log.h"

namespace FlowFunc {
namespace {
constexpr int32_t kDataTypeSizeBitOffset = 1000;
}
bool CheckAddOverflowUint64(const uint64_t &a, const uint64_t &b) {
    if (a > (UINT64_MAX - b)) {
        return true;
    }
    return false;
}

bool CheckMultiplyOverflowInt64(const int64_t &a, const int64_t &b) {
    if (a > 0) {
        if (b > 0) {
            if (a > (std::numeric_limits<int64_t>::max() / b)) {
                return true;
            }
        } else {
            if (b < (std::numeric_limits<int64_t>::min() / a)) {
                return true;
            }
        }
    } else {
        if (b > 0) {
            if (a < (std::numeric_limits<int64_t>::min() / b)) {
                return true;
            }
        } else {
            if ((a != 0) && (b < (std::numeric_limits<int64_t>::max() / a))) {
                return true;
            }
        }
    }
    return false;
}

int32_t GetSizeByDataType(TensorDataType data_type) {
    static const std::map<TensorDataType, int32_t> sizeMap = {
        {TensorDataType::DT_FLOAT,   4},
        {TensorDataType::DT_FLOAT16, 2},
        {TensorDataType::DT_BF16,    2},
        {TensorDataType::DT_INT8,    1},
        {TensorDataType::DT_INT16,   2},
        {TensorDataType::DT_UINT16,  2},
        {TensorDataType::DT_UINT8,   1},
        {TensorDataType::DT_INT32,   4},
        {TensorDataType::DT_INT64,   8},
        {TensorDataType::DT_UINT32,  4},
        {TensorDataType::DT_UINT64,  8},
        {TensorDataType::DT_BOOL,    1},
        {TensorDataType::DT_DOUBLE,  8},
        {TensorDataType::DT_QINT8,   1},
        {TensorDataType::DT_QINT16,  2},
        {TensorDataType::DT_QINT32,  4},
        {TensorDataType::DT_QUINT8,  1},
        {TensorDataType::DT_QUINT16, 2},
        {TensorDataType::DT_DUAL,    5},
        {TensorDataType::DT_INT4,    kDataTypeSizeBitOffset + 4},
        {TensorDataType::DT_UINT1,   kDataTypeSizeBitOffset + 1},
        {TensorDataType::DT_INT2,    kDataTypeSizeBitOffset + 2},
        {TensorDataType::DT_UINT2,   kDataTypeSizeBitOffset + 2}
    };
    const auto iter = sizeMap.find(data_type);
    if (iter == sizeMap.cend()) {
        return -1;
    }
    return iter->second;
}

int64_t CalcElementCnt(const std::vector<int64_t> &shape) {
    int64_t element_cnt = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            UDF_LOG_ERROR("dim is negative, not support now, dim=%ld.", dim);
            return -1;
        }
        if (CheckMultiplyOverflowInt64(element_cnt, dim)) {
            UDF_LOG_ERROR("CalcElementCnt failed, when multiplying %ld and %ld.", element_cnt, dim);
            return -1;
        }
        element_cnt *= dim;
    }
    return element_cnt;
}

int64_t CalcDataSize(const std::vector<int64_t> &shape, TensorDataType data_type) {
    int32_t type_size = GetSizeByDataType(data_type);
    if (type_size < 0) {
        UDF_LOG_ERROR("data_type=%d is not support.", static_cast<int32_t>(data_type));
        return -1;
    }
    int64_t element_cnt = CalcElementCnt(shape);
    if (element_cnt < 0) {
        UDF_LOG_ERROR("CalcElementCnt failed, result=%ld.", element_cnt);
        return -1;
    }

    if (type_size > kDataTypeSizeBitOffset) {
        int32_t type_bit_size = type_size - kDataTypeSizeBitOffset;
        if (CheckMultiplyOverflowInt64(element_cnt, static_cast<int64_t>(type_bit_size))) {
            UDF_LOG_ERROR("CalcDataSize failed, when multiplying %ld and %d.", element_cnt, type_bit_size);
            return -1;
        }
        int64_t data_bit_size = element_cnt * type_bit_size;
        constexpr int64_t byteBitSize = 8;
        int64_t data_size = data_bit_size / byteBitSize;
        if (data_bit_size % byteBitSize != 0) {
            data_size += 1;
        }
        return data_size;
    } else {
        if (CheckMultiplyOverflowInt64(element_cnt, static_cast<int64_t>(type_size))) {
            UDF_LOG_ERROR("CalcDataSize failed, when multiplying %ld and %d.", element_cnt, type_size);
            return -1;
        }
        return element_cnt * type_size;
    }
}

bool ConvertToInt32(const std::string &str, int32_t &val) {
    try {
        val = std::stoi(str);
    } catch (std::invalid_argument &) {
        UDF_LOG_ERROR("The digit str:%s is invalid", str.c_str());
        return false;
    } catch (std::out_of_range &) {
        UDF_LOG_ERROR("The digit str:%s is out of range int32", str.c_str());
        return false;
    } catch (...) {
        UDF_LOG_ERROR("The digit str:%s cannot change to int", str.c_str());
        return false;
    }
    return true;
}

bool ConvertToInt64(const std::string &str, int64_t &val) {
    try {
        val = std::stoll(str);
    } catch (std::invalid_argument &) {
        UDF_LOG_ERROR("The digit str:%s is invalid", str.c_str());
        return false;
    } catch (std::out_of_range &) {
        UDF_LOG_ERROR("The digit str:%s is out of range int64", str.c_str());
        return false;
    } catch (...) {
        UDF_LOG_ERROR("The digit str:%s cannot change to int64", str.c_str());
        return false;
    }
    return true;
}
}
