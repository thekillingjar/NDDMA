/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UDF_TOOLCHAIN_DUMP_UDF_DUMP_COMMON_UTILS_H
#define UDF_TOOLCHAIN_DUMP_UDF_DUMP_COMMON_UTILS_H

#include <map>
#include <string>
#include "ff_dump_data.pb.h"
#include "flow_func/tensor_data_type.h"

namespace FlowFunc {
namespace dumpNS = ff::toolkit::dumpdata;
class UdfDumpCommonUtils {
public:
    static dumpNS::OutputDataType GetDumpDataType(TensorDataType data_type) {
        static const std::map<TensorDataType, dumpNS::OutputDataType> dump_data_type_map = {
            // key:TensorDataType datatype,value:proto datatype
            {TensorDataType::DT_UNDEFINED, dumpNS::DT_UNDEFINED},
            {TensorDataType::DT_FLOAT, dumpNS::DT_FLOAT},
            {TensorDataType::DT_FLOAT16, dumpNS::DT_FLOAT16},
            {TensorDataType::DT_INT8, dumpNS::DT_INT8},
            {TensorDataType::DT_UINT8, dumpNS::DT_UINT8},
            {TensorDataType::DT_INT16, dumpNS::DT_INT16},
            {TensorDataType::DT_UINT16, dumpNS::DT_UINT16},
            {TensorDataType::DT_INT32, dumpNS::DT_INT32},
            {TensorDataType::DT_INT64, dumpNS::DT_INT64},
            {TensorDataType::DT_UINT32, dumpNS::DT_UINT32},
            {TensorDataType::DT_UINT64, dumpNS::DT_UINT64},
            {TensorDataType::DT_BOOL, dumpNS::DT_BOOL},
            {TensorDataType::DT_DOUBLE, dumpNS::DT_DOUBLE},
            {TensorDataType::DT_DUAL, dumpNS::DT_DUAL},
            {TensorDataType::DT_QINT8, dumpNS::DT_QINT8},
            {TensorDataType::DT_QINT16, dumpNS::DT_QINT16},
            {TensorDataType::DT_QINT32, dumpNS::DT_QINT32},
            {TensorDataType::DT_QUINT8, dumpNS::DT_QUINT8},
            {TensorDataType::DT_QUINT16, dumpNS::DT_QUINT16},
            {TensorDataType::DT_INT4, dumpNS::DT_INT4},
            {TensorDataType::DT_UINT1, dumpNS::DT_UINT1},
            {TensorDataType::DT_INT2, dumpNS::DT_INT2},
            {TensorDataType::DT_UINT2, dumpNS::DT_UINT2},
        };
        const auto iter = dump_data_type_map.find(data_type);
        if (iter == dump_data_type_map.cend()) {
            return dumpNS::DT_UNDEFINED;
        }
        return iter->second;
    }

    static void ReplaceStringElem(std::string &str) {
        replace(str.begin(), str.end(), '/', '_');
        replace(str.begin(), str.end(), '\\', '_');
        replace(str.begin(), str.end(), '.', '_');
    }
};
}

#endif // UDF_TOOLCHAIN_DUMP_UDF_DUMP_COMMON_UTILS_H