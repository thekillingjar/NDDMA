/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_engine_test_helper.h"
#include "llm_string_util.h"
#include "llm_test_helper.h"
#include "graph/utils/type_utils.h"

namespace ge {
const std::map<DataType, std::string> kDataTypeToStringMap = {
    {DT_UNDEFINED, "DT_UNDEFINED"},            // Used to indicate a DataType field has not been set.
    {DT_FLOAT, "DT_FLOAT"},                    // float type
    {DT_FLOAT16, "DT_FLOAT16"},                // fp16 type
    {DT_INT8, "DT_INT8"},                      // int8 type
    {DT_INT16, "DT_INT16"},                    // int16 type
    {DT_UINT16, "DT_UINT16"},                  // uint16 type
    {DT_UINT8, "DT_UINT8"},                    // uint8 type
    {DT_INT32, "DT_INT32"},                    // uint32 type
    {DT_INT64, "DT_INT64"},                    // int64 type
    {DT_UINT32, "DT_UINT32"},                  // unsigned int32
    {DT_UINT64, "DT_UINT64"},                  // unsigned int64
    {DT_BOOL, "DT_BOOL"},                      // bool type
    {DT_DOUBLE, "DT_DOUBLE"},                  // double type
    {DT_DUAL, "DT_DUAL"},                      // dual output type
    {DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},    // dual output int8 type
    {DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
    {DT_COMPLEX32, "DT_COMPLEX32"},            // complex32 type
    {DT_COMPLEX64, "DT_COMPLEX64"},            // complex64 type
    {DT_COMPLEX128, "DT_COMPLEX128"},          // complex128 type
    {DT_QINT8, "DT_QINT8"},                    // qint8 type
    {DT_QINT16, "DT_QINT16"},                  // qint16 type
    {DT_QINT32, "DT_QINT32"},                  // qint32 type
    {DT_QUINT8, "DT_QUINT8"},                  // quint8 type
    {DT_QUINT16, "DT_QUINT16"},                // quint16 type
    {DT_RESOURCE, "DT_RESOURCE"},              // resource type
    {DT_STRING_REF, "DT_STRING_REF"},          // string ref type
    {DT_STRING, "DT_STRING"},                  // string type
    {DT_VARIANT, "DT_VARIANT"},                // dt_variant type
    {DT_BF16, "DT_BFLOAT16"},                  // dt_bfloat16 type
    {DT_INT4, "DT_INT4"},                      // dt_variant type
    {DT_UINT1, "DT_UINT1"},                    // dt_variant type
    {DT_INT2, "DT_INT2"},                      // dt_variant type
    {DT_UINT2, "DT_UINT2"},                    // dt_variant type
    {DT_HIFLOAT8, "DT_HIFLOAT8"},
    {DT_FLOAT8_E5M2, "DT_FLOAT8_E5M2"},
    {DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"},
    {DT_FLOAT6_E3M2, "DT_FLOAT6_E3M2"},  // mxfp6
    {DT_FLOAT6_E2M3, "DT_FLOAT6_E2M3"},  // mxfp6
    {DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"},  // mxfp4
    {DT_FLOAT4_E1M2, "DT_FLOAT4_E1M2"},  // mxfp4
};

std::string TypeUtils::DataTypeToSerialString(const DataType data_type) {
  const auto it = kDataTypeToStringMap.find(data_type);
  if (it != kDataTypeToStringMap.end()) {
    return it->second.c_str();
  } else {
    LLMLOGW("DataTypeToSerialString: datatype not support %d", data_type);
    return "UNDEFINED";
  }
}
}  // namespace ge
