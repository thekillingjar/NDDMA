/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_INC_TENSOR_ENGINE_FUSION_CONSTANTS_H_
#define ATC_OPCOMPILER_INC_TENSOR_ENGINE_FUSION_CONSTANTS_H_

#include <string>
#include <map>
#include "tensor_engine/fusion_types.h"

namespace te {
const std::string NO_NEED_REASON = "No_Need_Reason";
const std::string CORE_TYPE_NAME = "coreType";
const std::string SHORT_SOC_VERSION = "short_soc_version";

const std::map<ATTR_DTYPE, std::string> TbeAttrDtypeToStringMap = {
    {ATTR_INT8,            "int8"},
    {ATTR_UINT8,           "uint8"},
    {ATTR_INT16,           "int16"},
    {ATTR_UINT16,          "uint16"},
    {ATTR_INT32,           "int32"},
    {ATTR_UINT32,          "uint32"},
    {ATTR_INT64,           "int64"},
    {ATTR_UINT64,          "uint64"},
    {ATTR_FLOAT32,         "float32"},
    {ATTR_DOUBLE,          "double"},
    {ATTR_BOOL,            "bool"},
    {ATTR_STR,             "str"},
    {ATTR_LIST_INT8,       "list_int8"},
    {ATTR_LIST_UINT8,      "list_uint8"},
    {ATTR_LIST_INT16,      "list_int16"},
    {ATTR_LIST_UINT16,     "list_uint16"},
    {ATTR_LIST_INT32,      "list_int32"},
    {ATTR_LIST_UINT32,     "list_uint32"},
    {ATTR_LIST_INT64,      "list_int64"},
    {ATTR_LIST_UINT64,     "list_uint64"},
    {ATTR_LIST_FLOAT32,    "list_float32"},
    {ATTR_LIST_DOUBLE,     "list_double"},
    {ATTR_LIST_BOOL,       "list_bool"},
    {ATTR_LIST_STR,        "list_str"},
    {ATTR_LIST_LIST_INT64, "list_list_int64"},
    {ATTR_LIST_LIST_FLOAT, "list_list_float"}
};
}
#endif  // ATC_OPCOMPILER_INC_TENSOR_ENGINE_FUSION_CONSTANTS_H_