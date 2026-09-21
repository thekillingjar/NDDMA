/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGISTER_OP_TILING_OP_TILING_CONSTANTS_H_
#define REGISTER_OP_TILING_OP_TILING_CONSTANTS_H_

#include <string>
#include <map>
#include "graph/types.h"

namespace optiling {
const std::string COMPILE_INFO_JSON = "compile_info_json";
const std::string COMPILE_INFO_KEY = "compile_info_key";
const std::string COMPILE_INFO_WORKSPACE_SIZE_LIST = "_workspace_size_list";
const std::string ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
const std::string ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";
const std::string ATTR_NAME_ATOMIC_CLEAN_WORKSPACE = "_optiling_atomic_add_mem_size";
const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";
const std::string OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN = "DynamicAtomicAddrClean";
const std::string OP_TYPE_AUTO_TILING = "AutoTiling";
const std::string kMemoryCheck = "_memcheck";
const std::string kOriOpParaSize = "ori_op_para_size";
const std::map<ge::DataType, std::string> DATATYPE_STRING_MAP {
    {ge::DT_FLOAT, "float32"},
    {ge::DT_FLOAT16, "float16"},
    {ge::DT_INT8, "int8"},
    {ge::DT_INT16, "int16"},
    {ge::DT_INT32, "int32"},
    {ge::DT_INT64, "int64"},
    {ge::DT_UINT8, "uint8"},
    {ge::DT_UINT16, "uint16"},
    {ge::DT_UINT32, "uint32"},
    {ge::DT_UINT64, "uint64"},
    {ge::DT_BOOL, "bool"},
    {ge::DT_DOUBLE, "double"},
    {ge::DT_DUAL, "dual"},
    {ge::DT_DUAL_SUB_INT8, "dual_sub_int8"},
    {ge::DT_DUAL_SUB_UINT8, "dual_sub_uint8"}
};

}  // namespace optiling

#endif  // REGISTER_OP_TILING_OP_TILING_CONSTANTS_H_
