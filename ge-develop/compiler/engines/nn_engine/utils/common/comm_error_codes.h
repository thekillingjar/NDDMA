/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/** @defgroup FE_ERROR_CODES_GROUP FE Error Code Interface */
/** FE error code definition
 *  @ingroup FE_ERROR_CODES_GROUP
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_COMM_ERROR_CODES_H_
#define FUSION_ENGINE_UTILS_COMMON_COMM_ERROR_CODES_H_

#include <string>
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
/** Fusion rule parser module ID */
const uint8_t FE_MODID_OP_CALCULATE = 62;

#define FE_DEF_ERRORNO_OP_CALCULATOR(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_OP_CALCULATE, name, value, desc)

FE_DEF_ERRORNO_COMMON(INVALID_FILE_PATH, 8, "Failed to get the valid file path.");                        // 0x3320008
FE_DEF_ERRORNO_COMMON(LACK_MANDATORY_CONFIG_KEY, 9, "The mandatory key is not configured in files.");     // 0x3320009
FE_DEF_ERRORNO_COMMON(OPSTORE_NAME_EMPTY, 10, "The name of opstore config is empty.");                    // 0x3320010
FE_DEF_ERRORNO_COMMON(OPSTORE_VALUE_EMPTY, 11, "The value of opstore config is empty.");                  // 0x3320011
FE_DEF_ERRORNO_COMMON(OPSTORE_VALUE_ITEM_SIZE_INCORRECT, 12, "The size of opstore items is incorrect.");  // 0x3320012
FE_DEF_ERRORNO_COMMON(OPSTORE_VALUE_ITEM_EMPTY, 13, "At least one of the opstore item is empty.");        // 0x3320013
FE_DEF_ERRORNO_COMMON(OPSTORE_EMPTY, 14, "There is no OP store in configuration files.");                 // 0x3320014
FE_DEF_ERRORNO_COMMON(OPSTORE_OPIMPLTYPE_REPEAT, 15, "Op impl type of OP stores can not be repeated.");  // 0x3320015
FE_DEF_ERRORNO_COMMON(OPSTORE_OPIMPLTYPE_INVALID, 16, "The op impl type of OP store is invalid.");       // 0x3320016
FE_DEF_ERRORNO_COMMON(OPSTORE_PRIORITY_INVALID, 17, "The priority of OP stores is invalid.");             // 0x3320017
FE_DEF_ERRORNO_COMMON(OPSTORE_CONFIG_NOT_INTEGRAL, 18, "The content of ops store is not integral.");      // 0x3320018

/** Op param calculate error code define */
FE_DEF_ERRORNO_OP_CALCULATOR(FAIL_GET_OP_IMPL_TYPE, 0, "Failed to get the op impl type of op desc.");  // 0x033D0000
FE_DEF_ERRORNO_OP_CALCULATOR(TENSOR_FORMAT_INVALID, 1, "This format is not valid.");                        // 0x3320019
FE_DEF_ERRORNO_OP_CALCULATOR(TENSOR_DATATYPE_INVALID, 2, "This data type is not valid.");                   // 0x3320020
FE_DEF_ERRORNO_OP_CALCULATOR(DIM_VALUE_INVALID, 3, "The dim value must be great than zero.");               // 0x3320021
FE_DEF_ERRORNO_OP_CALCULATOR(TENSOR_DATATYPE_NOT_SUPPORT, 4, "This tensor format is not supported.");       // 0x3320026
}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_COMM_ERROR_CODES_H_
