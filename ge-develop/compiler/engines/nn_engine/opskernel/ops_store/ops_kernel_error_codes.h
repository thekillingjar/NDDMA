/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPSKERNEL_OPS_STORE_OPS_KERNEL_ERROR_CODES_H_
#define FUSION_ENGINE_OPSKERNEL_OPS_STORE_OPS_KERNEL_ERROR_CODES_H_

#include <string>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
/** Op Kernel Store module ID */
const uint8_t FE_MODID_OP_KERNEL_STORE = 56;

#define FE_DEF_ERRORNO_OP_KERNEL_STORE(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_OP_KERNEL_STORE, name, value, desc)

FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_PARSE_FAILED, 0, "Failed to parse the op kernel store cfg file!");  // 0x3380000
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_MAP_KEY_FIND_FAILED, 1,
                               "Failed to find the map key in FEOpKernelStore!");  // 0x3380001
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_STRING_CONVERT_FAILED, 2,
                               "Failed to convert string in FEOpKernelStore!");                            // 0x3380002
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_READ_CFG_FILE_FAILED, 3, "Failed to read cfg file. I/O failed!");  // 0x3380003
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_CFG_FILE_EMPTY, 4,
                               "Failed to read cfg file. Empty configuration file path!");     // 0x3380004
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_CFG_NAME_EMPTY, 5, "Failed to get sub store name!");  // 0x3380005
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_CFG_FILE_NOT_EXIST, 6,
                               "Failed to read op information. Configuration file does not exist!");         // 0x3380006
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_MAKE_SHARED_FAILED, 7, "Failed to make shared in fe ops store!");  // 0x3380006
FE_DEF_ERRORNO_OP_KERNEL_STORE(OPS_SUB_STORE_NOT_EXIST, 8, "Failed to find specific sub store!");          // 0x3380007
FE_DEF_ERRORNO_OP_KERNEL_STORE(OPS_SUB_STORE_PTR_NULL, 9, "Failed to get sub store pointer!");             // 0x3380009
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL, 10,
                               "Failed to find op in all sub stores!");  // 0x33800A
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_ATTR_NOT_FOUND_IN_OP_KERNEL_INFO, 11,
                               "Failed to find attr name in op kernel info!");  // 0x33800B
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_ATTR_EMPTY_IN_OP_KERNEL_INFO, 12,
                               "None attribute found in op kernel info!");  // 0x33800C
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_INPUT_NOT_FOUND_IN_OP_KERNEL_INFO, 13,
                               "Failed to find input info in op kernel info!");  // 0x33800D
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_OUTPUT_NOT_FOUND_IN_OP_KERNEL_INFO, 14,
                               "Failed to find output info in op kernel info!");  // 0x33800E
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_NOT_FOUND_IN_GET_HIGH_PRIO_OP_KERNEL, 15,
                               "Failed to find op in all sub stores in GetHighPrioOpKernelInfoPtr!");  // 0x33800F
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_KERNEL_INFO_NULL_PTR, 16, "Param is null ptr!");  // 0x338010
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_SUB_STORE_PLGUIN_INIT_FAILED, 17,
                               "Failed to init plugin tbe sub op store!");  // 0x338011
FE_DEF_ERRORNO_OP_KERNEL_STORE(OP_SUB_STORE_ILLEGAL_JSON, 18,
                               "Illegal json file, fail to parse json file!");  // 0x338018

}  // namespace fe

#endif  // FUSION_ENGINE_OPSKERNEL_OPS_STORE_OPS_KERNEL_ERROR_CODES_H_
