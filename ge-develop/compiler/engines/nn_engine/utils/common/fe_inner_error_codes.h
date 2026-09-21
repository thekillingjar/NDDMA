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

#ifndef FUSION_ENGINE_UTILS_COMMON_FE_INNER_ERROR_CODES_H_
#define FUSION_ENGINE_UTILS_COMMON_FE_INNER_ERROR_CODES_H_

#include <string>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
/** Itf module ID */
const uint8_t FE_MODID_ITF = 51;
/** Shape Format Transfer module ID */
const uint8_t FE_MODID_SHAPE_FORMAT_TRANSFER = 52;
/** IGraph Optimizer module ID */
const uint8_t FE_MODID_GRAPH_OPTIMIZER = 53;
/** Op Judge module ID */
const uint8_t FE_MODID_OP_JUDGE = 54;
/** Op Compiler module ID */
const uint8_t FE_MODID_OP_COMPILER = 55;
/** Op Kernel Store module ID */
const uint8_t FE_MODID_OP_KERNEL_STORE = 56;
/** Task Builder module ID */
const uint8_t FE_MODID_TASK_BUILDER = 57;
/** Graph matcher module ID */
const uint8_t FE_MODID_GRAPH_MATCHER = 58;
/** Graph replace module ID */
const uint8_t FE_MODID_GRAPH_REPLACE = 59;
/** Fusion rule parser module ID */
const uint8_t FE_MODID_FUSION_RULE_PARSER = 60;
/** Op store adapter module ID */
const uint8_t FE_MODID_OP_STORE_ADAPTER = 61;
// LX Fusion module ID
const uint8_t FE_MODID_LX_FUSION = 63;

#define FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_SHAPE_FORMAT_TRANSFER, name, value, desc)
#define FE_DEF_ERRORNO_ITF(name, value, desc) FE_DEF_ERRORNO(SYSID_FE, FE_MODID_ITF, name, value, desc)
#define FE_DEF_ERRORNO_GRAPH_OPTIMIZER(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_GRAPH_OPTIMIZER, name, value, desc)
#define FE_DEF_ERRORNO_OP_JUDGE(name, value, desc) FE_DEF_ERRORNO(SYSID_FE, FE_MODID_OP_JUDGE, name, value, desc)
#define FE_DEF_ERRORNO_OP_COMPILER(name, value, desc) FE_DEF_ERRORNO(SYSID_FE, FE_MODID_OP_COMPILER, name, value, desc)
#define FE_DEF_ERRORNO_OP_KERNEL_STORE(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_OP_KERNEL_STORE, name, value, desc)
#define FE_DEF_ERRORNO_TASK_BUILDER(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_TASK_BUILDER, name, value, desc)
#define FE_DEF_ERRORNO_GRAPH_MATCHER(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_GRAPH_MATCHER, name, value, desc)
#define FE_DEF_ERRORNO_GRAPH_REPLACE(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_GRAPH_REPLACE, name, value, desc)
#define FE_DEF_ERRORNO_FUSION_RULE_PARSER(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_FUSION_RULE_PARSER, name, value, desc)
#define FE_DEF_ERRORNO_OP_STORE_ADAPTER(name, value, desc) \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_OP_STORE_ADAPTER, name, value, desc)

#define FE_DEF_ERRORNO_LX_FUSION(name, value, desc) FE_DEF_ERRORNO(SYSID_FE, FE_MODID_LX_FUSION, name, value, desc)

/** common module error code define */
FE_DEF_ERRORNO_COMMON(MEMALLOC_FAILED, 0, "Failed to allocate memory!");    // 0x3320000
FE_DEF_ERRORNO_COMMON(CALL_CCE_FAILED, 2, "Failed to call CCE API!");       // 0x3320002
FE_DEF_ERRORNO_COMMON(CALL_RT_FAILED, 3, "Failed to call runtime API!");    // 0x3320003
FE_DEF_ERRORNO_COMMON(INTERNAL_ERROR, 4, "Internal errors");                // 0x3320004
FE_DEF_ERRORNO_COMMON(CALL_CSEC_ERROR, 5, "Failed to call libc_sec API!");  // 0x3320005
FE_DEF_ERRORNO_COMMON(CALL_TEE_ERROR, 6, "Failed to call tee API!");        // 0x3320006
FE_DEF_ERRORNO_COMMON(FILE_NOT_EXIST, 8, "The file does not exist.");       // 0x3320007

FE_DEF_ERRORNO_COMMON(TENSOR_FORMAT_NOT_FOUND, 18, "This format has not been found.");        // 0x3320018
FE_DEF_ERRORNO_COMMON(INVALID_TENSOR_FORMAT, 19, "This format is not valid.");                // 0x3320019
FE_DEF_ERRORNO_COMMON(INVALID_TENSOR_DATATYPE, 20, "This data type is not valid.");           // 0x3320020
FE_DEF_ERRORNO_COMMON(INVALID_DIM_VALUE, 21, "The dim value must be greater than zero.");       // 0x3320021
FE_DEF_ERRORNO_COMMON(INVALID_DIM_SIZE, 22, "The size of dim must be greater than 4.");  // 0x3320022

FE_DEF_ERRORNO_COMMON(INVALID_NC1KHKWHWC0_SIZE, 23, "The size of NC1KHKWHWC0 format is not valid.");  // 0x3320023
FE_DEF_ERRORNO_COMMON(INVALID_C1HWNCoC0_SIZE, 24, "The size of C1HWNCoC0 format is not valid.");      // 0x3320024
FE_DEF_ERRORNO_COMMON(NOT_SUPPORT_TENSOR_FORMAT, 25, "This data type is not supported.");             // 0x3320025
FE_DEF_ERRORNO_COMMON(NOT_SUPPORT_TENSOR_DATATYPE, 26, "This tensor format is not supported.");       // 0x3320026
FE_DEF_ERRORNO_COMMON(BEYONG_MAX_TENSOR_ELEMENT_COUNT, 27,
                      "The amount of tensor element is more than the max amount.");  // 0x3320027

FE_DEF_ERRORNO_COMMON(VECTOR_INT64_EMPTY, 30, "The vector of int64 number is empty.");                      // 0x3320030
FE_DEF_ERRORNO_COMMON(ADD_OVERFLOW_INT64, 31, "The addition between two int64 number is overflow.");        // 0x3320031
FE_DEF_ERRORNO_COMMON(MUL_OVERFLOW_INT64, 32, "The multiplication between two int64 number is overflow.");  // 0x3320032
FE_DEF_ERRORNO_COMMON(ADD_OVERFLOW_SIZET, 33, "The addition between two size_t number is overflow.");       // 0x3320033
FE_DEF_ERRORNO_COMMON(CONTINUING_TRANSFORMAT, 34,
                      "We need to transformat in this case of one dimensional shape padding.");             // 0x3320034

/** shape format transfer error code define */
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(SHAPE_FORMAT_TRANSFER_SORTING_FAILED, 0, "Failed to sort!");      // 0x3340000
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(SHAPE_FORMAT_TRANSFER_INSERTING_FAILED, 1, "Failed to insert!");  // 0x3340001
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(SHAPE_FORMAT_TRANSFER_CHECKING_FAILED, 2, "Failed to check!");    // 0x3340002
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(CREATE_CAST_OP_FAILED, 3, "Failed to create CAST op!");           // 0x3340003
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(ADD_CAST_OP_NODE_FAILED, 4, "Failed to add CAST op node!");       // 0x3340004
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(RESHAPE_TYPE_NOT_SUPPORTED, 5,
                                     "Failed to insert Reshape! Reshape type is invalid!");  // 0x3340005
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(RESHAPE_NOT_NECESSARY, 6,
                                     "Failed to insert Reshape! Uncessary to reshape!");  // 0x3340006
FE_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(MERGE_TRANS_OP_NO_MORE_PREDECESSOR, 7,
                                     "There is no more trans op for merging!");  // 0x3340007

/** fusion manager error code define */
FE_DEF_ERRORNO_ITF(OPINFO_STORES_INIT_FAILED, 0, "Failed to init opinfo_kernel_stores!");      // 0x3330000
FE_DEF_ERRORNO_ITF(GRAPH_OPTIMIZER_INIT_FAILED, 1, "Failed to init graphoptimizer!");          // 0x3330001
FE_DEF_ERRORNO_ITF(OPINFO_STORES_FINI_FAILED, 2, "Failed to finilize opinfo_kernel_stores!");  // 0x3330002
FE_DEF_ERRORNO_ITF(GRAPHOPTIMIZER_FINI_FAILED, 3, "Failed to finilize graphoptimizer!");       // 0x3330003
FE_DEF_ERRORNO_ITF(CONFIGURATION_INIT_FAILED, 4, "Failed to initialize configuration!");       // 0x3330004

/** graph optimizer error code define */
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_MAKE_SHARED_FAILED, 0,
                               "Failed to make shared in graph optimizer!");  // 0x3350000
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_OTHER_ANCHORS, 1,
                               "We will not traverse any other node!");  // 0x3350001
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_NOT_HEAVY_FORMAT, 2,
                               "Still Set other input or output but not traverse farther!");  // 0x3350002
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_WEIGHT, 3,
                               "We will not distribute from scalar weight!");  // 0x3350003
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_TENSOR, 4,
                               "We will not distribute from result op!");                              // 0x3350004
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE, 5, "We will not fuse two scope!");  // 0x3350005
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(SKIP_SUB_GRAPH_DATA_OR_NETOUTPUT, 6,
    "We will skip this kind of sub graph data or netoutput!");  // 0x3350006
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(CONTINUE_TO_SET_FORMAT, 6,
                               "We will continue to set format for this op!");  // 0x3350006
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(STOP_PROPAGATION_FROM_WEIGHT, 7,
                               "We will stop propagation from this weight!");  // 0x3350007
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(AI_CORE_GRAPH_PASS_OWNER_ERROR, 8,
                               "The owner of ai core pass is invalid!");  // 0x3350008
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(VECTOR_CORE_GRAPH_PASS_OWNER_ERROR, 9,
                               "The owner of vector core pass is invalid!");  // 0x3350009
FE_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_SUBGRAPH, 10,
                               "We shall not distribute from sub graph to father graph.");  // 0x3350010

/** op judge error code define */
FE_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_MAP_KEY_FIND_FAILED, 0, "Failed to find map key in op judge!");  // 0x3360000
FE_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_CHECK_FALSE_FAILED, 1, "Failed to check false in op judge!");    // 0x3360001
FE_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_OPSTORE_NOT_FOUND, 2, "Failed to find op store info!");          // 0x3360002
FE_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_SET_CORE_TYPE_FAILED, 3, "Failed to set op core type!");         // 0x3360003

/** op compiler error code define */
FE_DEF_ERRORNO_OP_COMPILER(OP_COMPILER_MAKE_SHARED_FAILED, 0, "Failed to make shared in op compiler!");  // 0x3370000
FE_DEF_ERRORNO_OP_COMPILER(OP_COMPILER_CHECK_FALSE_FAILED, 1, "Failed to check false in op compiler!");  // 0x3370001
FE_DEF_ERRORNO_OP_COMPILER(OP_COMPILER_L1_FUSION_FAILED, 2, "Failed to compile l1 fusion op!");          // 0x3370002

/** op kenel store error code define */
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

/** Task Builder error code define */
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_CREATE_ADAPTER_FAILED, 0, "Create task builder adapter failed!"); // 0x03390000
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_RUN_ADAPTER_FAILED, 1, "Run task builder adapter failed!");       // 0x03390001
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_STATUS_BAD_PARAM, 2, "Run task builder bad param!");              // 0x03390002
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_STATUS_INTERNAL_ERROR, 3, "Run task builder internal failed!");   // 0x03390003
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_STATUS_KERNEL_ERROR, 4, "Run task builder kernel failed!");       // 0x03390004
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_STATUS_RUNTIME_ERROR, 5, "Run task builder runtime failed!");     // 0x03390005
FE_DEF_ERRORNO_TASK_BUILDER(TASK_BUILDER_CALL_RT_FAILED, 6, "Failed to call runtime API!");                // 0x03390006

/** Graph matcher error code define */
FE_DEF_ERRORNO_GRAPH_MATCHER(GRAPH_MATCHER_GET_RULE_OUTPUT_NODE_FAILED, 0,
                             "Get rule output node failed!");  // 0x033A0000
/** Graph replace error code define */
FE_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_CREATE_FUSION_NODES_FAILED, 0,
                             "Graph Replace:Create fusion nodes failed!");  // 0x033B0000
FE_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_UPDATE_ATTR_FAILED, 1,
                             "Graph Replace:Update attributes failed!");  // 0x033B0001
FE_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_REPLACE_INPUT_FAILED, 2,
                             "Graph Replace:Replace input failed!");  // 0x033B0002
FE_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_REPLACE_OUTPUT_FAILED, 3,
                             "Graph Replace:Replace output failed!");  // 0x033B0003
FE_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_DELETE_NODE_FAILED, 4,
                             "Graph Replace:Delete node failed!");  // 0x033B0004
FE_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_CHECKSUPPORTED_FAILED, 5,
                             "Graph Replace:Check supported failed!");  // 0x033B0005

FE_DEF_ERRORNO_FUSION_RULE_PARSER(ILLEGAL_JSON, 0,
                                  "Illegal json expression, parse fusion rule "
                                  "from json failed!");  // 0x033C0000
FE_DEF_ERRORNO_FUSION_RULE_PARSER(INVALID_RULE, 1,
                                  "Exist FEOpKernelStore not supported op in fusion rule!");  // 0x033C0000
FE_DEF_ERRORNO_FUSION_RULE_PARSER(ILLEGAL_RULE, 2,
                                  "Illegal fusion rule structure, topological check failed!");  // 0x033C0002

FE_DEF_ERRORNO_OP_STORE_ADAPTER(OP_STORE_ADAPTER_MANAGER_INIT_FAILED, 0,
                                "Failed to init op store adapter manager.");  // 0x033D0000
FE_DEF_ERRORNO_OP_STORE_ADAPTER(OP_ADAPTER_TYPE_CHECK_FAILED, 1,
                                "Can not find the corresponding adapter type.");  // 0x033D0001
FE_DEF_ERRORNO_OP_STORE_ADAPTER(OP_STORE_ADAPTER_MAKE_SHARED_FAILED, 2,
                                "Failed to make shared in op store adapter.");                           // 0x033D0002
FE_DEF_ERRORNO_OP_STORE_ADAPTER(OP_ADAPTER_CHECK_FAILED, 3, "Can not find the corresponding adapter.");  // 0x033D0003
}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_FE_INNER_ERROR_CODES_H_
