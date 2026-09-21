/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_INC_FFTS_ERROR_CODES_H_
#define FFTS_ENGINE_INC_FFTS_ERROR_CODES_H_

#include <cstdint>
#include <string>

namespace ffts {
/** Assigned SYS ID */
const uint8_t SYSID_FFTS = 18U;

/** Common module ID */
const uint8_t FFTS_MODID_COMMON = 50U;
/** Itf module ID */
const uint8_t FFTS_MODID_ITF = 51U;
/** Shape Format Transfer module ID */
const uint8_t FFTS_MODID_SHAPE_FORMAT_TRANSFER = 52U;
/** IGraph Optimizer module ID */
const uint8_t FFTS_MODID_GRAPH_OPTIMIZER = 53U;
/** Op Judge module ID */
const uint8_t FFTS_MODID_OP_JUDGE = 54U;
/** Op Compiler module ID */
const uint8_t FFTS_MODID_OP_COMPILER = 55U;
/** Op Kernel Store module ID */
const uint8_t FFTS_MODID_OP_KERNEL_STORE = 56U;
/** Graph matcher module ID */
const uint8_t FFTS_MODID_GRAPH_MATCHER = 58U;
/** Graph replace module ID */
const uint8_t FFTS_MODID_GRAPH_REPLACE = 59U;
/** Fusion rule parser module ID */
const uint8_t FFTS_MODID_FUSION_RULE_PARSER = 60U;
/** Op store adapter module ID */
const uint8_t FFTS_MODID_OP_STORE_ADAPTER = 61U;
/** Fusion rule parser module ID */
const uint8_t FFTS_MODID_OP_CALCULATOR = 62U;
// LX Fusion module ID
const uint8_t FFTS_MODID_LX_FUSION = 63U;

using Status = uint32_t;

#define FFTS_DEF_ERRORNO(sysid, modid, name, value, desc)                            \
  static constexpr Status name =                                               \
      ((((static_cast<uint32_t>((0xFF) & (static_cast<uint8_t>(sysid)))) << 24) |  \
       ((static_cast<uint32_t>((0xFF) & (static_cast<uint8_t>(modid)))) << 16)) |  \
       ((0xFFFF) & (static_cast<uint16_t>(value))));

#define FFTS_DEF_ERRORNO_COMMON(name, value, desc)                  \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_COMMON, name, value, desc)

FFTS_DEF_ERRORNO(0, 0, SUCCESS, 0, "success");
FFTS_DEF_ERRORNO(0xFFU, 0xFFU, FAILED, 0xFFFFU, "failed");
FFTS_DEF_ERRORNO_COMMON(PARAM_INVALID, 1, "Parameter's invalid!");
FFTS_DEF_ERRORNO_COMMON(NOT_CHANGED, 2, "Not changed!");

#define FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_SHAPE_FORMAT_TRANSFER, name, value, desc)
#define FFTS_DEF_ERRORNO_ITF(name, value, desc) FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_ITF, name, value, desc)
#define FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_GRAPH_OPTIMIZER, name, value, desc)
#define FFTS_DEF_ERRORNO_OP_JUDGE(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_OP_JUDGE, name, value, desc)
#define FFTS_DEF_ERRORNO_OP_COMPILER(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_OP_COMPILER, name, value, desc)
#define FFTS_DEF_ERRORNO_OP_KERNEL_STORE(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_OP_KERNEL_STORE, name, value, desc)
#define FFTS_DEF_ERRORNO_GRAPH_MATCHER(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_GRAPH_MATCHER, name, value, desc)
#define FFTS_DEF_ERRORNO_GRAPH_REPLACE(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_GRAPH_REPLACE, name, value, desc)
#define FFTS_DEF_ERRORNO_FUSION_RULE_PARSER(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_FUSION_RULE_PARSER, name, value, desc)
#define FFTS_DEF_ERRORNO_OP_STORE_ADAPTER(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_OP_STORE_ADAPTER, name, value, desc)
#define FFTS_DEF_ERRORNO_OP_CALCULATOR(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_OP_CALCULATOR, name, value, desc)
#define FFTS_DEF_ERRORNO_LX_FUSION(name, value, desc) \
  FFTS_DEF_ERRORNO(SYSID_FFTS, FFTS_MODID_LX_FUSION, name, value, desc)

/** common module error code define */
FFTS_DEF_ERRORNO_COMMON(MEMALLOC_FAILED, 0, "Failed to allocate memory!");    // 0x3320000
FFTS_DEF_ERRORNO_COMMON(CALL_CCE_FAILED, 2, "Failed to call CCE API!");       // 0x3320002
FFTS_DEF_ERRORNO_COMMON(CALL_RT_FAILED, 3, "Failed to call runtime API!");    // 0x3320003
FFTS_DEF_ERRORNO_COMMON(INTERNAL_ERROR, 4, "Internal errors");                // 0x3320004
FFTS_DEF_ERRORNO_COMMON(CALL_CSEC_ERROR, 5, "Failed to call libc_sec API!");  // 0x3320005
FFTS_DEF_ERRORNO_COMMON(CALL_TEE_ERROR, 6, "Failed to call tee API!");        // 0x3320006
FFTS_DEF_ERRORNO_COMMON(FILE_NOT_EXIST, 8, "The file does not exist.");       // 0x3320007

FFTS_DEF_ERRORNO_COMMON(TENSOR_FORMAT_NOT_FOUND, 18, "This format has not been found.");        // 0x3320018
FFTS_DEF_ERRORNO_COMMON(INVALID_TENSOR_FORMAT, 19, "This format is not valid.");                // 0x3320019
FFTS_DEF_ERRORNO_COMMON(INVALID_TENSOR_DATATYPE, 20, "This data type is not valid.");           // 0x3320020
FFTS_DEF_ERRORNO_COMMON(INVALID_DIM_VALUE, 21, "The dim value must be great than zero.");       // 0x3320021
FFTS_DEF_ERRORNO_COMMON(INVALID_DIM_SIZE, 22, "The size of dim must be greater than 4.");  // 0x3320022

FFTS_DEF_ERRORNO_COMMON(INVALID_NC1KHKWHWC0_SIZE, 23, "The size of NC1KHKWHWC0 format is not valid.");  // 0x3320023
FFTS_DEF_ERRORNO_COMMON(INVALID_C1HWNCoC0_SIZE, 24, "The size of C1HWNCoC0 format is not valid.");      // 0x3320024
FFTS_DEF_ERRORNO_COMMON(NOT_SUPPORT_TENSOR_FORMAT, 25, "This data type is not supported.");             // 0x3320025
FFTS_DEF_ERRORNO_COMMON(NOT_SUPPORT_TENSOR_DATATYPE, 26, "This tensor format is not supported.");       // 0x3320026
FFTS_DEF_ERRORNO_COMMON(BEYONG_MAX_TENSOR_ELEMENT_COUNT, 27,
"The amount of tensor element is more than the max amount.");  // 0x3320027

FFTS_DEF_ERRORNO_COMMON(VECTOR_INT64_EMPTY, 30, "The vector of int64 number is empty.");                  // 0x3320030
FFTS_DEF_ERRORNO_COMMON(ADD_OVERFLOW_INT64, 31, "The addition between int64 number is overflow.");        // 0x3320031
FFTS_DEF_ERRORNO_COMMON(MUL_OVERFLOW_INT64, 32, "The multiplication between int64 number is overflow.");  // 0x3320032
FFTS_DEF_ERRORNO_COMMON(ADD_OVERFLOW_SIZET, 33, "The addition between size_t number is overflow.");       // 0x3320033
FFTS_DEF_ERRORNO_COMMON(CONTINUING_TRANSFORMAT, 34,
"We need to transformat in this case of one dimensional shape padding.");             // 0x3320034

/** shape format transfer error code define */
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(SHAPE_FORMAT_TRANSFER_SORTING_FAILED, 0, "Failed to sort!");      // 0x3340000
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(SHAPE_FORMAT_TRANSFER_INSERTING_FAILED, 1, "Failed to insert!");  // 0x3340001
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(SHAPE_FORMAT_TRANSFER_CHECKING_FAILED, 2, "Failed to check!");    // 0x3340002
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(CREATE_CAST_OP_FAILED, 3, "Failed to create CAST op!");           // 0x3340003
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(ADD_CAST_OP_NODE_FAILED, 4, "Failed to add CAST op node!");       // 0x3340004
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(RESHAPE_TYPE_NOT_SUPPORTED, 5,
                                     "Failed to insert Reshape! Reshape type is invalid!");  // 0x3340005
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(RESHAPE_NOT_NECESSARY, 6,
                                     "Failed to insert Reshape! Uncessary to reshape!");  // 0x3340006
FFTS_DEF_ERRORNO_SHAPE_FORMAT_TRANSFER(MERGE_TRANS_OP_NO_MORE_PREDECESSOR, 7,
                                     "There is no more trans op for merging!");  // 0x3340007

/** fusion manager error code define */
FFTS_DEF_ERRORNO_ITF(OPINFO_STORES_INIT_FAILED, 0, "Failed to init opinfo_kernel_stores!");      // 0x3330000
FFTS_DEF_ERRORNO_ITF(GRAPH_OPTIMIZER_INIT_FAILED, 1, "Failed to init graphoptimizer!");          // 0x3330001
FFTS_DEF_ERRORNO_ITF(OPINFO_STORES_FINI_FAILED, 2, "Failed to finilize opinfo_kernel_stores!");  // 0x3330002
FFTS_DEF_ERRORNO_ITF(GRAPHOPTIMIZER_FINI_FAILED, 3, "Failed to finilize graphoptimizer!");       // 0x3330003
FFTS_DEF_ERRORNO_ITF(CONFIGURATION_INIT_FAILED, 4, "Failed to initialize configuration!");       // 0x3330004

/** graph optimizer error code define */
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_MAKE_SHARED_FAILED, 0,
                               "Failed to make shared in graph optimizer!");  // 0x3350000
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_OTHER_ANCHORS, 1,
                               "We will not traverse any other node!");  // 0x3350001
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_NOT_HEAVY_FORMAT, 2,
                               "Still Set other input or output but not traverse farther!");  // 0x3350002
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_WEIGHT, 3,
                               "We will not distribute from scalar weight!");  // 0x3350003
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_TENSOR, 4,
                               "We will not distribute from result op!");                              // 0x3350004
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE, 5, "We will not fuse two scope!");  // 0x3350005
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(SKIP_SUB_GRAPH_DATA_OR_NETOUTPUT, 6,
                               "We will skip this kind of sub graph data or netoutput!");  // 0x3350006
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(CONTINUE_TO_SET_FORMAT, 6,
                               "We will continue to set format for this op!");  // 0x3350006
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(STOP_PROPAGATION_FROM_WEIGHT, 7,
                               "We will stop propagation from this weight!");  // 0x3350007
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(AI_CORE_GRAPH_PASS_OWNER_ERROR, 8,
                               "The owner of ai core pass is invalid!");  // 0x3350008
FFTS_DEF_ERRORNO_GRAPH_OPTIMIZER(VECTOR_CORE_GRAPH_PASS_OWNER_ERROR, 9,
                               "The owner of vector core pass is invalid!");  // 0x3350009

/** op judge error code define */
FFTS_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_MAP_KEY_FIND_FAILED, 0, "Failed to find map key in op judge!");  // 0x3360000
FFTS_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_CHECK_FALSE_FAILED, 1, "Failed to check false in op judge!");    // 0x3360001
FFTS_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_OPSTORE_NOT_FOUND, 2, "Failed to find op store info!");          // 0x3360002
FFTS_DEF_ERRORNO_OP_JUDGE(OP_JUDGE_SET_CORE_TYPE_FAILED, 3, "Failed to set op core type!");         // 0x3360003

/** op compiler error code define */
FFTS_DEF_ERRORNO_OP_COMPILER(OP_COMPILER_MAKE_SHARED_FAILED, 0, "Failed to make shared in op compiler!");  // 0x3370000
FFTS_DEF_ERRORNO_OP_COMPILER(OP_COMPILER_CHECK_FALSE_FAILED, 1, "Failed to check false in op compiler!");  // 0x3370001
FFTS_DEF_ERRORNO_OP_COMPILER(OP_COMPILER_L1_FUSION_FAILED, 2, "Failed to compile l1 fusion op!");          // 0x3370002

/** op kenel store error code define */
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_PARSE_FAILED, 0, "Failed to parse the op kernel store cfg file!");  // 0x3380000
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_MAP_KEY_FIND_FAILED, 1,
                               "Failed to find the map key in FEOpKernelStore!");  // 0x3380001
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_STRING_CONVERT_FAILED, 2,
                               "Failed to convert string in FEOpKernelStore!");                            // 0x3380002
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_READ_CFG_FILE_FAILED, 3, "Failed to read cfg file. I/O failed!");  // 0x3380003
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_CFG_FILE_EMPTY, 4,
                               "Failed to read cfg file. Empty configuration file path!");     // 0x3380004
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_CFG_NAME_EMPTY, 5, "Failed to get sub store name!");  // 0x3380005
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_CFG_FILE_NOT_EXIST, 6,
                               "Failed to read op information. Configuration file is not exist!");         // 0x3380006
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_STORE_MAKE_SHARED_FAILED, 7, "Failed to make shared in ops store!");   // 0x3380006
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OPS_SUB_STORE_NOT_EXIST, 8, "Failed to find specific sub store!");        // 0x3380007
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OPS_SUB_STORE_PTR_NULL, 9, "Failed to get sub store pointer!");           // 0x3380009
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL, 10,
                               "Failed to find op in all sub stores!");  // 0x33800A
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_ATTR_NOT_FOUND_IN_OP_KERNEL_INFO, 11,
                               "Failed to find attr name in op kernel info!");  // 0x33800B
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_ATTR_EMPTY_IN_OP_KERNEL_INFO, 12,
                               "None attribute found in op kernel info!");  // 0x33800C
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_INPUT_NOT_FOUND_IN_OP_KERNEL_INFO, 13,
                               "Failed to find input info in op kernel info!");  // 0x33800D
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_OUTPUT_NOT_FOUND_IN_OP_KERNEL_INFO, 14,
                               "Failed to find output info in op kernel info!");  // 0x33800E
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_NOT_FOUND_IN_GET_HIGH_PRIO_OP_KERNEL, 15,
                               "Failed to find op in all sub stores in GetHighPrioOpKernelInfoPtr!");  // 0x33800F
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_KERNEL_INFO_NULL_PTR, 16, "Param is null ptr!");  // 0x338010
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_SUB_STORE_PLGUIN_INIT_FAILED, 17,
                               "Failed to init plugin tbe sub op store!");  // 0x338011
FFTS_DEF_ERRORNO_OP_KERNEL_STORE(OP_SUB_STORE_ILLEGAL_JSON, 18,
                               "Illegal json file, fail to parse json file!");  // 0x338018

/** Graph matcher error code define */
FFTS_DEF_ERRORNO_GRAPH_MATCHER(GRAPH_MATCHER_GET_RULE_OUTPUT_NODE_FAILED, 0,
                             "Get rule output node failed!");  // 0x033A0000
/** Graph replace error code define */
FFTS_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_CREATE_FUSION_NODES_FAILED, 0,
                             "Graph Replace:Create fusion nodes failed!");  // 0x033B0000
FFTS_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_UPDATE_ATTR_FAILED, 1,
                             "Graph Replace:Update attributes failed!");  // 0x033B0001
FFTS_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_REPLACE_INPUT_FAILED, 2,
                             "Graph Replace:Replace input failed!");  // 0x033B0002
FFTS_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_REPLACE_OUTPUT_FAILED, 3,
                             "Graph Replace:Replace output failed!");  // 0x033B0003
FFTS_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_DELETE_NODE_FAILED, 4,
                             "Graph Replace:Delete node failed!");  // 0x033B0004
FFTS_DEF_ERRORNO_GRAPH_REPLACE(GRAPH_REPLACE_CHECKSUPPORTED_FAILED, 5,
                             "Graph Replace:Check supported failed!");  // 0x033B0005

FFTS_DEF_ERRORNO_FUSION_RULE_PARSER(ILLEGAL_JSON, 0,
                                  "Illegal json expression, parse fusion rule "
                                  "from json failed!");  // 0x033C0000
FFTS_DEF_ERRORNO_FUSION_RULE_PARSER(INVALID_RULE, 1,
                                  "Exist FEOpKernelStore not supported op in fusion rule!");  // 0x033C0000
FFTS_DEF_ERRORNO_FUSION_RULE_PARSER(ILLEGAL_RULE, 2,
                                  "Illegal fusion rule structure, topological check failed!");  // 0x033C0002

FFTS_DEF_ERRORNO_OP_STORE_ADAPTER(OP_STORE_ADAPTER_MANAGER_INIT_FAILED, 0,
                                "Failed to init op store adapter manager.");  // 0x033D0000
FFTS_DEF_ERRORNO_OP_STORE_ADAPTER(OP_ADAPTER_TYPE_CHECK_FAILED, 1,
                                "Can not find the corresponding adapter type.");  // 0x033D0001
FFTS_DEF_ERRORNO_OP_STORE_ADAPTER(OP_STORE_ADAPTER_MAKE_SHARED_FAILED, 2,
                                "Failed to make shared in op store adapter.");                           // 0x033D0002
FFTS_DEF_ERRORNO_OP_STORE_ADAPTER(OP_ADAPTER_CHECK_FAILED, 3, "Can not find the corresponding adapter.");  // 0x033D0003

FFTS_DEF_ERRORNO_OP_CALCULATOR(FAIL_GET_OP_IMPL_TYPE, 0, "Failed to get the op impl type of op desc.");  // 0x033

const std::string EM_INNER_ERROR = "E29999";
}
#endif  // FFTS_ENGINE_INC_FFTS_ERROR_CODES_H_
