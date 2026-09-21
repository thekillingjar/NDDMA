/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_INC_TENSOR_ENGINE_FUSION_TYPES_H_
#define ATC_OPCOMPILER_INC_TENSOR_ENGINE_FUSION_TYPES_H_

#include <string>
#include "graph/op_desc.h"

namespace te {
enum TbeMemType {
    TBE_MEMORY_DDR = 0,               /**< DDR memory on device */
    TBE_MEMORY_L1 = 1,               /**< L1 memory on device */
    TBE_MEMORY_L2 = 2,               /**< L2 memory on device */
    TBE_MEMORY_UB = 3,               /**< UB memory on device */
    TBE_MEMORY_RESERVED,
};

enum ATTR_DTYPE {
    ATTR_INT8             = 0,
    ATTR_UINT8            = 1,
    ATTR_INT16            = 2,
    ATTR_UINT16           = 3,
    ATTR_INT32            = 4,
    ATTR_UINT32           = 5,
    ATTR_INT64            = 6,
    ATTR_UINT64           = 7,
    ATTR_FLOAT32          = 8,
    ATTR_DOUBLE           = 9,
    ATTR_BOOL             = 10,
    ATTR_STR              = 11,
    ATTR_LIST_INT8        = 12,
    ATTR_LIST_UINT8       = 13,
    ATTR_LIST_INT16       = 14,
    ATTR_LIST_UINT16      = 15,
    ATTR_LIST_INT32       = 16,
    ATTR_LIST_UINT32      = 17,
    ATTR_LIST_INT64       = 18,
    ATTR_LIST_UINT64      = 19,
    ATTR_LIST_FLOAT32     = 20,
    ATTR_LIST_DOUBLE      = 21,
    ATTR_LIST_BOOL        = 22,
    ATTR_LIST_STR         = 23,
    ATTR_LIST_LIST_INT64  = 24,
    ATTR_LIST_LIST_FLOAT  = 25,

    // illegal type which can't be fused
    ATTR_MAX
};

enum ATTR_SHAPETYPE {
    ATTR_SHAPE_LIST     = 0,
    ATTR_SHAPE_TUPLE    = 1,
    ATTR_SHAPE_TYPE_MAX = ATTR_SHAPE_TUPLE,
};

enum LX_QUERY_STATUS {
    LX_QUERY_FAIL       = -1,
    LX_QUERY_NOT_FOUND  = 0,
    LX_QUERY_SUCC       = 1
};

enum CheckSupportedResult {
    NOT_SUPPORTED = 0,
    FULLY_SUPPORTED = 1,
    PARTIALLY_SUPPORTED = 2
};

enum OP_VALUE_DEPEND {
    VALUE_DEPEND_IGNORE = 0,
    VALUE_DEPEND_REQUIRED,
    VALUE_DEPEND_OPTIONAL
};

enum TE_GENERALIZE_TYPE {
    REGISTER_FUNC = 0,
    DEFAULT_TBE_OP_INFO,             // default generalize rule, node range unlimited, info get from tbeOpInfo
    DEFAULT_NODE,                    // default generalize rule, node info get from node (no op store)
    DEFAULT_LIMITED_TBE_OP_INFO,     // default generalize rule, node in limited graph, info get from tbeOpInfo
    GENERALIZE_TYPE_MAX
};

/**
 * @brief jit compile defination in op info config
 *
 */
enum class JitCompileType {
    DEFAULT = 0,
    ONLINE,     // static_true, dynamic_true
    REUSE_BINARY,   // static_true, dynamic_false
    STATIC_BINARY_DYNAMIC_ONLINE,   // static_false, dynamic_true
    STATIC_BINARY_DYNAMIC_BINARY    // static_false, dynamic_false
};

enum class DynamicRankType {
    NOT_SUPPORT = 0,
    SUPPORT,
    UPGRADE_TO_SUPPORT
};

enum class RangeLimitType {
    UNLIMITED = 0,
    LIMITED,
    DYNAMIC
};

enum class VectorCoreType {
    DEFAULT = 0,
    ENABLE,
    DISABLE
};

/** TT_REQ: request tensor, just one tensor in one input
  * TT_OPT: option tensor, one or zero tensor in one input
  * TT_DYN: dynamic tensor, one or more tensors in one input
  */
enum TensorType {
    TT_REQ,
    TT_OPT,
    TT_DYN,
};

enum OpBuildResCode {
    OP_BUILD_SUCC,
    OP_BUILD_FAIL,
    OP_DYNSHAPE_NOT_SUPPORT
};

enum BUILD_TYPE {
    INITIALLY_BUILD    = 0x0,
    FUZZILY_BUILD      = 0x1,
    ACCURATELY_BUILD   = 0x2,
};

enum class DdrBaseType {
    NONE = -1,
    WORKSPACE = 0,
    WEIGHT = 1,
    NET_EDGE = 2  // Data or NetOutput
};

/**
 * @brief: finished compilation task struce
 */
struct FinComTask {
    // indicate which thread exec compilation
    uint64_t graphId;
    // task id
    uint64_t taskId;
    // task status: 0 indicate success, others indicate fail
    int status;
    // error message
    std::string errMsg;
    // OpDesc with json path writen
    ge::OpDescPtr teNodeOpDesc;
};

struct CheckSupportedInfo {
    // required
    CheckSupportedResult isSupported = NOT_SUPPORTED;
    // optional
    std::string reason;
    // optional
    bool dynamicCompileStatic = false;
    // optional
    bool allImplChecked = false;
};
}
#endif  // ATC_OPCOMPILER_INC_TENSOR_ENGINE_FUSION_TYPES_H_