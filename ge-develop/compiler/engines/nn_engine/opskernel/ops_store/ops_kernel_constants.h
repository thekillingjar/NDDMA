/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPSKERNEL_OPS_KERNEL_STORE_OPS_KERNEL_CONSTANTS_H_
#define FUSION_ENGINE_OPSKERNEL_OPS_KERNEL_STORE_OPS_KERNEL_CONSTANTS_H_

#include <string>
#include "common/aicore_util_types.h"
#include "common/aicore_util_constants.h"

namespace fe {
const int32_t COMPUTE_COST_DEFAULT = 10;

constexpr char const *STR_INT = "int";
constexpr char const *STR_FLOAT = "float";
constexpr char const *STR_BOOL = "bool";
constexpr char const *STR_STR = "str";
constexpr char const *STR_LIST_INT = "listInt";
constexpr char const *STR_LIST_FLOAT = "listFloat";
constexpr char const *STR_LIST_BOOL = "listBool";
constexpr char const *STR_LIST_STR = "listStr";
constexpr char const *STR_LIST_LIST_INT = "listListInt";

constexpr char const *STR_NAME = "name";
constexpr char const *STR_INPUT_LOWERCASE = "input";
constexpr char const *STR_OUTPUT_LOWERCASE = "output";
constexpr char const *STR_INPUT_FIRST_LETTER_UPPERCASE = "Input";
constexpr char const *STR_OUTPUT_FIRST_LETTER_UPPERCASE = "Output";
constexpr char const *STR_OP_FILE = "opFile";
constexpr char const *STR_OP_INTERFACE = "opInterface";
constexpr char const *STR_PRECISION_POLICY = "precision_reduce";
constexpr char const *STR_RESHAPE_TYPE = "reshapeType";
constexpr char const *STR_FLAG = "flag";
constexpr char const *STR_OP = "op";
constexpr char const *STR_PARAM_TYPE = "paramType";
constexpr char const *STR_TUNEFORMAT_SWITCH = "tuneformatSwitch";
constexpr char const *STR_CONST_VALUE_DEPEND = "valueDepend";
constexpr char const *STR_DTYPE = "dtype";
constexpr char const *STR_FORMAT = "format";
constexpr char const *STR_SUB_FORMAT = "sub_format";
constexpr char const *STR_UNKNOWN_SHAPE_FORMAT = "unknownshape_format";
constexpr char const *STR_FALLBACK = "fallback";
constexpr char const *STR_ENABLE = "enable";
constexpr char const *STR_UNKNOWN_SHAPE_ENABLE = "unknownshape_enable";
constexpr char const *STR_ALL = "all";
constexpr char const *STR_LIST = "list";
constexpr char const *STR_REQUIRED = "required";
constexpr char const *STR_ATTR = "attr";
constexpr char const *STR_ATTR_PREFIX = "attr_";
constexpr char const *STR_TYPE = "type";
constexpr char const *STR_VALUE = "value";
constexpr char const *STR_DEFAULT_VALUE = "defaultValue";
constexpr char const *STR_PATTERN = "pattern";
constexpr char const *STR_HEAVY_OP = "heavyOp";
constexpr char const *STR_SLICE_PATTERN = "slicePattern";
constexpr char const *STR_IMP_PATH = "imp_path";
constexpr char const *STR_PATH = "path";
constexpr char const *kStrNeedCheckSupport = "needCheckSupport";
constexpr char const *kStrDynamicFormat = "dynamicFormat";
constexpr char const *kStrDynamicShapeSupport = "dynamicShapeSupport";
constexpr char const *kStrDynamicRankSupport = "dynamicRankSupport";
constexpr char const *kStrInputMemContinues = "inputMemContinues";
constexpr char const *kStrOutputMemContinues = "outputMemContinues";
constexpr char const *kStrEnableVectorCore = "enableVectorCore";
constexpr char const *kStrDynamicCompileStatic = "dynamicCompileStatic";
constexpr char const *kStrOpImplSwitch = "opImplSwitch";
constexpr char const *kStrJitCompile = "jitCompile";
constexpr char const *kSoftSync = "softsync";
constexpr char const *kCoreType = "coreType";
constexpr char const *kNullOpFile = "Null";
constexpr char const *kAicoreNull = "AICore_Null";
constexpr char const *kAclnnSupport = "aclnnSupport";
constexpr char const *kMultiKernelSupportDynamicGraph = "multiKernelSupportDynamicGraph";

const std::map<std::string, bool> kStrBoolMap {{kStrTrue, true}, {kStrFalse, false}};

// maps utilized to transform string to enum
const std::map<std::string, OpParamType> kParamTypeMap{
        {"dynamic", DYNAMIC}, {"optional", OPTIONAL}, {"required", REQUIRED}, {"reserved", RESERVED}};

const std::map<std::string, bool> kBoolToStringMap{
        {"true", true}, {"false", false}};

const std::map<std::string, OpConstValueDepend> kConstValueDependMap{
        {"required", CONST_REQUIRED}, {"optional", CONST_OPTIONAL}};

const std::map<std::string, int64_t> kOpPatternMap {
        {"formatagnostic", static_cast<int64_t>(OP_PATTERN_FORMAT_AGNOSTIC)},
        {"broadcast", static_cast<int64_t>(OP_PATTERN_BROADCAST)},
        {"reduce", static_cast<int64_t>(OP_PATTERN_REDUCE)},
        {"dynamic", static_cast<int64_t>(OP_PATTERN_OP_CUSTOMIZE)},
        {"rangeagnostic", static_cast<int64_t>(OP_PATTERN_RANGE_AGNOSTIC)},
        {"broadcastenhanced", static_cast<int64_t>(OP_PATTERN_BROADCAST_ENHANCED)}
};

const std::map<std::string, int64_t> kStrSlicePatternMap{
        {"elemwise", static_cast<int64_t>(ELEMENT_WISE)},
        {"elemwisebroadcast", static_cast<int64_t>(ELEMENT_WISE_BROADCAST)},
        {"broadcast", static_cast<int64_t>(BROADCAST)},
        {"slidingwindow", static_cast<int64_t>(SLIDING_WINDOW)},
        {"slidingwindowdeconv", static_cast<int64_t>(SLIDING_WINDOW_DECONV)},
        {"cubematmul", static_cast<int64_t>(CUBE_MATMUL)},
        {"reduce", static_cast<int64_t>(SLICE_PATTERN_REDUCE)},
        {"resize", static_cast<int64_t>(SLICE_PATTERN_RESIZE)},
        {"scatter", static_cast<int64_t>(SLICE_PATTERN_SCATTER)},
        {"segment", static_cast<int64_t>(SLICE_PATTERN_SEGMENT)}
};

const std::map<std::string, int64_t> kStrPrecisionPolicyMap{
        {kStrTrue, static_cast<int64_t>(WHITE)},
        {kStrFalse, static_cast<int64_t>(BLACK)}
};

const std::map<std::string, int64_t> kStrJitCompileMap {
        {"true", static_cast<int64_t>(JitCompile::ONLINE)},
        {"false", static_cast<int64_t>(JitCompile::REUSE_BINARY)},
        {"static_true", static_cast<int64_t>(JitCompile::ONLINE)},
        {"static_false", static_cast<int64_t>(JitCompile::STATIC_BINARY_DYNAMIC_ONLINE)},
        {"dynamic_true", static_cast<int64_t>(JitCompile::ONLINE)},
        {"dynamic_false", static_cast<int64_t>(JitCompile::REUSE_BINARY)},
        {"static_true,dynamic_true", static_cast<int64_t>(JitCompile::ONLINE)},
        {"static_true,dynamic_false", static_cast<int64_t>(JitCompile::REUSE_BINARY)},
        {"static_false,dynamic_true", static_cast<int64_t>(JitCompile::STATIC_BINARY_DYNAMIC_ONLINE)},
        {"static_false,dynamic_false", static_cast<int64_t>(JitCompile::STATIC_BINARY_DYNAMIC_BINARY)},
        {"dynamic_true,static_true", static_cast<int64_t>(JitCompile::ONLINE)},
        {"dynamic_true,static_false", static_cast<int64_t>(JitCompile::STATIC_BINARY_DYNAMIC_ONLINE)},
        {"dynamic_false,static_true", static_cast<int64_t>(JitCompile::REUSE_BINARY)},
        {"dynamic_false,static_false", static_cast<int64_t>(JitCompile::STATIC_BINARY_DYNAMIC_BINARY)},
};

const std::map<std::string, int64_t> kStrDynamicRankTypeMap {
        {"true", static_cast<int64_t>(DynamicRankType::SUPPORT)},
        {"false", static_cast<int64_t>(DynamicRankType::NOT_SUPPORT)},
        {"upgrade_to_dynamic_rank", static_cast<int64_t>(DynamicRankType::UPGRADE_TO_SUPPORT)}
};

const std::map<std::string, int64_t> kStrRangeLimitTypeMap {
        {"limited", static_cast<int64_t>(RangeLimitType::LIMITED)},
        {"unlimited", static_cast<int64_t>(RangeLimitType::UNLIMITED)},
        {"dynamic", static_cast<int64_t>(RangeLimitType::DYNAMIC)}
};

const std::map<std::string, int64_t> kStrDynamicCompileStaticMap {
        {"tune", static_cast<int64_t>(DynamicCompileStatic::TUNE)},
        {"true", static_cast<int64_t>(DynamicCompileStatic::COMPILE)},
        {"false", static_cast<int64_t>(DynamicCompileStatic::NOT_SUPPORT)}
};

const std::map<std::string, int64_t> kAclnnSupportStrParamToEnumMap {
        {"default", static_cast<int64_t>(AclnnSupportType::DEFAULT)},
        {"support_aclnn", static_cast<int64_t>(AclnnSupportType::SUPPORT_ACLNN)},
        {"aclnn_only", static_cast<int64_t>(AclnnSupportType::ACLNN_ONLY)}
};

const std::map<std::string, int64_t> kMultiKernelSupportStrParamToEnumMap {
    {"default", static_cast<int64_t>(MultiKernelSupportType::DEFAULT)},
    {"multi_kernel", static_cast<int64_t>(MultiKernelSupportType::MULTI_SUPPORT)},
    {"single_kernel", static_cast<int64_t>(MultiKernelSupportType::SINGLE_SUPPORT)}
};

const std::map<std::string, int64_t> kEnableVectorCoreStrParamToEnumMap {
        {"true", static_cast<int64_t>(VectorCoreType::ENABLE)},
        {"false", static_cast<int64_t>(VectorCoreType::DISABLE)}
};
}
#endif  // FUSION_ENGINE_OPSKERNEL_OPS_KERNEL_STORE_OPS_KERNEL_CONSTANTS_H_
