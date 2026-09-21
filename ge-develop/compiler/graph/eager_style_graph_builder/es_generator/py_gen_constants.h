/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GEN_CONSTANTS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GEN_CONSTANTS_H_

namespace ge {
namespace es {

namespace PyGenConstants {
// 字符串常量
constexpr const char *kPythonGeneratorName = "Python generator";
constexpr const char *kCommentStart = "\"\"\"";
constexpr const char *kCommentLinePrefix = " # ";
constexpr const char *kCommentEnd = "\"\"\"";
constexpr const char *kOutputSuffix = "Output";
constexpr const char *kCtypesPrefix = "C";
constexpr const char *kPerOpFilePrefix = "es_";
constexpr const char *kPerOpFileSuffix = ".py";
constexpr const char *kOwnerGraphBuilder = "owner_graph_builder";
constexpr const char *kOwnerBuilder = "owner_builder";

// 类型相关常量
constexpr const char *kTensorHolderType = "TensorHolder";
constexpr const char *kTensorLikeType = "TensorLike";
constexpr const char *kTensorHolderPtrType = "EsCTensorHolderPtr";
constexpr const char *kGraphBuilderType = "GraphBuilder";
constexpr const char *kGraphBuilderPtrType = "EsCGraphBuilderPtr";
constexpr const char *kGraphType = "Graph";
constexpr const char *kGraphPtrType = "ctypes.POINTER(EsCGraph)";
constexpr const char *kDynamicGraphPtrType = "ctypes.POINTER(ctypes.POINTER(EsCGraph))";
constexpr const char *kEsCTensorPtrType = "EsCTensorPtr";
constexpr const char *kCtypesVoidPtr = "ctypes.c_void_p";
constexpr const char *kCtypesInt64 = "ctypes.c_int64";
constexpr const char *kCtypesCharP = "ctypes.c_char_p";
constexpr const char *kCtypesFloat = "ctypes.c_float";
constexpr const char *kCtypesBool = "ctypes.c_bool";
constexpr const char *kCtypesInt = "ctypes.c_int";
constexpr const char *kEsCTensor = "EsCTensor";

// 字符串处理常量
constexpr const char *kTrueValue = "true";
constexpr const char *kFalseValue = "false";
constexpr const char *kPythonTrue = "True";
constexpr const char *kPythonFalse = "False";
constexpr const char *kEmptyList = "[]";
constexpr const char *kNoneValue = "None";
constexpr const char *kDataTypePrefix = "DataType.";
constexpr const char *kGePrefix = "ge::";
constexpr const char *kUtf8Encoding = "utf-8";

// 符号常量
constexpr const char *kLeftCurlyBracket = "{";
constexpr const char *kRightCurlyBracket = "}";
constexpr const char *kLeftBracket = "[";
constexpr const char *kRightBracket = "]";

// 数值常量
constexpr size_t kGePrefixLength = 4;  // ge::
constexpr size_t kTrueLength = 4;
constexpr size_t kFalseLength = 5;
}  // namespace PyGenConstants

}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GEN_CONSTANTS_H_