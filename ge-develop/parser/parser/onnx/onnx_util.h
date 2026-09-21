/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_ONNX_ONNX_UTIL_PARSER_H_
#define PARSER_ONNX_ONNX_UTIL_PARSER_H_

#include <string>
#include <cstdint>
#include "graph/types.h"

namespace OnnxDataType {
enum OnnxDataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
  FLOAT8E4M3FN = 17,
  FLOAT8E4M3FNUZ = 18,
  FLOAT8E5M2 = 19,
  FLOAT8E5M2FNUZ = 20,
};
}

namespace ge {
const char *const kAttrNameValue = "value";
const char *const kAttrNameInput = "input_tensor";
const char *const kAttrNameIndex = "index";
const char *const kAttrNameIsSubgraphOp = "is_subgraph_op";
const char *const kOpTypeConstant = "Constant";
const char *const kOpTypeInput = "Input";
const char *const kFileConstant = "FileConstant";

class OnnxUtil {
 public:
  static ge::DataType ConvertOnnxDataType(int64_t onnx_data_type);
  static void GenUniqueSubgraphName(int subgraph_index, const std::string &original_subgraph_name,
                                    const std::string &parent_node_name, std::string &unique_subgraph_name);
  static std::string GenUniqueNodeName(const std::string &graph_name, const std::string &node_name);
};
}  // namespace ge

#endif // PARSER_ONNX_ONNX_UTIL_PARSER_H_
