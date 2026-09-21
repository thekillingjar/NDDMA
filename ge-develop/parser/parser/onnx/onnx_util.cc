/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_util.h"
#include <map>

namespace {
const std::map<uint32_t, ge::DataType> onnx_data_type_map = {
    {OnnxDataType::UNDEFINED, ge::DataType::DT_UNDEFINED}, {OnnxDataType::FLOAT, ge::DataType::DT_FLOAT},
    {OnnxDataType::UINT8, ge::DataType::DT_UINT8},         {OnnxDataType::INT8, ge::DataType::DT_INT8},
    {OnnxDataType::UINT16, ge::DataType::DT_UINT16},       {OnnxDataType::INT16, ge::DataType::DT_INT16},
    {OnnxDataType::INT32, ge::DataType::DT_INT32},         {OnnxDataType::INT64, ge::DataType::DT_INT64},
    {OnnxDataType::STRING, ge::DataType::DT_STRING},       {OnnxDataType::BOOL, ge::DataType::DT_BOOL},
    {OnnxDataType::FLOAT16, ge::DataType::DT_FLOAT16},     {OnnxDataType::DOUBLE, ge::DataType::DT_DOUBLE},
    {OnnxDataType::UINT32, ge::DataType::DT_UINT32},       {OnnxDataType::UINT64, ge::DataType::DT_UINT64},
    {OnnxDataType::COMPLEX64, ge::DataType::DT_COMPLEX64}, {OnnxDataType::COMPLEX128, ge::DataType::DT_COMPLEX128},
    {OnnxDataType::BFLOAT16, ge::DataType::DT_BF16},       {OnnxDataType::FLOAT8E5M2, ge::DataType::DT_FLOAT8_E5M2},
    {OnnxDataType::FLOAT8E4M3FN, ge::DataType::DT_FLOAT8_E4M3FN},
};
}

namespace ge {
ge::DataType OnnxUtil::ConvertOnnxDataType(int64_t onnx_data_type) {
  auto search = onnx_data_type_map.find(onnx_data_type);
  if (search != onnx_data_type_map.end()) {
    return search->second;
  } else {
    return ge::DataType::DT_UNDEFINED;
  }
}

void OnnxUtil::GenUniqueSubgraphName(int subgraph_index, const std::string &original_subgraph_name,
                                     const std::string &parent_node_name, std::string &unique_subgraph_name) {
  unique_subgraph_name = parent_node_name + "_" + std::to_string(subgraph_index) + "_" + original_subgraph_name;
}

std::string OnnxUtil::GenUniqueNodeName(const std::string &graph_name, const std::string &node_name) {
  return graph_name + "/" + node_name;
}
}  // namespace ge
