/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "register/node_converter_registry.h"
#include "graph/node.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_memory.h"

#include "graph/utils/type_utils.h"
#include "engine/node_converter_utils.h"

namespace gert {
namespace {
constexpr size_t kInputIndex = 0U;
constexpr size_t kAxesIndex = 1U;
constexpr size_t kAxesSize = 2UL;
std::string PrintAxes(const std::vector<std::vector<int64_t>> &axes) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < axes.size(); ++i) {
    ss << "[";
    for (size_t j = 0U; j < axes[i].size(); ++j) {
      ss << axes[i][j] << ", ";
    }
    ss << "], ";
  }
  ss << "]";
  return ss.str();
}
ge::Status CheckAxisSize(const std::vector<std::vector<int64_t>> &axes, const ge::OpDescPtr &op_desc) {
  for (const auto &axis : axes) {
    if (axis.size() != kAxesSize) {
      GELOGE(ge::FAILED, "attr [axes] is invalid, axis.size() is %zu, valid value is 2, axes: %s",
             axis.size(), PrintAxes(axes).c_str());
      return ge::FAILED;
    }
    const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(axis[kInputIndex]));
    if ((axis[kInputIndex] >= static_cast<int64_t>(op_desc->GetInputsSize())) ||
        (axis[kAxesIndex] >= static_cast<int64_t>(input_desc->MutableShape().GetDimNum()))) {
      GELOGE(ge::FAILED, "attr [axes] is invalid, axis[0] is %ld, valid range[0, %zu);"
             " axis[1] is %ld, valid range[0, %zu), axes: %s",
             axis[kInputIndex], op_desc->GetInputsSize(), axis[kAxesIndex], input_desc->MutableShape().GetDimNum(),
             PrintAxes(axes).c_str());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}
ge::Status CheckDataType(const ge::OpDescPtr op_desc) {
  uint32_t out_type = ge::DT_INT32;
  (void)ge::AttrUtils::GetInt(op_desc, "dtype", out_type);
  if ((out_type == ge::DT_INT32) || (out_type == ge::DT_INT64)) {
    return ge::SUCCESS;
  }
  GELOGE(ge::FAILED, "attr [dtype][%s] is invalid, allow data type is DT_INT32 and DT_INT64.",
         ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(out_type)).c_str());
  return ge::FAILED;
}
}  // namespace
LowerResult LoweringGatherShapesNode(const ge::NodePtr &node, const LowerInput &input) {
  LOWER_REQUIRE_NOTNULL(node);
  const ge::OpDescPtr &op_desc = node->GetOpDesc();
  LOWER_REQUIRE_NOTNULL(op_desc);

  std::vector<std::vector<int64_t>> axes;
  LOWER_REQUIRE(ge::AttrUtils::GetListListInt(op_desc, "axes", axes));
  LOWER_REQUIRE_GRAPH_SUCCESS(CheckAxisSize(axes, op_desc));
  LOWER_REQUIRE_GRAPH_SUCCESS(CheckDataType(op_desc));

  StorageShape ss{{static_cast<int64_t>(axes.size())}, {static_cast<int64_t>(axes.size())}};
  auto output_shape_tensor = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), ss);

  auto out_tensor_data =
      bg::DevMemValueHolder::CreateSingleDataOutput("GatherShapesKernel", input.input_shapes, op_desc->GetStreamId());
  LOWER_REQUIRE_NOTNULL(out_tensor_data);
  out_tensor_data->SetPlacement(kOnHost);

  return {HyperStatus::Success(), {}, {output_shape_tensor}, {out_tensor_data}};
}
REGISTER_NODE_CONVERTER("GatherShapes", LoweringGatherShapesNode);
}  // namespace gert
