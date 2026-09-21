/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include "graph/node.h"
#include "graph_builder/converter_checker.h"
#include "graph/utils/type_utils.h"
#include "engine/node_converter_utils.h"

namespace gert {
namespace {
ge::Status CheckDataType(const ge::OpDesc *const op_desc) {
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

LowerResult LoweringSizeNode(const ge::NodePtr &node, const LowerInput &input) {
  LOWER_REQUIRE_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  LOWER_REQUIRE_NOTNULL(op_desc);
  LOWER_REQUIRE_GRAPH_SUCCESS(CheckDataType(op_desc));

  StorageShape ss{{}, {}};
  auto output_shape = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), ss);
  auto output_data = bg::DevMemValueHolder::CreateSingleDataOutput(
      "GetShapeSizeKernel", {input.input_shapes[0]}, op_desc->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(output_data, "compute size kernel fail.");
  output_data->SetPlacement(kOnHost);
  return {HyperStatus::Success(), {}, {output_shape}, {output_data}};
}

REGISTER_NODE_CONVERTER("Size", LoweringSizeNode);
}  // namespace gert
