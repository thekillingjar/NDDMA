/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/node.h"
#include "register/node_converter_registry.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/converter_checker.h"
#include "engine/node_converter_utils.h"

namespace gert {
LowerResult LoweringReshape(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto out_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  if (out_shapes.size() != 1U) {
    return CreateErrorLowerResult("Failed to generate infershape for node %s", node->GetName().c_str());
  }
  return {HyperStatus::Success(), {}, out_shapes, {lower_input.input_addrs[0]}};
}
REGISTER_NODE_CONVERTER("Reshape", LoweringReshape);
REGISTER_NODE_CONVERTER("Bitcast", LoweringReshape);
REGISTER_NODE_CONVERTER("ExpandDims", LoweringReshape);
REGISTER_NODE_CONVERTER("Squeeze", LoweringReshape);
REGISTER_NODE_CONVERTER("Unsqueeze", LoweringReshape);
REGISTER_NODE_CONVERTER("SqueezeV2", LoweringReshape);
REGISTER_NODE_CONVERTER("UnsqueezeV2", LoweringReshape);
REGISTER_NODE_CONVERTER("SqueezeV3", LoweringReshape);
REGISTER_NODE_CONVERTER("UnsqueezeV3", LoweringReshape);
REGISTER_NODE_CONVERTER("FlattenV2", LoweringReshape);
REGISTER_NODE_CONVERTER("Flatten", LoweringReshape);

LowerResult LoweringReshape(const ge::NodePtr &node, const FFTSLowerInput &lower_input) {
  LOWER_REQUIRE(!lower_input.mem_pool_types.empty(), "Input memory type is empty");
  GELOGD("Lowering reshape_like node[%s], type: [%s], mem_type: [%u].",
         node->GetNamePtr(), node->GetTypePtr(), lower_input.mem_pool_types[0UL]);
  std::vector<uint32_t> out_mem_type{lower_input.mem_pool_types[0UL]};
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  (void)op_desc->SetExtAttr(kFtfsMemoryPoolType, out_mem_type);

  if (!bg::IsOutputUnkownShape(op_desc)) {
    const auto &out_shapes = NodeConverterUtils::CreateOutputShapes(op_desc);
    return {HyperStatus::Success(), {}, out_shapes, {lower_input.input_addrs[0L]}};
  }
  auto out_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  if (out_shapes.size() != 1UL) {
    return CreateErrorLowerResult("Failed to generate infershape for node %s", node->GetName().c_str());
  }
  return {HyperStatus::Success(), {}, out_shapes, {lower_input.input_addrs[0L]}};
}

FFTS_REGISTER_NODE_CONVERTER("Reshape", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("ExpandDims", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("Squeeze", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("Unsqueeze", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("SqueezeV2", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("UnsqueezeV2", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("SqueezeV3", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("UnsqueezeV3", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("FlattenV2", LoweringReshape);
FFTS_REGISTER_NODE_CONVERTER("Flatten", LoweringReshape);
}  // namespace gert
