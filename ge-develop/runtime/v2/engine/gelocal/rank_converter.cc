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
#include "engine/node_converter_utils.h"

namespace gert {
LowerResult LoweringRankNode(const ge::NodePtr &node, const LowerInput &input) {
  LOWER_REQUIRE_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  LOWER_REQUIRE_NOTNULL(op_desc);

  StorageShape ss{{}, {}};
  auto output_shapes = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), ss);
  auto compute_holder =
      bg::DevMemValueHolder::CreateSingleDataOutput("RankKernel", input.input_shapes, op_desc->GetStreamId());
  LOWER_REQUIRE_NOTNULL(compute_holder);
  compute_holder->SetPlacement(kOnHost);
  LOWER_REQUIRE_VALID_HOLDER(compute_holder, "compute Rank kernel fail.");

  return {HyperStatus::Success(),
          {compute_holder},
          {output_shapes},
          {compute_holder}};
}
REGISTER_NODE_CONVERTER("Rank", LoweringRankNode);
}  // namespace gert
