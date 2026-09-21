/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cinttypes>
#include "common/checker.h"
#include "graph/node.h"
#include "exe_graph/lowering/value_holder.h"
#include "register/node_converter_registry.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_variable.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace gert {
namespace {
constexpr ge::char_t const kPlacementOnHost[] = "host";
constexpr int32_t kShapeIndex = 0;
constexpr int32_t kAddrIndex = 1;
}
bool IsVarPlacedOnHost(const ge::NodePtr &node) {
  const std::string *placement = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), ge::ATTR_VARIABLE_PLACEMENT);
  return (placement != nullptr) && (*placement == kPlacementOnHost);
}

LowerResult LoweringVariable(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto shape_and_addr = bg::Variable(node->GetName(), *lower_input.global_data, node->GetOpDesc()->GetStreamId());
  LOWER_REQUIRE(shape_and_addr.size() == 2U);
  LowerResult result;
  result.out_shapes.push_back(shape_and_addr[kShapeIndex]);
  shape_and_addr[1]->SetPlacement(IsVarPlacedOnHost(node) ? TensorPlacement::kOnHost : TensorPlacement::kOnDeviceHbm);
  result.out_addrs.push_back(shape_and_addr[kAddrIndex]);
  return result;
}

REGISTER_NODE_CONVERTER("Variable", LoweringVariable);
/**
 *  Constant usually used in online, and weight lifecycle is session level.
 *  Its addr managed in var manager. So here lowering constant from var manager.
 *  Later move to host resource manager.
 */
REGISTER_NODE_CONVERTER("Constant", LoweringVariable);
}  // namespace gert
