/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_UTILS_RESOURCE_GUARDER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_UTILS_RESOURCE_GUARDER_H_
#include <cstdint>
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "graph/fast_graph/fast_node.h"
#include "graph/utils/fast_node_utils.h"
#include "exe_graph/lowering/exe_graph_attrs.h"

namespace gert {
namespace bg {
inline bool IsGuarderOf(const ge::FastNode *src_node, const ge::FastNode *guarder_node) {
  int32_t guard_index;
  if (!ge::AttrUtils::GetInt(guarder_node->GetOpDescBarePtr(), kReleaseResourceIndex, guard_index)) {
    return false;
  }
  return ge::FastNodeUtils::GetInDataNodeByIndex(guarder_node, guard_index) == src_node;
}
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_UTILS_RESOURCE_GUARDER_H_
