/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/mem.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "common/checker.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"
#include "common/memory/mem_type_utils.h"

namespace ge {
Status ContinuousInOutAndRtsSpecailMemTypeInOutChecker(CheckFuncContext &context) {
  auto continuous_node = context.node_a;
  auto specail_mem_node = context.node_b;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_a, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT) ||
      MemLayoutConflictUtil::IsContainTargetType(context.type_a, ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT)) {
    continuous_node = context.node_b;
    specail_mem_node = context.node_a;
  }

  const auto mem_type = MemLayoutConflictUtil::GetAnchorMemType(specail_mem_node.node_, specail_mem_node.io_type_,
                                                                specail_mem_node.index_);
  GELOGI("[MemConflict] rts special memory type: %lld(p2p:%lld, ts:%lld)", mem_type, RT_MEMORY_P2P_DDR, RT_MEMORY_TS);
  const auto continuous_mem_type =
      MemLayoutConflictUtil::GetAnchorMemType(continuous_node.node_, continuous_node.io_type_, continuous_node.index_);
  if (MemTypeUtils::IsMemoryTypeSpecial(continuous_mem_type)) {
    return SUCCESS;
  }
  if ((continuous_node.io_type_ == kIn)
      && (MemLayoutConflictUtil::IsNodeOutRefFromInput(continuous_node, context.all_nodes))) {
    context.result.insert(MemLayoutConflictUtil::GetAnchorFromIndexIo(specail_mem_node));
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, specail_mem_node);
  } else {
    context.result.insert(MemLayoutConflictUtil::GetAnchorFromIndexIo(continuous_node));
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, continuous_node);
  }
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_INPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_INPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);

REGISTER_FUNC(ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT,
              ContinuousInOutAndRtsSpecailMemTypeInOutChecker);
}  // namespace ge
