/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_fusion_strategy.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/not_fuse_reason_code.h"
#include "utils/auto_fuse_config.h"
#include "can_fuse/backend/asc_graph_axis_mapping.h"

namespace ge {
bool ReduceFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  // reduce不支持水平融合
  if (BackendUtils::IsHorizontal(node1, node2)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 have horizontal link, and one "
        "node is Reduce]", node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
        node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kReduceCanNotFuseHorizontal));
    return false;
  }

  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  if (attr1->HasFuseType(loop::FuseType::kReduction)) {
    uint64_t supported_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));
    // 不支持reduce后融合 非 elementwise
    if (attr2->GetAllFuseType() != supported_type) {
      GELOGI(
          "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Reduce can only backward fuse with "
          "elementwise nodes]",
          node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
          ge::NotFuseReasonCode(ge::NotFuseReason::kReduceCanOnlyBackwardFuse3Elementwise));
      return false;
    }
    // reduce最多只能向后融合3个elementwise（非AscBackend数量，而是AscBackend内部的elementwise数量）
    const auto config = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver();
    if (attr1->GetReduceFusedElementwiseNodeNum() + BackendUtils::GetComputeNodeNumInAscgraph(node2) >
        config.max_reduce_can_fuse_elementwise_nums) { // reduce已后融合的elementwise数量+将要融合的elementwise数量不大于3
      GELOGI(
          "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Reduce can only backward fuse with at most"
          " 3 elementwise nodes]", node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
          node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kReduceCanOnlyBackwardFuse3Elementwise));
      return false;
    }
  }
  return true;
}
REGISTER_FUSION_STRATEGY(ReduceFusionStrategy, loop::FuseType::kReduction);
}