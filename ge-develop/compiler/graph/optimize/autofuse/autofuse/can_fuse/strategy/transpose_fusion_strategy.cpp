/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_fusion_strategy.h"
#include <cstdlib>
#include <string>
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/not_fuse_reason_code.h"
#include "backend/backend_spec.h"

namespace ge {
bool HasLoad(const NodePtr &node) {
  const auto attr = node->GetOpDesc()->GetAttrsGroup<ge::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  for (const auto &subnode : attr->GetAscGraph()->GetAllNodes()) {
    if (subnode->GetType() == kLoadType) {
      return  true;
    }
  }
  return false;
}

bool TransposeFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  constexpr uint64_t pointwise_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));
  auto const backend_spec = optimize::BackendSpec::GetInstance();
  uint32_t transpose_mode = backend_spec->transpose_mode;
  if (transpose_mode == static_cast<uint32_t>(optimize::TransposeMode::TRANSPOSE_MODE_UNNORMAL)) { // 1:非normal模式
    // a5单独的融合控制逻辑
  } else {
    // a3单独的融合控制逻辑
    // 1、如果另一个节点是Brc节点，则不能融合
    if ((attr1->GetAllFuseType() == pointwise_type) &&
        attr2->HasFuseType(loop::FuseType::kTranspose) &&
        (!BackendUtils::IsNodeAllInputsAreSimplestLoad(node1))) {
      GELOGD("Transpose don't support fusion with Broadcast type node.");
      return false;
    }
    if ((attr2->GetAllFuseType() == pointwise_type) &&
        attr1->HasFuseType(loop::FuseType::kTranspose) &&
        (!BackendUtils::IsNodeAllInputsAreSimplestLoad(node2))) {
      GELOGD("Transpose don't support fusion with Broadcast type node.");
      return false;
    }
  }
  // transpose和elementwise融合(A3、A5共同逻辑)
  // 1、仅支持垂直融合
  if (BackendUtils::IsVertical(node1, node2)) {
    // 2、支持transpose后融合elementwise
    if (attr1->HasFuseType(loop::FuseType::kTranspose) && (attr2->GetAllFuseType() == pointwise_type)) {
      return true;
    }
    // 3、支持transpose前融合elementwise，但elementwise节点不能被多引用，且输入不是纯Scalar
    if (node1->GetOutAllNodes().size() > 1UL) {
      GELOGD("Node1 %s with single output and multiple refs, do not support fuse with Transpose.", node1->GetNamePtr());
      return false;
    }
    if (!HasLoad(node1)) {
      GELOGD("No LoadType in Node1 %s， do not support fuse with Transpose.", node1->GetNamePtr());
      return false;
    }
    if ((attr1->GetAllFuseType() == pointwise_type) && attr2->HasFuseType(loop::FuseType::kTranspose)) {
      return true;
    }
  }
  GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s]"
         "Transpose can only undergo vertical fusion with Pointwise.",
         node1->GetName().c_str(), node1->GetType().c_str(),
         node2->GetName().c_str(), node2->GetType().c_str(),
         ge::NotFuseReasonCode(ge::NotFuseReason::kTransposeCanNotFuseWithNotPointWise));
  return false;
}
REGISTER_FUSION_STRATEGY(TransposeFusionStrategy, loop::FuseType::kTranspose);
}