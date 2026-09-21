/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_STRATEGY_TRANSPOSE_FUSION_STRATEGY_H
#define AUTOFUSE_CAN_FUSE_STRATEGY_TRANSPOSE_FUSION_STRATEGY_H

#include "can_fuse/strategy/fusion_strategy.h"
namespace ge {
class TransposeFusionStrategy : public FusionStrategy {
public:
  TransposeFusionStrategy() = default;

  ~TransposeFusionStrategy() override = default;

  TransposeFusionStrategy(const TransposeFusionStrategy &) = delete;
  TransposeFusionStrategy &operator=(const TransposeFusionStrategy &) = delete;

  // 检查两个节点是否可以融合
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) override;
};
}
#endif //AUTOFUSE_CAN_FUSE_STRATEGY_TRANSPOSE_FUSION_STRATEGY_H