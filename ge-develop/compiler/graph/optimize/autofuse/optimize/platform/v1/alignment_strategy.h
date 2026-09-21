/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V1_ALIGNMENT_STRATEGY_H
#define OPTIMIZE_PLATFORM_V1_ALIGNMENT_STRATEGY_H

#include "platform/common/base_alignment_strategy.h"
namespace optimize {
class AlignmentStrategy : public BaseAlignmentStrategy {
 public:
  ~AlignmentStrategy() override = default;
  AlignmentStrategy() = default;

 protected:
  AlignmentType GetDefaultAlignmentType() override;

  ge::Status DefaultAlignmentInferFunc(const ge::AscNodePtr &node) override;
  ge::Status BroadcastAlignmentInferFunc(const ge::AscNodePtr &node) override;
  ge::Status ConcatAlignmentInferFunc(const ge::AscNodePtr &node) override;
  ge::Status EleWiseAlignmentInferFunc(const ge::AscNodePtr &node) override;
  ge::Status LoadAlignmentInferFunc(const ge::AscNodePtr &node) override;
  ge::Status StoreAlignmentInferFunc(const ge::AscNodePtr &node) override;
};

}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V1_ALIGNMENT_STRATEGY_H
