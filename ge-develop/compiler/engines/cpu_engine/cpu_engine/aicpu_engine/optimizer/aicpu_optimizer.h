/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DAVINCI_AICPU_OPTIMIZER_H_
#define DAVINCI_AICPU_OPTIMIZER_H_

#include "common/optimizer/cpu_optimizer.h"

namespace aicpu {
using OptimizerPtr = std::shared_ptr<Optimizer>;

class AicpuOptimizer : public CpuOptimizer {
 public:
  /**
   * get optimizer instance
   * @return ptr stored aicpu optimizer
   */
  static OptimizerPtr Instance();

  /**
   * Destructor
   */
  virtual ~AicpuOptimizer() = default;
 private:
  /**
   * Constructor
   */
  AicpuOptimizer() = default;

  // Copy prohibited
  AicpuOptimizer(const AicpuOptimizer& aicpu_optimizer) = delete;

  // Move prohibited
  AicpuOptimizer(const AicpuOptimizer&& aicpu_optimizer) = delete;

  // Copy prohibited
  AicpuOptimizer& operator=(const AicpuOptimizer& aicpu_optimizer) = delete;

  // Move prohibited
  AicpuOptimizer& operator=(AicpuOptimizer&& aicpu_optimizer) = delete;

 private:
  // singleton instance
  static OptimizerPtr instance_;
};
}  // namespace aicpu

#endif  // DAVINCI_AICPU_OPTIMIZER_H
