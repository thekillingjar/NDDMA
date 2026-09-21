/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_optimizer.h"
#include <mutex>
#include "util/constant.h"

namespace aicpu {

OptimizerPtr AicpuOptimizer::instance_ = nullptr;

OptimizerPtr AicpuOptimizer::Instance() {
  static std::once_flag flag;
  std::call_once(flag,
                 [&]() { instance_.reset(new (std::nothrow) AicpuOptimizer); });
  return instance_;
}

FACTORY_GRAPH_OPTIMIZER_CLASS_KEY(AicpuOptimizer, "AICPUOptimizer")
}  // namespace aicpu
