/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_AUTOSCHEDULE_PASS_RUNNER_HANDLER_H
#define OPTIMIZE_AUTOSCHEDULE_PASS_RUNNER_HANDLER_H
#include "platform/platform_factory.h"

namespace optimize {
namespace autoschedule {
class PassRunnerHandler {
 public:
  static Status RunPasses(ge::AscGraph &graph);
};
}  // namespace autoschedule
}  // namespace optimize

#endif  // OPTIMIZE_AUTOSCHEDULE_PASS_RUNNER_HANDLER_H
