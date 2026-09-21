/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alignment_handler.h"

namespace optimize::autoschedule {
ge::Status AlignmentHandler::AlignVectorizedStrides(ascir::ImplGraph &impl_graph) {
  const auto &platform = PlatformFactory::GetInstance().GetPlatform();
  GE_CHECK_NOTNULL(platform, "Platform is not found.");
  const auto &strategy = platform->GetAlignmentStrategy();
  GE_CHECK_NOTNULL(strategy, "Get alignment strategy by platform failed.");
  return strategy->AlignVectorizedStrides(impl_graph);
}
}  // namespace optimize::autoschedule
