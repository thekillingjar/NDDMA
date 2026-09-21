/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/symbolizer/guard_dfx_context.h"
#include "attribute_group/attr_group_shape_env.h"

namespace ge {
GuardDfxContext::GuardDfxContext(const std::string &guard_dfx_info) {
  if (GetCurShapeEnvContext() != nullptr) {
    GetCurShapeEnvContext()->SetGuardDfxContextInfo(guard_dfx_info);
  }
}
GuardDfxContext::~GuardDfxContext() {
  if (GetCurShapeEnvContext() != nullptr) {
    GetCurShapeEnvContext()->ClearGuardDfxContextInfo();
  }
}
}
