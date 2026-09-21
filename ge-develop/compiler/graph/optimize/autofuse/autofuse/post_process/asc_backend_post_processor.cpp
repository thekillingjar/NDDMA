/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_backend_post_processor.h"
#include "common/checker.h"
#include "can_fuse/backend/backend_utils.h"
#include "common/scope_tracing_recorder.h"

namespace ge {
Status AscBackendPostProcessor::Do(const ComputeGraphPtr &graph) const {
  TRACING_PERF_SCOPE(TracingModule::kModelCompile, "PostProcess", "CanFuse", graph->GetName());
  GE_ASSERT_SUCCESS(adapter_.DoBeforePass(graph));
  GE_ASSERT_SUCCESS(pass_.Run(graph));
  GE_ASSERT_SUCCESS(adapter_.DoAfterPass(graph));
  return SUCCESS;
}
}  // namespace ge