/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_ASC_BACKEND_POST_PROCESSOR_H
#define AUTOFUSE_POST_PROCESS_ASC_BACKEND_POST_PROCESSOR_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "scheduler_adapter/asc_backend_scheduler_adapter.h"
#include "pass/asc_graph_pass.h"

namespace ge {
class AscBackendPostProcessor {
 public:
  Status Do(const ComputeGraphPtr &graph) const;

 private:
  AscBackendSchedulerAdapter adapter_;
  AscGraphPass pass_;
};
}  // namespace ge

#endif  // AUTOFUSE_POST_PROCESS_ASC_BACKEND_POST_PROCESSOR_H
