/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_graph_pass.h"
#include "common/checker.h"

namespace ge {

Status AscGraphPass::Run(const ComputeGraphPtr &graph) const {
  // 循环外提，把Broadcast移到store前面，在cse前面可以做Broadcast的cse优化
  GE_ASSERT_SUCCESS(cyclic_external_lift_pass_.Run(graph));
  // Broadcast后移，在cse前面可以做Broadcast的cse优化
  GE_ASSERT_SUCCESS(broadcast_backward_pass_.Run(graph));
  // CSE PASS
  GE_ASSERT_SUCCESS(cse_pass_.Run(graph));
  // relu optimize
  GE_ASSERT_SUCCESS(cube_fixpip_pass_.Run(graph));
  GELOGI("Graph %s asc graph pass.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
