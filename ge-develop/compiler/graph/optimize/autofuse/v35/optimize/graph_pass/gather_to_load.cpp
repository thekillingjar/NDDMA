/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_to_load.h"
#include "ascir_ops.h"
#include "ascgraph_info_complete.h"
#include "node_utils.h"
#include "schedule_utils.h"
#include "graph_utils.h"
#include "can_fuse/backend/backend_utils.h"

using namespace ge::ops;
using namespace ge::ascir_op;
namespace optimize {
Status GatherToLoadPass::RunPass(ge::AscGraph &graph) {
  for (auto node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsGather(node)) {
      GELOGD("gather node name %s Type %s compute type %d",node->GetNamePtr(), node->GetType().c_str(), node->attr.api.compute_type);
      node->attr.api.compute_type = ge::ComputeType::kComputeLoad;
    }
  }
  return ge::SUCCESS;
}
}  // namespace optimize