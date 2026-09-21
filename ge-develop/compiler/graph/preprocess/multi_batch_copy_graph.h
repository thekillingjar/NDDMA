/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_MULTI_BATCH_COPY_GRAPH_H_
#define GE_GRAPH_PREPROCESS_MULTI_BATCH_COPY_GRAPH_H_
#include <map>
#include <queue>
#include <vector>
#include <set>

#include "ge/ge_api_error_codes.h"

#include "graph/compute_graph.h"

namespace ge {
namespace multibatch {
Status ProcessMultiBatch(const ComputeGraphPtr &graph, const uint64_t session_id);

Status GetDynamicOutputShape(const ComputeGraphPtr &graph);

enum NodeStatus {
  kNodeInBatchBranch,
  kNodeOutBatchBranch,
  kNodeStartNode,
  kNodeNotSupportNode,
};

enum DynamicType {
  kDynamicBatch,
  kDynamicImageSize,
  kDynamicDims,
  kDynamicUnknown,
};
}  // namespace multibatch
}  // namespace ge
#endif  // GE_GRAPH_PREPROCESS_MULTI_BATCH_COPY_GRAPH_H_
