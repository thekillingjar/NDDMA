/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_STAGE_PARTITION_H_
#define GE_GRAPH_PARTITION_STAGE_PARTITION_H_

#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"

namespace ge {
class StagePartitioner {
 public:
  explicit StagePartitioner(ComputeGraphPtr graph) : root_graph_(std::move(graph)) {}
  ~StagePartitioner() = default;

  Status Partition();

 private:
  Status SplitStageLevel();
  Status SplitTailStage();

  Status StagePartition();

  ComputeGraphPtr root_graph_;
  std::map<uint32_t, std::set<NodePtr>> stage_nodes_;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_STAGE_PARTITION_H_
