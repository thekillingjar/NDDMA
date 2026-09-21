/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_PARTITION_OPTIMIZER_DYNAMIC_DATA_FLOW_PARTITIONER_PASS_H_
#define GRAPH_PARTITION_OPTIMIZER_DYNAMIC_DATA_FLOW_PARTITIONER_PASS_H_
#include "graph/partition/dynamic_shape_partition.h"

namespace ge {
class DynamicDataFlowPartitionerPass : public PartitionerPass {
public:
  DynamicDataFlowPartitionerPass() = default;
  ~DynamicDataFlowPartitionerPass() override = default;
  Status Run(const ge::ComputeGraphPtr &graph, const std::vector<std::shared_ptr<BaseCluster>> &sorted_unique_clusters,
             bool &result) override;
  Status MarkDataFlowOpAttr(const ge::ComputeGraphPtr &graph) const;
  Status GetDataFlowOpMapping(const ge::ComputeGraphPtr &graph);
  Status GetUnknownDataFlowOp();
  Status GetDataFlowOpNeedProcess(bool &result);
  Status GetDataFlowOpAttr(const std::vector<std::shared_ptr<BaseCluster>> &sorted_unique_clusters);
  void ClearDataFlowsource();

  std::unordered_set<int64_t> data_flow_handle_;
  std::unordered_set<NodePtr> unknown_data_flow_ops_;
  std::unordered_map<int64_t, std::unordered_set<NodePtr>> data_flow_ops_;
  std::unordered_map<NodePtr, bool> data_flow_ops_attr_;
};

}  // namespace ge

#endif  // GRAPH_PARTITION_OPTIMIZER_DYNAMIC_DATA_FLOW_PARTITIONER_PASS_H_