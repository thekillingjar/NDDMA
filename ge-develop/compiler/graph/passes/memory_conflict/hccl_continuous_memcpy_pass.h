/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_HCCL_CONTINUOUS_MEMCPY_PASS_H_
#define GE_GRAPH_PASSES_HCCL_CONTINUOUS_MEMCPY_PASS_H_

#include <string>
#include <unordered_map>

#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
class HcclContinuousMemcpyPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  OpDescPtr CreateIdentityOpDesc(const OutDataAnchorPtr &out_data_anchor);

  NodePtr CreateAssignNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor);

  std::string CheckDuplicateName(const std::string &node_name);

  Status ModifyEdgeConnection(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
          const InDataAnchorPtr &hccl_in_anchor);

  Status InsertIdentityBeforeHccl(const OutDataAnchorPtr &src_out_anchor,
                                  const InDataAnchorPtr &hccl_in_anchor);

  Status InsertAssignAfterBroadcastIfNeed(const ComputeGraphPtr &graph,
                                          const OutDataAnchorPtr &var_out_anchor,
                                          const InDataAnchorPtr &hccl_in_anchor);

  Status ContinuousInputProcess(const ComputeGraphPtr &graph, const NodePtr node);

  Status P2pmemInputProcess(const ComputeGraphPtr &graph, const NodePtr node);

  bool IsDataNode(const std::string& node_type) const;

  std::map<std::string, uint32_t> node_num_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_HCCL_MEMCPY_PASS_H_
