/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_PASS_H_
#define GE_GRAPH_PASSES_FOLDING_PASS_H_

#include <map>
#include <memory>
#include <vector>

#include "graph/passes/base_pass.h"
#include "host_kernels/kernel.h"

namespace ge {
namespace folding_pass {
shared_ptr<Kernel> GetKernelByType(const NodePtr &node);
bool IsNoNeedConstantFolding(const NodePtr &node);
bool IsUserSpecifiedSkipConstantFold(const NodePtr &node);
}

using IndexsToAnchors = std::map<int32_t, std::vector<InDataAnchorPtr>>;

class FoldingPass : public BaseNodePass {
 protected:
  Status Folding(NodePtr &node, std::vector<GeTensorPtr> &outputs);
  bool IsResultSuccess(const Status compute_ret) const;

 private:
  Status AddConstNode(const NodePtr &node,
                      IndexsToAnchors indexes_to_anchors,
                      std::vector<GeTensorPtr> &v_weight);
  Status DealWithInNodes(const NodePtr &node) const;
  Status RemoveNodeKeepingCtrlEdges(const NodePtr &node);
  Status ConnectNodeToInAnchor(const InDataAnchorPtr &in_anchor, const NodePtr &node, int32_t node_index);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_PASS_H_
