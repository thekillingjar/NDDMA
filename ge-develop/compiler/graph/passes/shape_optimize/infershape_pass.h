/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_INFERSHAPE_PASS_H_
#define GE_GRAPH_PASSES_INFERSHAPE_PASS_H_

#include <stack>
#include "graph/passes/shape_optimize/infer_base_pass.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "graph/resource_context_mgr.h"
#include "graph/inference_context.h"

namespace ge {
class InferShapePass : public InferBasePass {
 public:
  InferShapePass() {}
  InferShapePass(GraphRebuildStateCtrl *ctrl, ResourceContextMgr *resource_mgr)
      : resource_op_access_ctrl_(ctrl), resource_context_mgr_(resource_mgr) {}
  std::string SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const override;
  graphStatus Infer(NodePtr &node) override;

  graphStatus UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) override;
  graphStatus UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src, const GeTensorDescPtr &dst) override;
  graphStatus UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                    const GeTensorDescPtr &dst) override;
  graphStatus UpdateOutputFromSubgraphsForSubgraphMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                            const GeTensorDescPtr &dst) override;

  Status OnSuspendNodesLeaked() override;

 private:
  graphStatus InferShapeAndType(NodePtr &node);
  graphStatus CallInferShapeFunc(NodePtr &node, Operator &op) const;
  bool SameTensorDesc(const GeTensorDescPtr &src, const GeTensorDescPtr &dst) const;
  void UpdateCurNodeOutputDesc(const NodePtr &node) const;
  Status UpdateNetOutputIODesc(const NodePtr &node) const;
  Status RepassReliedNodeIfResourceChanged(const InferenceContextPtr &inference_context, const NodePtr &cur_node);
  Status RegisterNodesReliedOnResource(const InferenceContextPtr &inference_context, NodePtr &node) const;
  Status SuspendV1LoopExitNodes(const NodePtr &node);
  struct SuspendNodes {
    std::stack<NodePtr> nodes;
    std::unordered_set<NodePtr> nodes_set;

    NodePtr PopSuspendedNode() {
      auto top_node = nodes.top();
      nodes.pop();
      nodes_set.erase(top_node);
      return top_node;
    }
  };
  std::map<std::string, SuspendNodes> graphs_2_suspend_nodes_;
  // if resource shape changed, use this ctrl to mark graph relied on that resource rebuild
  GraphRebuildStateCtrl *resource_op_access_ctrl_ = nullptr;
  // This mgr belongs to session, give it when create inference context, so that resource op can
  // get/set resource_shapes to mgr
  ResourceContextMgr *resource_context_mgr_ = nullptr;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INFERSHAPE_PASS_H_
