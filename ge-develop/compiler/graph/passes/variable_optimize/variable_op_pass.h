/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
#include <map>
#include <set>
#include "common/op/transop_util.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "graph/passes/graph_pass.h"

namespace ge {
struct SameVariable {
    std::string var_name;
    std::set<NodePtr> var_nodes;
};
using SameVarPtr = std::shared_ptr<SameVariable>;

class VariableOpPass : public GraphPass {
 public:
  explicit VariableOpPass(GraphRebuildStateCtrl *ctrl) : var_accelerate_ctrl_(ctrl) {}

  ~VariableOpPass() override = default;

  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  Status DealFusion(const SameVarPtr &same_vars);

  Status CheckVariableRefLegally(const SameVarPtr &same_vars, bool &is_var_ref_legally);

  Status UpdateVarAndRefOutputFormatInfo(const GeTensorDesc &final_output, const ge::NodePtr &node,
                                         const SameVarPtr &same_vars);

  Status GenerateVariableVariableRefMap(const ComputeGraphPtr &compute_graph);

  Status CheckVarAndVarRefAreAlike(const NodePtr &var_node, const NodePtr &var_ref_node,
                                   bool &is_var_and_variable_ref_are_alike) const;

  bool IsOpDescSame(const GeTensorDescPtr &op_desc_a, const GeTensorDescPtr &op_desc_b) const;

  Status CheckTransNodeAreInverse(const NodePtr &node_a, const NodePtr &node_b, bool &is_same) const;

  void CopyVariableFormatDataTypeAndShape(const GeTensorDesc &src_tensor_desc, GeTensorDesc &dst_tensor_desc) const;

  Status CheckSameAndTransOp(const SameVarPtr &same_vars, bool &is_matched, VarTransRoad &fusion_road) const;

  Status CheckIfCouldBeOptimized(const SameVarPtr &same_vars, bool &flag, VarTransRoad &fusion_road);

  Status FusionIfNeed(const SameVarPtr &same_vars, VarTransRoad &fusion_road);

  Status UpdateIOFormatInfo(const GeTensorDesc &final_output, const SameVarPtr &same_vars);
  Status RenewVarDesc(const ge::ComputeGraphPtr &graph) const;
  Status RenewVarDesc(uint64_t session_id, const NodePtr &node, const VarTransRoad &fusion_road) const;

  std::vector<NodePtr> GetRefVars(const SameVarPtr &same_vars);
  std::map<SameVarPtr, std::set<NodePtr>> var_and_var_ref_map_;

  GraphRebuildStateCtrl *var_accelerate_ctrl_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
