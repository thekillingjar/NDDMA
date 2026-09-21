/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_VAR_IS_INITIALIZED_OP_PASS_H_
#define GE_GRAPH_PASSES_VAR_IS_INITIALIZED_OP_PASS_H_
#include <map>
#include <memory>
#include <set>
#include <vector>
#include "graph/passes/base_pass.h"

namespace ge {
class VarIsInitializedOpPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  Status CheckSrcNode(const NodePtr &node, bool &inited) const;
  Status CreateConstant(const NodePtr &node, OpDescPtr &op_desc, bool inited) const;
  Status ProcessInAnchor(NodePtr &node, NodePtr &new_node) const;
  Status ChangeNodeToConstant(NodePtr &node, bool inited);
  Status UpdateInitedVars(const NodePtr &node);
  Status CheckAndSetVarInited(const NodePtr &node, bool &inited, int64_t &inited_var) const;
  std::set<int64_t> *CreateInitedVars();
  bool IsVarInitedOnTheGraphAndNode(const NodePtr &node, int64_t var_id) const;

  std::vector<std::unique_ptr<std::set<int64_t>>> var_inited_keeper_;
  std::map<int64_t, std::set<int64_t> *> nodes_to_inited_vars_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_VAR_IS_INITIALIZED_OP_PASS_H_
