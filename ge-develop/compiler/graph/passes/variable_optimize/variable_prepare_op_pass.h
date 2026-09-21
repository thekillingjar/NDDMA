/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_

#include <map>
#include <stack>
#include <string>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/passes/graph_pass.h"

namespace ge {
class VariablePrepareOpPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);

 private:
  Status DealVariableNode(ge::NodePtr &var_node);
  Status DealWritableNode(const ge::NodePtr &writable_node, int32_t input_index, int32_t output_index,
                          const ge::NodePtr &var_node);
  Status GetPeerNodeOfRefOutput(const ge::NodePtr &node, int32_t output_index,
                                std::stack<std::pair<NodePtr, std::pair<int32_t, int32_t>>> &nodes);
  Status AddVariableRef(ge::NodePtr &final_writable_node, const ge::NodePtr &var_node, int32_t index) const;
  Status InsertVariableRef(ge::NodePtr &node, int32_t in_index, const ge::NodePtr &var_node) const;
  Status AddControlEdge(const ge::NodePtr &node, const ge::NodePtr &variable_ref_node) const;
  NodePtr CreateVariableRef(const std::string &variable_ref_name, const ge::NodePtr &var_node) const;
  NodePtr CreateRefIdentity(const std::string &ref_identity_name, const ge::NodePtr &node, uint32_t input_index) const;
  void GetWritableNodeOutIndex(const NodePtr &node, int32_t input_index, std::vector<int32_t> &output_indexes);
  void GenerateRefTypeAndInputOutputMap(const NodePtr &node);
  void FindRefOutIndex(const std::string &node_type, int32_t input_index,
                       const std::map<std::string, std::map<int32_t, std::vector<int32_t>>> &ref_map,
                       std::vector<int32_t> &output_indexes) const;
  Status CheckStreamLabel(const ge::NodePtr &var_ref_node, const ge::NodePtr &final_writable_node) const;
  bool HasControlOut(const ge::NodePtr &node) const;
  Status InsertIdentityNode(const NodePtr &src_node, const AnchorPtr &out_anchor, const NodePtr &peer_node,
                            const AnchorPtr &peer_in_anchor, NodePtr &identity_node) const;

  std::map<std::string, std::map<int32_t, std::vector<int32_t>>> ref_input_output_map_;
  static std::map<std::string, std::map<int32_t, std::vector<int32_t>>> ref_node_without_prototype_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_
