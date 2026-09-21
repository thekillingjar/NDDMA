/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_
#define GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_
#include <string>
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"

namespace ge {
using GraphToNodeMap = std::map<ge::ComputeGraphPtr, std::map<std::string, ge::NodePtr>>;

class VarMemAssignUtil {
 public:
  static Status AssignVarMemory(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignConstantOpMemory(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignStaticMemory2Node(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignVarAttr2Nodes(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignMemory2HasRefAttrNode(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignData2Fp32Var(const ge::NodePtr &node, const uint64_t session_id);
  static std::string GetNameForVarManager(const OpDescPtr &op_desc);
 private:
  static Status SetOutVariableAttr(const ge::NodePtr &node, const ge::NodePtr &var_node, const size_t index,
                                   const uint64_t session_id);
  static Status DealExportVariableNode(const ge::NodePtr &node, const ge::NodePtr &var_node, const uint64_t session_id,
                                       const uint32_t depth = 0U);
  static Status DealVariableNode(const uint32_t graph_id, const ge::NodePtr &node, const uint64_t session_id);

  static Status DealBroadCastNode(const uint32_t graph_id, const ge::NodePtr &node,
                                  const ge::InDataAnchorPtr &in_data_anchor,
                                  const ge::NodePtr &var_node, const uint64_t session_id);

  static ge::NodePtr GetFinalTransNode(const ge::NodePtr &trans_node, const uint32_t depth = 0U);

  static Status DealTransNode(const ge::NodePtr &final_trans_node);
  static Status DealExportTransNode(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                    const uint32_t depth = 0U);
  static Status AssignData2VarRef(const ge::NodePtr &has_ref_attr_node, const std::string &src_var_name,
                                  const uint64_t session_id, const uint32_t out_index, GraphToNodeMap &graph_to_node);

  static Status SetOutTransNodeToAssign(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                        const size_t index);
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_
