/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MEM_CHECK_UTIL_H_
#define GE_GRAPH_PASSES_MEM_CHECK_UTIL_H_

#include <map>
#include <vector>
#include <bitset>
#include "mem_layout_conflict_optimizer.h"
#include "graph/graph.h"
#include "graph/passes/graph_pass.h"
#include "graph/node.h"
#include "framework/common/ge_types.h"

namespace ge {

struct MemLayoutConflictUtil {
  static bool HasRefVarName(const OutDataAnchorPtr &out_data_anchor, std::string &ref_var_src_var_name);

  static bool HasRefVarName(const OutDataAnchor *out_data_anchor, std::string &ref_var_src_var_name);

  static AnchorPtr GetAnchorFromIndexIo(const ge::NodeIndexIO &node);

  static std::string GetAnchorName(std::string name, IOType io_type, int32_t index);

  static Status CreateIdentityOpDesc(const std::vector<InDataAnchorPtr> &in_data_anchors, OpDescPtr &identity_op);

  static Status UpdateIsInputConstForNetoutput(const std::vector<InDataAnchorPtr> &in_data_anchors,
                                               const NodePtr &identity_node);
  static bool IsGraphFeatureMapRefreshable(const ComputeGraphPtr &graph);

  static bool IsSupportUserInputNopaddingContinuousOutput(const ComputeGraphPtr &graph);

  static bool IsAddressRefreshable(const NodePtr &node);

  static bool IsPhysicalAddressRefreshable(const NodePtr &node);

  static bool HasNotSupportPhysicalMemoryRefreshNode(const CheckFuncContext &context) ;

  static bool IsStaticGraph(const ge::ComputeGraphPtr &graph);

  static bool IsConst(const NodePtr &node);

  static bool PeerIsIdentityOrMemcpyAsync(const InDataAnchorPtr &in_anchor);

  static bool PeerIsIdentityOrMemcpyAsync(const OutDataAnchorPtr &out_anchor);

  static bool IsUnknownShape(const InDataAnchorPtr &in_anchor);

  static bool IsUnknownShape(const OutDataAnchorPtr &out_anchor);

  static bool IsSkipInsert(const InDataAnchorPtr &in_anchor);

  static bool IsSkipInsert(const OutDataAnchorPtr &out_anchor);

  static bool IsNodeOutRefFromInput(const ge::NodeIndexIO &in_anchor, const NodeIndexIOVector &all_nodes);

  static ge::NodeIndexIO GetRefInput(const ge::NodeIndexIO &out_anchor, const NodeIndexIOVector &all_nodes);
  static bool TraverseRefChainReverse(const Node *const node, int32_t out_index,
                                      const std::function<bool(const Node *const, int32_t)>& judge_func);
  static bool TraverseRefChain(const Node *const node, int32_t in_index,
                               const std::function<bool(const Node *const, int32_t)>& judge_func);
  static bool IsNoPaddingContinuousInput(const NodePtr &node);
  static bool IsNoPaddingContinuousOutput(const NodePtr &node);
  static bool IsNoPaddingContinuousInput(const Node *node);
  static bool IsNoPaddingContinuousOutput(const Node *node);
  // 是不是连接到需要连续输入的节点，通过RefNode连接的也考虑在内
  static bool IsContinuousInputThroughRefNode(InDataAnchor *const in_data_anchor,
                                              const bool no_padding, InDataAnchor *&continuous_node_in_anchor);
  static bool IsContinuousOutputThroughRefNode(OutDataAnchor *const out_data_anchor,
                                              const bool no_padding, OutDataAnchor *&continuous_node_out_anchor);
  static bool IsContinuousInput(const NodePtr &node);
  static bool IsContinuousInput(const Node *node);
  static bool IsContinuousOutput(const NodePtr &node);
  static bool IsContinuousOutput(const Node *node);
  static bool IsContainTargetType(const SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> &type,
                                  const AnchorAttribute &target_type);
  static int64_t GetAnchorMemType(const NodePtr &node, const IOType io_type, const uint32_t index);
  static bool IsSameMemType(const CheckFuncContext &context);
  static Status IsLinkAssignLinkVar(const NodeIndexIO &in_anchor, bool &flag);
  static Status IsVarLinkAssignLinkRefNode(const CheckFuncContext &context, const NodeIndexIO &ref_node_out_anchor,
                                           std::vector<NodeIndexIO> &in_anchors);
  static Status AssignVarInsertIdentity(CheckFuncContext &context, const AnchorAttribute &var_type, bool &done);
  static Status IsNoPaddingContinuousNodeConflict(const CheckFuncContext &context, bool &has_conflict);
  static Status IsContinuousOutAndInConflict(const CheckFuncContext &context, bool &has_conflict);
  static std::vector<OutDataAnchorPtr> GetAllRealInPeer(const NodePtr &node);
  static bool AllRealInputsAreTheSameOutAnchor(const NodePtr &node);
  static bool IsRefFromVar(const OutDataAnchorPtr &out_anchor, NodePtr &src_node,
                           const SymbolToAnchors &symbol_to_anchors,
                           const AnchorToSymbol &anchor_to_symbol);
  static void MarkInTypes(const NodePtr &node, const ComputeGraphPtr &graph,
                                        const std::unique_ptr<Checker> &checker);
  static void MarkOutTypes(const NodePtr &node, const ComputeGraphPtr &graph,
                           const SymbolToAnchors &symbol_to_anchors,
                           const AnchorToSymbol &anchor_to_symbol,
                           const std::unique_ptr<Checker> &checker);
  static Status FindConflictNodes(const NodeIndexIOVector &all_nodes, AnchorSet &result,
                         const GraphInfo &graph_info, const std::unique_ptr<Checker> &checker);
  static Status IsGraphExistMemConflictSymbol(const ComputeGraphPtr &graph,
                                              const AnchorToSymbol &anchor_to_symbol,
                                              const SymbolToAnchors &symbol_to_anchors,
                                              bool &has_conflict);
  static Status CheckOneSubGraphConflict(const ComputeGraphPtr &sub_graph,
                                                std::vector<InDataAnchorPtr> &in_data_anchors);
  static Status CheckIfConflict(const NodePtr &ctrl_node, std::vector<InDataAnchorPtr> &in_data_anchors);
  static Status CheckCaseConflict(const NodePtr &ctrl_node, std::vector<InDataAnchorPtr> &in_data_anchors);
  static Status IsWhileNeedInsertIdentityAtOutput(const ComputeGraphPtr &while_body,
                                    bool &is_need_insert);
  static Status GetWhileBodyDataToNetoutputNodes(const ComputeGraphPtr &while_body,
                                                          std::vector<NodePtr> &data_nodes_change,
                                                          std::set<uint32_t> &bypass_index_no_change);
  static Status CheckWhileConflict(const NodePtr &ctrl_node,
                                   std::vector<InDataAnchorPtr> &conflit_anchors);
  static bool IsCtrlNodeSubgraphExistMemConflictSymbol(const ComputeGraphPtr &graph);
 private:
  using SubGraphSolveConflictCall =
    std::function<Status(const NodePtr &ctrl_node, std::vector<InDataAnchorPtr> &in_data_anchors)>;
  static std::map<std::string, SubGraphSolveConflictCall> check_subgraph_conflict_call;
};

};  // namespace ge

#endif  // GE_GRAPH_PASSES_MEM_CHECK_UTIL_H_
