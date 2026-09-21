/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_MODE_MANUAL_MANUAL_THREAD_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_MODE_MANUAL_MANUAL_THREAD_TASK_BUILDER_H_

#include <tuple>
#include <stack>
#include "task_builder/mode/thread_task_builder.h"
#include "graph/utils/graph_utils.h"

namespace ffts {
using ControlGraphMap = std::map<ge::NodePtr, std::vector<ge::ComputeGraphPtr>>;
using GernaterElement = std::tuple<std::vector<ge::NodePtr>, uint64_t, uint64_t>;
using GernaterElement1 = std::pair<std::vector<ge::NodePtr>, std::vector<uint64_t>>;
using RunContextPtr = std::shared_ptr<ge::RunContext>;
class ManualTheadTaskBuilder : public TheadTaskBuilder {
 public:
  ManualTheadTaskBuilder();
  ~ManualTheadTaskBuilder() override;

  Status Initialize() override;

  Status GenFftsPlusContextId(ge::ComputeGraph &sgt_graph, std::vector<ge::NodePtr> &sub_graph_nodes,
                                uint64_t &ready_context_num, uint64_t &total_context_number,
                                std::vector<ge::NodePtr> &memset_nodes) override;

  Status GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes, std::vector<ge::NodePtr> &sub_graph_nodes,
                            domi::TaskDef &task_def) override;
 private:
  void GenFftsPlusHcclId(const ge::NodePtr &node, uint32_t &contextId) const;
  Status DoWithControlOpImpl(ge::ComputeGraphPtr &cur_graph,
                             ControlGraphMap &controlnode_graphmap,
                             std::stack<ge::NodePtr> &control_node_stacks) const;
  Status PreInsertIfNode(ge::NodePtr &control_node,
                         ControlGraphMap &controlnode_graphmap,
                         std::stack<ge::NodePtr> &control_node_stacks) const;

  Status PreInsertWhileNode(ge::NodePtr &control_node,
                            ControlGraphMap &controlnode_graphmap,
                            std::stack<ge::NodePtr> &control_node_stacks) const;

  Status PreInsertCaseNode(ge::NodePtr &control_node,
                           ControlGraphMap &controlnode_graphmap,
                           std::stack<ge::NodePtr> &control_node_stacks) const;

  ge::NodePtr HasControlNode(ge::ComputeGraphPtr &cur_graph) const;
  Status DoWithFftsPlusCtxIdForControlNode(ControlGraphMap controlnode_graphmap,
                                           const RunContextPtr &contextptr,
                                           std::stack<ge::NodePtr> &control_node_stacks) const;
  Status DoWithFftsPlusContextIdForIfNode(ge::NodePtr &control_node,
                                          std::vector<ge::ComputeGraphPtr> &sub_graphs,
                                          const RunContextPtr &contextptr) const;

  Status DoWithFftsPlusContextIdForWhileNode(ge::NodePtr &control_node,
                                             std::vector<ge::ComputeGraphPtr> &sub_graphs,
                                             const RunContextPtr &contextptr) const;
  Status DoWithFftsPlusContextIdForCaseNode(ge::NodePtr &control_node,
                                            std::vector<ge::ComputeGraphPtr> &sub_graphs,
                                            const RunContextPtr &contextptr) const;
  void DoWithIfNodeStreamActiveNetOutPut(std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  void DoWithWhileNodeStreamActiveNetOutPut(ge::NodePtr &control_node,
                                            std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  void DoWithCaseNodeStreamActiveNetOutPut(std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithIfNodeOhterNode(ge::NodePtr &control_node,
                               std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithWhileNodeOhterNode(ge::NodePtr &control_node,
                                  std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithCaseNodeOhterNode(ge::NodePtr &control_node,
                                 std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithIfElseLabeSetEnter(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithIfElseLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithIfThenLabeSwitch(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithIfThenLabelSet(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithIfThenLabelGoto(ge::NodePtr &control_node,
                               ge::NodePtr &node,
                               ge::ComputeGraphPtr cur_graph,
                               std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithIfNetOutput(ge::NodePtr &node) const;
  Status DoWithIfSwitchActive(ge::NodePtr &node) const;
  Status DoWithIfOther(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithWhileBodyLabeSetEnter(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithWhileBodyLabelGoto(ge::NodePtr &control_node,
                                  ge::NodePtr &node,
                                  ge::ComputeGraphPtr cur_graph,
                                  std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithWhileBodyLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithWhileCondLabelSet(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithWhileCondLabeSwitch(ge::NodePtr &control_node,
                                   ge::NodePtr &node,
                                   ge::ComputeGraphPtr cur_graph,
                                   std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithWhileNetOutput(ge::NodePtr &node) const;
  Status DoWithWhileSwitchActive(ge::NodePtr &control_node,
                                 ge::NodePtr &node,
                                 ge::ComputeGraphPtr cur_graph,
                                 std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithWhileOther(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithCaseLabeSetEnter(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithCaseLabelGoto(ge::NodePtr &control_node,
                             ge::NodePtr &node,
                             ge::ComputeGraphPtr cur_graph,
                             std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithCaseLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithCaseLabeSwitch(ge::NodePtr &control_node,
                              ge::NodePtr &node,
                              ge::ComputeGraphPtr cur_graph,
                              std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status DoWithCaseNetOutput(ge::NodePtr &node) const;
  Status DoWithCaseSwitchActive(ge::NodePtr &node) const;
  Status DoWithCaseOther(ge::NodePtr &control_node, ge::NodePtr &node) const;
  bool IsOrdinaryNode(ge::NodePtr node) const;
  Status GenFftsPlusContextIdWithoutControlType(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                                std::vector<ge::NodePtr> &no_pre_sub_graph_nodes,
                                                ge::ComputeGraph &sgt_graph) const;
  void GenFftsPlusContextIdCommon(ge::NodePtr node, std::vector<ge::NodePtr> &sub_graph_nodes,
                                  uint32_t &contextId) const;
  ge::NodePtr FindNodeByType(ge::ComputeGraphPtr cur_graph, std::string nodetypename) const;
  ge::NodePtr FindLabelSetEenter(ge::ComputeGraphPtr cur_graph) const;
  ge::NodePtr FindLabelSetLeave(ge::ComputeGraphPtr cur_graph) const;
  ge::NodePtr FindLabelSwitch(ge::ComputeGraphPtr cur_graph) const;
  ge::NodePtr FindLabelGoto(ge::ComputeGraphPtr cur_graph) const;
  Status SetIfWhileLastLabelNext(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status SetLabelGoto(ge::ComputeGraphPtr &cur_graph,
                      ge::NodePtr &node,
                      std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  ge::NodePtr FindJumpIfWhileLableX(uint32_t jump_id,
                                    ge::NodePtr &node,
                                    ge::ComputeGraphPtr &cur_graph,
                                    std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status RepalceDataWithReal(ge::NodePtr &control_node, ge::NodePtr &node) const;
  ge::NodePtr FindFinalParentNode(ge::NodePtr &node, uint32_t &max_depth) const;
  Status SetLabelSwitchByIndexJumpNode(ge::NodePtr &control_node,
                                       ge::ComputeGraphPtr &cur_graph,
                                       ge::NodePtr &node,
                                       std::vector<ge::ComputeGraphPtr> &sub_graphs) const;
  Status SetLabelSwitchByIndexAddr(ge::NodePtr &node,
                                   const RunContextPtr &contextptr) const;
  Status DoWithLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const;
  Status DoWithNetOutPut(ge::NodePtr &node) const;
  bool JudgeNodeToNeedDoWith(ge::NodePtr node) const;
  void GenContextIdWithLabelGoto(ge::NodePtr &node, std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const;
  void GenContextIdOhter(ge::NodePtr &node, std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const;
  void GenContextIdWithLabelSet(ge::NodePtr &node, std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const;
  void GenContextIdWithLabelSwitch(ge::NodePtr &node,
                                   std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                   std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const;
  Status GenFftsPlusContextIdForControlNode(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                            std::vector<ge::NodePtr> &no_pre_sub_graph_nodes,
                                            ControlGraphMap controlnode_graphmap) const;
  Status GenFftsPlusContextIdAll(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                 std::vector<ge::NodePtr> &no_pre_sub_graph_nodes,
                                 std::vector<ge::NodePtr> &sub_graph_nodes,
                                 uint64_t &ready_context_num,
                                 uint64_t &total_context_number) const;
  void GenerateAtomicCtx(std::vector<ge::NodePtr> &memset_nodes, domi::FftsPlusTaskDef *ffts_plus_task_def) const;
  void UpdateAtomicSuccList(const SameAtomicNodeMap &same_memset_nodes_map,
                            const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def);
  void UpdateAtomicForOneNode(const ge::NodePtr &atomic_node, const ge::NodePtr &updated_node,
                              domi::FftsPlusTaskDef *ffts_plus_task_def);
  Status GenSuccList(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def);
};
}  // namespace ffts
#endif  // FFTS_ENGINE_TASK_BUILDER_MODE_MANUAL_MANUAL_THREAD_TASK_BUILDER_H_
