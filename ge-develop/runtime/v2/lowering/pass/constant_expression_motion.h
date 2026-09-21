/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_CONSTANT_EXPRESSION_MOTION_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_CONSTANT_EXPRESSION_MOTION_H_
#include "pass.h"
#include "lowering/lowering_global_data.h"
#include "graph/fast_graph/edge.h"
namespace gert {
namespace bg {
using EquivalantNodesFromMainToInit = std::unordered_map<ge::FastNode *, ge::FastNode *>;
class ConstantExpressionMotion : public Pass {
 public:
  explicit ConstantExpressionMotion(const LoweringGlobalData *global_data) : global_data_(global_data) {}
  ~ConstantExpressionMotion() override = default;

 public:
  ge::graphStatus Run(ge::ExecuteGraph *const graph, bool &changed) override;
  ge::EdgeSrcEndpoint GetSrcFromInitGraph(ge::FastNode *const node, const size_t variant_num);
  ge::FastNode *GetInitNode();
  ge::EdgeSrcEndpoint GetInitOut(ge::FastNode *const inner_data) const;
  ge::ExecuteGraph *GetInitGraph();
  ge::ExecuteGraph *GetDeInitGraph();
  ge::FastNode *GetInitNetOutput() const;
  ge::FastNode *GetEquivalantNodesInInit(ge::FastNode *const node) const;

 private:
  ge::graphStatus Init(ge::ExecuteGraph *const root_graph);
  ge::ExecuteGraph *GetInitGraph(ge::FastNode *const init_node);
  ge::FastNode *GetDeInitNode();
  ge::graphStatus CollectNoOpSrcNodesFromInitGraph();
  void UpdateNoOpSrcNodes(ge::FastNode *const no_op_node, std::vector<ge::FastNode *> &no_op_src_nodes_next_time);

 private:
  const LoweringGlobalData *global_data_{nullptr};
  ge::ExecuteGraph *root_graph_{nullptr};
  ge::ExecuteGraph *cached_init_graph_{nullptr};
  ge::ExecuteGraph *cached_de_init_graph_{nullptr};
  ge::FastNode *cached_netoutput_from_init_{nullptr};
  ge::FastNode *cached_init_node_{nullptr};
  ge::FastNode *cached_de_init_node_{nullptr};
  EquivalantNodesFromMainToInit cached_equivalant_nodes_;
  std::unordered_map<ge::FastNode *, ge::EdgeSrcEndpoint> cached_inner_datas_to_init_out_;
  std::vector<ge::FastNode *> no_op_src_nodes_;  // 用于缓存需要连接到 NOOP 的 src_node
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_CONSTANT_EXPRESSION_MOTION_H_
