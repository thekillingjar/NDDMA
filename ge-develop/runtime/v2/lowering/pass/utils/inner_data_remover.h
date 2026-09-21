/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_INNER_DATA_REMOVER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_INNER_DATA_REMOVER_H_
#include <set>
#include <queue>
#include <utility>

#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/math_util.h"
#include "graph/fast_graph/fast_node.h"
#include "graph/fast_graph/execute_graph.h"

#include "core/builder/node_types.h"
namespace gert {
namespace bg {
/**
 * 消除一个Node所有子图中InnerData的Remover，本类支持重入
 */
class SubgraphNodeInnerDataRemover {
 public:
  // New interfaces implemented based on the execution graph
  explicit SubgraphNodeInnerDataRemover(ge::FastNode *const parent_node);

  /**
   * 初始化，请确保调用其他public方法前，首先调用Init函数，Init函数仅需要调用一次
   * @return
   */
  ge::graphStatus Init();

  /**
   * 删除本Node中其中一个子图的所有无用InnerData
   * @param subgraph 外部传入的子图必须是本Node的子图，否则会失败
   * @return
   */

  ge::graphStatus RemoveUnusedInnerDataNodes(ge::ExecuteGraph *const subgraph, bool &changed);

  /**
   * 删除本Node中所有无用的输入，并更新子图中InnerData的index
   * @return
   */
  ge::graphStatus RemoveUnusedInputs();

 private:
  ge::graphStatus UpdateInnerDataIndex(const std::vector<int32_t> &old_indexes_to_new);

 private:
  // private member varibles for ExecuteGraph
  ge::FastNode *parent_fast_node_;
  std::map<ge::ExecuteGraph *, std::vector<ge::FastNode *>> exe_subg_to_indexed_inner_data_nodes_;

  size_t variant_num_ = 0U;
  std::vector<size_t> input_indexes_to_used_count_;
};

/**
 * 消除图上无用的InnerData，所谓无用的InnerData，是指没有连接任何输出的InnerData。
 * 本类支持重入，支持的方式是每次执行完`RemoveUnusedInnerDataNodes`后，清空所有状态
 */
class InnerDataRemoverForSubgraphs {
 public:
  explicit InnerDataRemoverForSubgraphs(std::vector<ge::ExecuteGraph *> graphs);
  ge::graphStatus RemoveUnusedInnerDataNodes();
  ge::graphStatus RemoveUnusedInnerDataNodes(bool &changed);

 private:
  // New interfaces implemented based on the execution graph
  ge::graphStatus AddGraph(ge::ExecuteGraph *const graph);
  ge::ExecuteGraph *GetNextExeGraph();
  bool HasUndoneExeGraphs() const;
  ge::graphStatus RemoveAllInnerDataNodes(ge::ExecuteGraph *const graph, bool &changed);
  void Clear();

 private:
  struct FastNodeToBeDone {
    std::set<ge::ExecuteGraph *> unpass_subgraphs;
    SubgraphNodeInnerDataRemover inputs_stat;
  };

  // private member varibles for ExecuteGraph
  std::vector<ge::ExecuteGraph *> init_exe_subgraphs_;
  std::queue<ge::ExecuteGraph *> exe_graphs_to_be_done_;
  std::set<ge::ExecuteGraph *> exe_graphs_to_be_done_set_;
  std::map<ge::FastNode *, FastNodeToBeDone> cached_fast_nodes_;
  std::map<ge::ExecuteGraph *, FastNodeToBeDone *> exe_subgraphs_to_cached_node_;
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_INNER_DATA_REMOVER_H_
