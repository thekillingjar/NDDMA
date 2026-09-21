/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CHECK_DEPENDENCY_ANALYZER_H_
#define GE_GRAPH_BUILD_MEMORY_CHECK_DEPENDENCY_ANALYZER_H_

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <stack>

#include "runtime/rt.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "common/large_bm.h"
#include "ge/ge_api_types.h"

namespace ge {
struct WrapperInfo {
  WrapperInfo(const ComputeGraphPtr &sub_graph, const size_t bitmap_size);
  std::vector<int64_t> direct_nodes;
  LargeBitmap subgraph_all_nodes_bitmap;
};

struct MemCheckParam {
  const ge::Node *const a;
  uint32_t a_output_index;
  const ge::Node *const b;
  uint32_t b_output_index;
};

class DependencyAnalyzer {
 public:
  /// @ingroup ge
  /// @brief constructor, will modify anchor_to_symbol and symbol_to_anchors, merge symbol for no paddign continuous
  /// input/output node
  /// @param [in] graph
  /// @param [in] anchor_to_symbol
  /// @param [in] symbol_to_anchors
  DependencyAnalyzer(ComputeGraphPtr &graph, AnchorToSymbol &anchor_to_symbol, SymbolToAnchors &symbol_to_anchors);
  ~DependencyAnalyzer() = default;

  /// @ingroup ge
  /// @brief Initialize maps, which stores the mapping of nodes to its reusable nodes set
  /// @return error code of Init
  Status Init();

  /// @ingroup GE
  /// @brief Check if the a_out_index output of a can reuse the b_out_index output of b
  /// @param [in] a
  /// @param [in] a_output_index, output index of a
  /// @param [in] b
  /// @param [in] b_output_index, output index of b
  /// @return bool
  bool CanAReuseB(const Node *const a, uint32_t a_output_index, const Node *const b, uint32_t b_output_index) const;

  /// @ingroup GE
  /// @brief tell why a can not reuse b
  /// @param [in] a
  /// @param [in] a_output_index, output index of a
  /// @param [in] b
  /// @param [in] b_output_index, output index of b
  /// @return td::string
  std::string WhyACannotReuseB(const Node *const a, uint32_t a_output_index, const Node *const b,
                               uint32_t b_output_index);

  /// @ingroup GE
  /// @brief rebuild reach nodes table, and print log.
  /// After calling Debug, CanAReuseB and WhyACannotReuseB cannot be called again
  /// @return void
  void Debug();

  bool IsInit() const {
    return init_flag_;
  }

 private:
  // extend anchor_to_symbol_ and symbol_to_anchors_
  void ExtendSymbolMapByContinuousMemory();
  void OutContinuousMergeSymbol(const Node *cur_node);
  void InContinuousMergeSymbol(const Node *cur_node);
  void MergeOutAnchorSymbol(const std::vector<OutDataAnchorPtr> &out_anchors);
  void MergeSymbols(const std::vector<NodeIndexIO> &anchors_vec, const std::unordered_set<std::string> &symbols_set);

  bool CanAReuseBInner(const Node *const a, uint32_t a_output_index, const Node *const b,
                       const uint32_t b_output_index) const;
  bool CanReachAllSameBlockAnchor(const Node *const a, const uint32_t a_output_index, const uint32_t out_node_id) const;

  // init same_out_block_nodes_table_
  void InitSameOutBlockNodesMap();
  std::list<const Node *> GetAllUseSameBlockNodesByType(const Node *node, uint32_t out_index, IOType type);
  std::list<const Node *> GetAllUseSameInBlockNodes(const Node *node, uint32_t out_index);
  std::list<OutDataAnchorPtr> GetAllUseSameOutBlockAnchors(const Node *node, uint32_t out_index);

  const std::list<NodeIndexIO> &GetSymbolNodeIndexIOList(const Node *const node, const uint32_t output_index) const;

  // init stream to nodes and id map
  void InitPreAndNextNodeOnSameStreamTable();
  void InitNodeTypeTable();
  void InitWrapperInfo();
  void DeInitWrapperInfo();
  const Node *GetPreNodeOnSameStream(const int64_t topo_id) const;
  const Node *GetNextNodeOnSameStream(const int64_t topo_id) const;

  // init reach_nodes_table_
  void InitReachNodesMap();
  void InitReachNodesMap(const Node *root_node, const vector_bit_t &searched_flag);
  void ExtendReachNodesMapCrossSubGraph();
  void WrapperNotReachSubGraphNodes();
  std::unordered_set<int64_t> GetAllNodeIdsSameSymbolWithWrapper(const Node *node);

  void PushPreNode(const Node *node, std::stack<const Node *> &node_stack) const;
  void PushNextNode(const Node *node, std::stack<const Node *> &nodes_stack) const;
  void PushNotVisited(const vector_bit_t &visited_flag, std::stack<const Node *> &nodes_stack) const;

  Status InitTables();
  void InitOutDataAnchorId(const std::vector<const Node *> &all_nodes);

  // debug
  void ErrorLogReachNodes(const std::list<const Node *> &nodes) const;
  void ErrorLogReachNodes(const Node *node) const;
  void ErrorLogDependNodes(const std::list<OutDataAnchorPtr> &out_anchors) const;
  void ErrorLogDependNodes(const Node *node) const;
  void ErrorLogOutSymbol(const Node *node, uint32_t out_index) const;
  void ErrorLogSymbol(const NodeIndexIO &node_index_io) const;

  bool CheckParam(const Node *const a, uint32_t a_output_index, const Node *const b, uint32_t b_output_index) const;
  bool IsOutNodeSkip(const Node *const node, const Node *const origin_node) const;
  void DebugWrapperInfo() const;
  void WhyACannotReuseBInner(const MemCheckParam &param, const std::list<const Node *> &b_output_nodes,
                             std::stringstream &ss);
 private:
  ComputeGraphPtr compute_graph_;
  ComputeGraphPtr root_graph_;
  bool init_flag_ = false;

  AnchorToSymbol &anchor_to_symbol_;
  SymbolToAnchors &symbol_to_anchors_;

  // compute_graph_中所有的节点
  std::vector<const Node *> graph_all_nodes_;
  size_t id_max_ = 0U;

  // topo id作为索引，保存的是同stream上前一个节点的指针
  std::vector<int64_t> pre_node_on_same_stream_table_;
  // topo id作为索引，保存的是同stream上后一个节点的指针
  std::vector<int64_t> next_node_on_same_stream_table_;
  // topo id作为索引，保存的是node的指针
  std::vector<const Node *> id_2_node_table_;

  // 父节点到Wrapper信息的映射, 条件算子有多个子图
  std::map<const Node *, std::vector<WrapperInfo>> wrapper_info_holder_;
  // 父节点到所在子图内父节点的映射
  std::map<const Node *, std::vector<const Node *>> parent_2_childs_map_;
  std::unordered_set<const Node *> parents_on_root_graph_;
  vector_bit_t is_wrapper_;
  std::vector<int64_t> wrapper_ids_;

  // 由于引用节点及连续内存的场景的存在，有一些节点输出anchor内存使用同一个内存块，把这些anchor id保存到一个列表。
  std::list<std::list<size_t>> same_block_anchor_id_holder_;
  std::vector<const std::list<size_t> *> same_block_anchor_ids_;

  // 为了reusable_nodes_bit_map_table_中能够使用id表示out data anchor，将其映射为一个整数
  std::unordered_map<OutDataAnchorPtr, size_t> out_data_anchor_2_id_map_;
  std::vector<OutDataAnchorPtr> id_2_out_data_anchor_table_;

  // 保存了节点能到达的节点集合，（控制边到达也统计在内, 考虑了stream）,topo_id作为索引
  // reach_nodes_bit_map_table_[topoid0]是一个bit map，长度为图上所有节点个数，对应bit为1的表示topoid0能够到达的节点
  std::vector<LargeBitmap> reach_nodes_bit_map_table_;

  // for debug
  bool debug_mode = false;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_CHECK_DEPENDENCY_ANALYZER_H_
