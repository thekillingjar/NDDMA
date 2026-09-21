/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_OPTIMIZER_H_
#define DVPP_ENGINE_COMMON_DVPP_OPTIMIZER_H_

#include "util/dvpp_error_code.h"
#include "common/optimizer/graph_optimizer.h"

namespace dvpp {
class DvppOptimizer {
 public:
  /**
   * @brief Construct
   */
  DvppOptimizer() = default;

  /**
   * @brief Destructor
   */
  virtual ~DvppOptimizer() = default;

  // Copy constructor prohibited
  DvppOptimizer(const DvppOptimizer& dvpp_optimizer) = delete;

  // Move constructor prohibited
  DvppOptimizer(const DvppOptimizer&& dvpp_optimizer) = delete;

  // Copy assignment prohibited
  DvppOptimizer& operator=(const DvppOptimizer& dvpp_optimizer) = delete;

  // Move assignment prohibited
  DvppOptimizer& operator=(DvppOptimizer&& dvpp_optimizer) = delete;

  /**
   * @brief optimize graph prepare, before infer shape, use to set dynamic shape
   * @param graph computation graph
   * @return status whether success
   */
  DvppErrorCode OptimizeGraphPrepare(ge::ComputeGraph& graph);

  /**
   * @brief optimize original graph, using in graph preparation stage
   * @param graph computation graph
   * @return status whether success
   */
  DvppErrorCode OptimizeOriginalGraph(ge::ComputeGraph& graph);

  /**
   * @brief optimize original graph
   *        using for conversion oprater insert in graph prepration stage
   * @param graph computation graph
   * @return status whether success
   */
  DvppErrorCode OptimizeOriginalGraphJudgeInsert(ge::ComputeGraph& graph);

  /**
   * @brief optimize fused graph
   * @param graph computation graph
   * @return status whether success
   */
  DvppErrorCode OptimizeFusedGraph(ge::ComputeGraph& graph) const;

  /**
   * @brief optimize whole graph
   * @param graph computation graph
   * @return status whether success
   */
  DvppErrorCode OptimizeWholeGraph(ge::ComputeGraph& graph) const;

 private:
  /**
   * @brief check if single op graph and dvpp support it
   * @param graph computation graph
   * @return whether support
   */
  bool CheckSupportedSingleOpGraph(ge::ComputeGraph& graph) const;

  /**
   * @brief set graph to dynamic shape
   * @param graph computation graph
   * @return status whether success
   */
  DvppErrorCode SetGraphDynamicShape(ge::ComputeGraph& graph);

  /**
   * @brief do some special optimize
   * @param graph computation graph
   * @param node current node
   * @return status whether success
   */
  DvppErrorCode SpecialOptimize(
      ge::ComputeGraph& graph, ge::NodePtr& node) const;

  /**
   * @brief Sub and Mul fused into NormalizeV2
   * @param graph computation graph
   * @param mul_node current node Mul
   * @return status whether success
   */
  DvppErrorCode SubAndMulFusedIntoNormalizeV2(
      ge::ComputeGraph& graph, ge::NodePtr& mul_node) const;

  /**
   * @brief set dvpp op attrs, such as engine name
   * @param op_desc_ptr op info
   * @return status whether success
   */
  DvppErrorCode SetDvppOpAttr(ge::OpDescPtr& op_desc_ptr) const;

  /**
   * @brief set memory type
   * @param op_desc_ptr op info
   * @return status whether success
   */
  DvppErrorCode SetMemoryType(ge::OpDescPtr& op_desc_ptr) const;

  /**
   * @brief check if enable dvpp
   * @param op_desc_ptr op info
   * @return enable or not
   */
  bool EnableDvppEngine(ge::OpDescPtr& op_desc_ptr) const;

  /**
   * @brief insert node
   * @param graph computation graph
   * @param node src node
   * @return status whether success
   */
  DvppErrorCode InsertNode(ge::ComputeGraph& graph, ge::NodePtr& node);

  /**
   * @brief insert if graph
   * @param graph computation graph
   * @param node src node
   * @return status whether success
   */
  DvppErrorCode InsertIfGraph(
      ge::ComputeGraph& graph, ge::NodePtr& node) const;

  /**
   * @brief build if then subgraph
   * @param then_graph then subgraph computation graph shared_ptr
   * @param node src node
   * @return status whether success
   */
  DvppErrorCode BuildIfThenGraph(
      ge::ComputeGraphPtr& then_graph, ge::NodePtr& node) const;

  /**
   * @brief build if else subgraph
   * @param else_graph else subgraph computation graph shared_ptr
   * @param op_desc src node op desc
   * @return status whether success
   */
  DvppErrorCode BuildIfElseGraph(
      ge::ComputeGraphPtr& else_graph, ge::OpDescPtr& op_desc) const;

  /**
   * @brief create op_type node and insert before src node
   * @param graph computation graph shared_ptr
   * @param node src node
   * @param op_type insert op type
   * @return insert node shared_ptr or nullptr
   */
  ge::NodePtr InsertGraphNodeBefore(ge::ComputeGraphPtr& graph,
      ge::NodePtr& node, const std::string& op_type) const;

  /**
   * @brief insert node after src node
   * @param node src node
   * @param insert_node insert node
   * @return status whether success
   */
  DvppErrorCode InsertNodeAfter(ge::NodePtr& node,
      ge::NodePtr& insert_node) const;

  /**
   * @brief create op_type node and insert after src node
   * @param graph computation graph shared_ptr
   * @param node src node
   * @param op_type insert op type
   * @return insert node shared_ptr or nullptr
   */
  ge::NodePtr InsertGraphNodeAfter(ge::ComputeGraphPtr& graph,
      ge::NodePtr& node, const std::string& op_type) const;

  /**
   * @brief insert Data node in subgraph
   * @param subgraph subgraph computation graph shared_ptr
   * @param node src node
   * @return insert node shared_ptr or nullptr
   */
  DvppErrorCode InsertDataNodeInIfSubgraph(ge::ComputeGraphPtr& subgraph,
      ge::NodePtr& node) const;

  /**
  * @brief create op_type node
  * @param graph computation graph shared_ptr
  * @param op_type insert op type
  * @return insert node shared_ptr or nullptr
  */
  ge::NodePtr GraphAddNode(ge::ComputeGraphPtr& graph,
      const std::string& op_type) const;

  /**
  * @brief create reduce maen node
  * @param graph computation graph shared_ptr
  * @return insert node shared_ptr or nullptr
  */
  ge::NodePtr GraphAddAICpuReduceMeanNode(ge::ComputeGraphPtr& graph) const;

  /**
  * @brief add edge of adjust contrast with mean node
  * @param contrast_with_mean_node contrast with mean node shared_ptr
  * @param contrast_node contrast node shared_ptr
  * @return status whether success
  */
  DvppErrorCode AddAdujustContrastWithMeanEdge(
      ge::NodePtr& contrast_with_mean_node, ge::NodePtr& contrast_node) const;
  /**
  * @brief insert adjust contrast with mean node
  * @param graph computation graph shared_ptr
  * @param contrast_node contrast node shared_ptr
  * @return new node shared_ptr or nullptr
  */
  ge::NodePtr InsertAdujustContrastWithMean(ge::ComputeGraphPtr graph_ptr,
      ge::NodePtr& contrast_node) const;

  /**
  * @brief insert aicpu raduce mean node
  * @param graph computation graph shared_ptr
  * @param contrast_node contrast node shared_ptr
  * @param contrast_with_mean_node contrast with mean node shared_ptr
  * @return new node shared_ptr or nullptr
  */
  ge::NodePtr InsertAICpuReduceMean(ge::ComputeGraphPtr graph_ptr,
      ge::NodePtr& contrast_node, ge::NodePtr& contrast_with_mean_node) const;

  /**
  * @brief get const tensor for reduce mean node
  * @param contrast_node contrast node shared_ptr
  * @return new tensor shared_ptr or nullptr
  */
  ge::GeTensorPtr GetConstTensorForReduceMean(ge::NodePtr& contrast_node) const;

  /**
  * @brief insert const node for reduce mean node
  * @param graph computation graph shared_ptr
  * @param contrast_node contrast node shared_ptr
  * @param contrast_with_mean_node contrast with mean node shared_ptr
  * @return new node shared_ptr or nullptr
  */
  ge::NodePtr InsertConstForReduceMean(ge::ComputeGraphPtr graph_ptr,
      ge::NodePtr& contrast_node, ge::NodePtr& reduce_mean_node) const;

  /**
  * @brief insert rgb to gray node
  * @param graph computation graph shared_ptr
  * @param contrast_node contrast node shared_ptr
  * @param reduce_mean_node reduce mean node shared_ptr
  * @return new node shared_ptr or nullptr
  */
  ge::NodePtr InsertRgbToGrayscale(ge::ComputeGraphPtr graph_ptr,
      ge::NodePtr& contrast_node, ge::NodePtr& reduce_mean_node) const;

  /**
  * @brief optimier adjust contrast node to a sub graph
  * @param graph computation graph shared_ptr
  * @param contrast_node contrast node shared_ptr
  * @return status whether success
  */
  DvppErrorCode InsertAdjustContrastGraph(ge::ComputeGraph& graph,
      ge::NodePtr& contrast_node) const;

  /**
   * @brief set aicpu op attrs, such as engine name
   * @param op_desc_ptr op info
   * @return status whether success
   */
  DvppErrorCode SetAiCpuOpAttr(ge::OpDescPtr& op_desc_ptr) const;

  /**
   * @brief set subgraph and parent graph data index
   * @param data_node subgraph Data node
   * @param data_index subgraph Data node data index
   * @param output_node subgraph NetOutput node
   * @param output_index subgraph NetOutput node data index
   * @return status whether success
   */
  DvppErrorCode SetSubgraphDataIndex(ge::NodePtr& data_node,
      uint32_t data_index, ge::NodePtr& output_node,
      uint32_t output_index) const;

  /**
   * @brief set subgraph parent session id
   * @param then_graph then subgraph
   * @param else_graph else subgraph
   * @param parent_graph_session_id parent graph session id
   * @return status whether success
   */
  DvppErrorCode SetSubgraphParentSessionId(
      ge::ComputeGraphPtr& then_graph, ge::ComputeGraphPtr& else_graph,
      std::string& parent_graph_session_id) const;

  /**
   * @brief get parent graph session id
   * @param graph computation graph shared_ptr
   * @param parent_graph_session_id parent graph session id
   * @return status whether success
   */
  DvppErrorCode GetParentGraphSessionId(ge::ComputeGraphPtr graph,
      std::string& parent_graph_session_id) const;
}; // class DvppOptimizer
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_OPTIMIZER_H_
