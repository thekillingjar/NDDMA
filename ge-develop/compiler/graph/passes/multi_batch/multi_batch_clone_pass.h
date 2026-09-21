/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MULTI_BATCH_CLONE_PASS_H_
#define GE_GRAPH_PASSES_MULTI_BATCH_CLONE_PASS_H_

#include <map>
#include <string>
#include <vector>

#include "graph/passes/graph_pass.h"

namespace ge {
class MultiBatchClonePass : public GraphPass {
 public:
  explicit MultiBatchClonePass(const uint64_t session_id) : session_id_(session_id) {}
  ~MultiBatchClonePass() override = default;
  Status Run(ComputeGraphPtr graph);

 private:
  /// @ingroup ge
  /// @brief Collect input output node from original graph.
  /// @param [in] const ComputeGraphPtr &graph: original graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CollectIoNodes(const ComputeGraphPtr &graph);
  Status InitParamsOfGetNext(const NodePtr &node);

  /// @ingroup ge
  /// @brief Create nodes for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateRootGraph(const ComputeGraphPtr &graph);
  Status GetInputAndOutputNum(uint32_t &input_num, uint32_t &output_num);

  /// @ingroup ge
  /// @brief Create index data node for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @param [in] NodePtr shape_node: index data node, DATA or GETDYNAMICDIMS type.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateIndexDataNode(const ComputeGraphPtr &graph, NodePtr &shape_node);

  Status CreateGetDynamicDimsNode(const ComputeGraphPtr &graph, NodePtr &shape_node);

  /// @ingroup ge
  /// @brief Create index const node for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @param [in] NodePtr node: index const node.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateIndexConstNode(const ComputeGraphPtr &graph, NodePtr &node);

  /// @ingroup ge
  /// @brief Create index node for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateIndexNode(const ComputeGraphPtr &graph);
  Status AddAttrForGetDynamicDims(const NodePtr &shape_node) const;
  Status LinkGetNextToGetDynamicDims(const NodePtr &getnext_node, const NodePtr &shape_node) const;
  Status LinkGetDynamicDimsToNetOutput(const NodePtr &output_node);

  /// @ingroup ge
  /// @brief Create input node for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateInputNode(const ComputeGraphPtr &graph);
  Status LinkEdgeForGetNext(const NodePtr &getnext_node, size_t &case_input_index);

  /// @ingroup ge
  /// @brief Set max shape to Data node in root graph.
  /// @param [in] const NodePtr &data: data in Root/Case graph.
  /// @return 0: SUCCESS / others: FAILED
  Status SetMaxShape(const NodePtr &data);
  Status SetMaxShapeToData(const NodePtr &node, size_t out_anchor_index);

  /// @ingroup ge
  /// @brief Set max shape to Data/GetNext node in root graph.
  /// @param [in] const std::vector<int64_t> &shapes: dims of shape.
  /// @param [in] const NodePtr &data: data in Root/Case graph.
  /// @param [in] GeShape &data_shape: dims of data node.
  /// @param [in] size_t out_anchor_index: out anchor index of data node.
  /// @return 0: SUCCESS / others: FAILED
  Status SetShapeToData(const std::vector<int64_t> &shapes, const NodePtr &data, GeShape &data_shape,
                        size_t out_anchor_index) const;
  Status UpdateShapeOfShapeNode(const NodePtr &node, size_t out_anchor_index) const;

  /// @ingroup ge
  /// @brief Create Const node for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateConstNode(const ComputeGraphPtr &graph);
  void ChangeConstToData();
  NodePtr AddOutputNodeToGraph(const ComputeGraphPtr &graph);

  /// @ingroup ge
  /// @brief Create output node for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateOutputNode(const ComputeGraphPtr &graph);

  /// @ingroup ge
  /// @brief Sort user input dims, node name and dynamic dims with index.
  /// @param [in] data_nodes data node.
  /// @return 0: SUCCESS / others: FAILED
  Status SortDynamicDimsWithIndex();

  /// @ingroup ge
  /// @brief Update Data node in Subgraph.
  /// @param [in] const NodePtr &data: data in Subgraph.
  /// @param [in] size_t batch_index: The batch index.
  /// @return 0: SUCCESS / others: FAILED
  Status UpdateSubgraphData(const NodePtr &data, size_t batch_index);

  /// @ingroup ge
  /// @brief Update output_node in Subgraph.
  /// @return 0: SUCCESS / others: FAILED
  Status UpdateSubgraphOutput();

  void GetDataNodes(std::vector<NodePtr> &data_nodes) const;

  int32_t FindIndexForInputDims(const std::string &node_name,
      const std::vector<std::pair<std::string, std::vector<int64_t>>> &data_name_and_shape) const;
  /// @ingroup ge
  /// @brief Create nodes for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Original graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateOriGraph(const ComputeGraphPtr &graph);
  Status NormalizeDataType(const NodePtr &node) const;
  void SetIndexListAndOriginShape(const OpDescPtr &data_desc);
  void GetUnknownDimIndex(const std::vector<int64_t> &shape,
                          std::vector<int32_t> &unknown_dim_index);
  NodePtr CreateDataNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor,
                         size_t data_index);

  static void RenameAndAddRecursiveSubgraphs(const ComputeGraphPtr &graph, const ComputeGraphPtr &batch_graph,
                                             const std::string &suffix);
  Status AddBatchSubgraph(const ComputeGraphPtr &graph, const ComputeGraphPtr &batch_graph, const size_t batch_idx);

  /// @ingroup ge
  /// @brief Create nodes for root graph.
  /// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
  /// @param [in] const ComputeGraphPtr &branch: original graph.
  /// @return 0: SUCCESS / others: FAILED
  Status CreateSubgraphs(const ComputeGraphPtr &graph, const ComputeGraphPtr &branch);

  /// @ingroup ge
  /// @brief Remove subgraph supend output anchor.
  /// @param [in] ComputeGraphPtr &graph: Parent compute graph.
  /// @return 0: SUCCESS / others: FAILED
  Status PruneDirectOutput(const ComputeGraphPtr &graph);

  /// @ingroup ge
  /// @brief Update subgraph suspend output tensor.
  /// @param [in] parent_index: parent index for check.
  /// @param [in] unused_num: total unused tensor.
  /// @return 0: SUCCESS / others: FAILED
  Status UpdateOutputTensor(uint32_t parent_index, uint32_t unused_num);

  Status CheckAndParseDynamicData();

  Status SetCtrlEdgeToShapeNode();

  std::string session_graph_id_;
  std::vector<std::vector<int64_t>> batch_shapes_;

  std::vector<NodePtr> all_data_nodes_;
  std::vector<NodePtr> all_const_nodes_;
  std::vector<NodePtr> all_output_nodes_;
  std::map<uint32_t, std::string> direct_output_;
  std::map<ComputeGraphPtr, NodePtr> all_branch_output_;
  std::map<std::string, std::vector<vector<int64_t>>> data_to_dynamic_info_;
  std::map<int32_t, int32_t> index_to_case_input_;
  NodePtr case_node_;
  size_t data_count_from_getnext_ = 0;
  bool getnext_sink_dynamic_dims_ = false;
  NodePtr shape_node_;
  std::set<NodePtr> out_control_nodes_;
  uint64_t session_id_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MULTI_BATCH_CLONE_PASS_H_
