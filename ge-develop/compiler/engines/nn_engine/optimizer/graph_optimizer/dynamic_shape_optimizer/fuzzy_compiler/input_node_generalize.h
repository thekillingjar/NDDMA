/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_FUZZY_COMPILER_INPUT_NODE_GENERALIZE_H_
#define COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_FUZZY_COMPILER_INPUT_NODE_GENERALIZE_H_
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "graph/compute_graph.h"
#include "graph/utils/type_utils.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "fusion_config_manager/fusion_attr_manager.h"
namespace fe {
std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges);
std::string ShapeToString(const std::vector<int64_t> &shapes);
void UpdateTensorDesc(const ge::GeTensorDescPtr &src, ge::GeTensorDescPtr &dst);
using FusionAttrManagerPtr = std::shared_ptr<FusionAttrManager>;

struct GraphType {
    bool is_limited_graph;
    bool is_single_op_graph;
};
class InputNodeGeneralize {
 public:
  InputNodeGeneralize(const std::unordered_set<ge::NodePtr> &input_nodes, const GraphType &graph_type,
                      const std::map<ge::NodePtr, NodeGeneralInfoPtr> &node_info_map,
                      const OpStoreAdapterPtr &op_store_adapter,
                      const FusionAttrManagerPtr &fusion_attr_mgr);

  explicit InputNodeGeneralize(const OpStoreAdapterPtr &op_store_adapter);

  ~InputNodeGeneralize();

  Status GeneralizeAllInputNodesInGraph();

  Status GeneralizeOneNode(const ge::NodePtr &node_ptr, const NodeGeneralInfoPtr &node_info_ptr) const;

 private:
  std::vector<ge::ComputeGraphPtr> GetSubgraphsByCurNode(const ge::NodePtr &node_ptr) const;

  Status MergeRangeWithUpperLimitMax(const std::pair<int64_t, int64_t> &upper_limit_max_range,
                                     const std::pair<int64_t, int64_t> &range, const size_t &dim_index,
                                     std::vector<std::pair<int64_t, int64_t>> &dst_shape_range) const;

  Status MergeTensorDesc(const ge::GeTensorDescPtr &src, const ge::GeTensorDescPtr &dst) const;

  Status GetParentNodeBySubGraphNode(const ge::NodePtr &sub_node, ge::NodePtr &parent_node) const;

  Status UpdateSubGraphInputToRootGraph(const std::unordered_set<ge::NodePtr> &sub_graph_input_nodes,
                                        const ge::ComputeGraphPtr &sub_graph) const;

  Status GeneralizeSubGraphs(const ge::NodePtr &root_graph_first_node);

  Status UpdateFirstNodeTensorDescToInputNodes(const ge::NodePtr &first_node);

  Status GeneralizeFirstNodeOfGraph(ge::NodePtr &first_node);

  Status UnlimitedNodeGeneralize(const ge::NodePtr &unlimited_node, const NodeGeneralInfoPtr &node_info_ptr) const;

  Status LimitedNodeGeneralize(const ge::NodePtr &limited_node, const NodeGeneralInfoPtr &node_info_ptr) const;

  Status SetValueDependFlagToInputNodes(const ge::NodePtr &first_node, const NodeGeneralInfoPtr &node_info_ptr) const;

  Status MergeRange(const std::vector<std::pair<int64_t, int64_t>> &src_range,
                    std::vector<std::pair<int64_t, int64_t>> &dst_range) const;

  std::unordered_set<ge::NodePtr> input_nodes_;
  std::unordered_set<ge::NodePtr> prime_nodes_;
  GraphType graph_type_;
  std::map<ge::NodePtr, NodeGeneralInfoPtr> node_info_map_;
  OpStoreAdapterPtr op_store_adapter_;
  FusionAttrManagerPtr fusion_attr_mgr_;
};
} // namespace fe
#endif // COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_FUZZY_COMPILER_INPUT_NODE_GENERALIZE_H_
