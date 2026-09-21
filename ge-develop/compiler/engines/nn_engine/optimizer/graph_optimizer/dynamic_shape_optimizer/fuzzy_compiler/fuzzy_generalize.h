/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_FUZZY_COMPILER_FUZZY_GENERALIZE_H_
#define COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_FUZZY_COMPILER_FUZZY_GENERALIZE_H_

#include <vector>
#include <map>
#include <unordered_set>
#include <utility>

#include "common/optimizer/optimize_utility.h"
#include "ops_store/binary_kernel_info.h"
#include "ops_store/op_kernel_info.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "graph/utils/graph_utils.h"
#include "fusion_config_manager/fusion_priority_manager.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/input_node_generalize.h"

namespace fe {
const uint32_t kDowngradesTimeMax = 30;
const uint32_t MAX_DECENT_TIMES = 5;
const uint32_t MAX_UNSET_DYNAMIC_TIMES = 5;
const uint32_t TOTAL_DECENT_TIMES = 4;
const int64_t MAX_RANGE_UPPER = -1;
const std::string ORIGIN_SHAPE_RANGE = "origin_shape_range";
const std::string SHAPE_RANGE = "shape_range";

class FuzzyGeneralize {
 public:
  FuzzyGeneralize(ge::OptimizeUtility *optimize_utility,
                  const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr,
                  const FusionPriorityMgrPtr &fusion_priority_mgr_ptr);
  ~FuzzyGeneralize();

  FuzzyGeneralize(const FuzzyGeneralize &) = delete;
  FuzzyGeneralize &operator=(const FuzzyGeneralize &) = delete;

  Status GeneralizeGraph(ge::ComputeGraph &graph);

 private:
  Status FeedInputsRootSet(const ge::NodePtr &node_ptr, const NodeGeneralInfoPtr &node_info_ptr) const;

  void CheckIsSubGraphNode(const ge::NodePtr &node_ptr, const NodeGeneralInfoPtr &node_info_ptr) const;

  bool CheckIsExternalNode(const ge::NodePtr &node) const;

  bool CheckIsFirstNode(const ge::NodePtr &node) const;

  Status CheckAndUpdateLimitedNodes(const OpStoreAdapterPtr &op_store_adapter,
                                    const std::vector<ge::NodePtr> &limited_nodes, bool &generalize_flag);

  Status LimitNode(const ge::NodePtr &limited_node) const;

  Status InputNodeDowngrades(const ge::NodePtr &cur_node, const std::vector<size_t> &upper_limited_input_indexs,
                             const std::vector<size_t> &lower_limited_input_indexs, bool &is_range_out);

  Status UpdateDynamicShapeToNewInputNode(const std::unordered_set<ge::NodePtr> &external_input_nodes,
                                          const std::map<std::string, ge::NodePtr> &new_input_nodes) const;

  Status GetReshapeTypeByOpStore(const ge::NodePtr &node, const std::string &input_name,
                                 std::string &reshape_type) const;

  Status GetReshapeType(const ge::Format &origin_format, ge::GeShape &ori_shape, const ge::NodePtr &first_node,
                        const std::string &input_name, std::string &reshape_type) const;

  Status CorrectCAxisByOriginalFormat(const ge::Format &origin_format, const ge::NodePtr &input_node,
                                      const ge::NodePtr &first_node, const std::string &input_name) const;

  Status CorrectInputNodeCAxisByFirstNode(const ge::NodePtr &input_node) const;

  Status CAxisCorrection() const;

  Status UpdateDynamicShapeToFirstNode(const ge::NodePtr &ori_input_node) const;

  Status UpdateDynamicShapeToOriginalGraph(const ge::ComputeGraph &graph) const;

  Status UpdateDynamicShapeToNewBakGraph(const ge::ComputeGraph &graph) const;

  Status GraphDynamicShapeInfer(const OpStoreAdapterPtr &op_store_adapter, ge::ComputeGraphPtr &graph_bak,
                                const ge::ComputeGraphPtr &ori_graph);

  Status GraphPreprocessing(const ge::ComputeGraph &graph, const OpStoreAdapterPtr &op_store_adapter);

  Status RangeDecent(const ge::NodePtr &external_node, uint32_t &decent_times);

  Status GetCurNodeInfo(const ge::NodePtr &node, const OpStoreAdapterPtr &op_store_adapter,
                        NodeGeneralInfoPtr &node_info_ptr);

  Status GetRangeLimitValue(const OpStoreAdapterPtr &op_store_adapter,
                            const NodeGeneralInfoPtr &node_info_ptr, const ge::NodePtr &node);

  Status CalDecentSteps(const ge::NodePtr &external_node, const ge::OpDescPtr &opdesc);

  Status SingleOpDowngrades(const ge::NodePtr &external_node, const bool &is_upper_limited, bool &is_range_out);

  Status Downgrades(const ge::NodePtr &cur_node, const bool &is_upper_limited,
      const std::vector<size_t> &limited_input_indexs, bool &is_range_out);

  Status InitOriginalGraphInfos(const ge::ComputeGraph &graph);

  void UpdateOpAttrs(const ge::GeTensorDescPtr &cur_tensor_desc, const ge::GeTensorDescPtr &ori_tensor_desc,
      const ge::OpDescPtr &op_desc_ptr, bool is_input) const;

  bool IsOpDescExtInputOutputTensor(const ge::OpDescPtr &opdesc, bool input) const;

  static bool IsAllUnknownDimNum(const ge::GeShape &shape);

  static bool IsAllUnknownDim(const std::vector<std::pair<int64_t, int64_t>> &range);

  static bool IsAllUnKnownShapeAndRange(const ge::GeTensorDescPtr &output_tensor_desc);

  static void FurtherGeneralizeShapeAndRange(const ge::GeTensorDescPtr &output_tensor_desc);

  Status DoFurtherGeneralize(const std::string &graph_name) const;

  static void GeneralizeNodeToUnknownDim(const ge::NodePtr &single_node);

  ge::OptimizeUtility *optimize_utility_;
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;
  std::map<ge::NodePtr, NodeGeneralInfoPtr> node_info_map_;
  std::unordered_set<ge::NodePtr> external_input_nodes_;
  std::map<std::string, ge::NodePtr> original_input_nodes_;
  std::vector<ge::NodePtr> limited_range_nodes_;
  std::map<std::string, uint32_t> decent_times_count_;
  std::map<std::string, std::vector<double>> decent_steps_;
  bool is_range_limited_graph_ = false;
  bool is_need_generalize_graph_ = true;
  bool is_single_op_graph_ = false;
};
} // namespace fe
#endif // COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_FUZZY_COMPILER_FUZZY_GENERALIZE_H_
