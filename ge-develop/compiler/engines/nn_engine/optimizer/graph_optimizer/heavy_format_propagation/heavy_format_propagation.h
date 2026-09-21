/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_PROPAGATION_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_PROPAGATION_H_

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/fe_inner_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "graph/compute_graph.h"
#include "graph/ref_relation.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_selector.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_supportformats_updater.h"

namespace fe {
constexpr int32_t INVALID_FORMAT_INDEX = 0xffff;
constexpr int32_t INVALID_LAST_NODE_ANCHOR_INDEX = 0xffffffff;
constexpr int32_t NEED_IGNORE = 50331649;

using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;
using FormatDtypeSetterPtr = std::shared_ptr<FormatDtypeSetter>;
using HeavyFormatSelectorPtr = std::shared_ptr<HeavyFormatSelector>;
using HeavyFormatSupportFormatsUpdaterPtr = std::shared_ptr<HeavyFormatSupportFormatsUpdater>;
using RefRelationsPtr = std::shared_ptr<ge::RefRelations>;

struct PeerFormatAndShapeInfo {
  ge::NodePtr current_node;
  ge::Format peer_primary_format = ge::FORMAT_RESERVED;
  int32_t peer_sub_format = 0;
  int32_t peer_c0_format = 0;
  ge::GeShape peer_shape;
  PeerFormatAndShapeInfo(const ge::NodePtr &current_node_param, ge::Format format,
                         int32_t sub_format, int32_t c0_format)
      : current_node(current_node_param),
      peer_primary_format(format),
      peer_sub_format(sub_format),
      peer_c0_format(c0_format) {}
};

class HeavyFormatPropagation {
 public:
  HeavyFormatPropagation(const std::string &engine_name, RefRelationsPtr reflection_builder_ptr);

  ~HeavyFormatPropagation();

  HeavyFormatPropagation(const HeavyFormatPropagation &) = delete;

  HeavyFormatPropagation &operator=(const HeavyFormatPropagation &) = delete;

  Status Initialize();
  /**
   * The main function of distributing heavy format and it is also the
   * interface given to graph optimizer.
   * @return SUCCESS or FAIL
   */
  Status PropagateHeavyFormat(const ge::ComputeGraph &graph);

 private:
  /**
   * Propagate back ward from all input of current node to the peer outputs'
   * owner node.
   * @param last_node: we will not traversing to the node where the heavy
   * format is from.
   * @param format_index: The heavy format index in ops kernel
   * @param heavy_format: The heavy format which is propagated on the main
   * route
   * @return SUCCESS or FAIL
   */
  Status PropagateBackwards(const NodeInfoPtr &node_info, int32_t format_index,
                            FormatSelectionType format_selection_type, std::deque<NodeInfoPtr> &next_node_queue);
  /**
   * Propagate farward from all output of current node to the peer inputs'
   * owner node.
   * @param last_node: we will not traversing to the node where the heavy
   * format is from.
   * @param format_index: The heavy format index in ops kernel
   * @param heavy_format: The heavy format which is propagated on the main
   * route
   * @return SUCCESS or FAIL
   */
  Status PropagateForwards(const NodeInfoPtr &node_info, int32_t format_index,
                           FormatSelectionType format_selection_type, std::deque<NodeInfoPtr> &next_node_queue);

  Status RunPropagation(const NodeInfoPtr &node_info, std::deque<NodeInfoPtr> &next_node_queue);

  /* The heavy format of node_info may not passed to predecessors, because
   * each input/output's format may be different. so we use
   * format_of_current_tensor to represent the real format which is being
   * propagated. */
  void PropagateFarther(const NodeInfoPtr &curr_node_info, const NodeInfoPtr &next_node_info,
                        std::deque<NodeInfoPtr> &next_node_queue);

  Status SetInferFormat(const TensorInfoPtr &tensor_info_ptr);

  Status GetHeavyFormatIndex(const NodeInfoPtr &node_info, const HeavyFormatInfo &heavy_format_info,
                             int32_t &heavy_format_index) const;

  Status GetTensorKernelInfo(const ge::NodePtr &current_node, const TensorInfoPtr &tensor_info_ptr,
                             const OpKernelInfoPtr &op_kernel_info_ptr,
                             InputOrOutputInfoPtr &input_or_output_info) const;

  Status GetFormatAndDtypeFromOpKernel(const ge::NodePtr &current_node, const TensorInfoPtr &tensor_info_ptr,
                                       const OpKernelInfoPtr &op_kernel_info_ptr) const;

  Status SetFormats(const TensorInfoPtr &tensor_info_ptr, const OpKernelInfoPtr &op_kernel_info_ptr);

  Status SetNewShape(const TensorInfoPtr &tensor_info_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                     const ge::GeTensorDescPtr &current_tensor, ge::GeShape &original_shape, ge::GeShape &new_shape);

  Status SetFormatAndDTypeToOpdesc(const TensorInfoPtr &tensor_info_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                   Status set_format_result);

  static bool IsPropagateFromNode(const ge::NodePtr &node,
                                  const OpKernelInfoPtr &op_kernel_info_ptr);

  static void RefreshHeavyOpC0Val(const ge::OpDescPtr &op_desc);

 private:
  std::string engine_name_;

  RefRelationsPtr reflection_builder_ptr_;

  FormatDtypeQuerierPtr format_dtype_querier_ptr_;

  FormatDtypeSetterPtr format_dtype_setter_ptr_;

  HeavyFormatSelectorPtr heavy_format_selector_ptr_;

  HeavyFormatSupportFormatsUpdaterPtr supportformats_updater_ptr_;

  vector<NodeInfoPtr> format_agnostic_nodes_info_;
  /* If the op format agnostic type is
   * FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS, we still need to check
   * whether there is some input or output which is not format agnostic. */
  bool IsAnchorFormatAgnostic(const TensorInfoPtr &tensor_info_ptr, bool is_input, int64_t anchor_idex) const;
  /**
   * Propagate farward from all output of current node to the peer inputs'
   * owner node.
   * @param last_node: we will not traversing to the node where the heavy
   * format is from.
   * @param format_index: The heavy format index in ops kernel
   * @param heavy_format: The heavy format which is propagated on the main
   * route
   * @return SUCCESS or FAIL
   */
  Status PropagateNormalNodeForwards(const NodeInfoPtr &curr_node_info, int32_t format_index,
                                     const TensorInfoPtr &tensor_info_ptr, std::deque<NodeInfoPtr> &next_node_queue);

  /**
   * Propagate farward from all input of sub graph netoutput to peer inputs'
   * owner node.
   * @param last_node: we will not traversing to the node where the heavy
   * format is from.
   * @param format_index: The heavy format index in ops kernel
   * @param heavy_format: The heavy format which is propagated on the main
   * route
   * @return SUCCESS or FAIL
   */
  Status PropagateSubNetoutputForwards(const NodeInfoPtr &curr_node_info, int32_t format_index,
                                       const TensorInfoPtr &tensor_info_ptr, std::deque<NodeInfoPtr> &next_node_queue);

  /**
   * Propagate back ward from all input of current node to the peer outputs'
   * owner node.
   * @param last_node: we will not traversing to the node where the heavy
   * format is from.
   * @param format_index: The heavy format index in ops kernel
   * @param heavy_format: The heavy format which is propagated on the main
   * route
   * @return SUCCESS or FAIL
   */
  Status PropagateNormalNodeBackwards(const NodeInfoPtr &curr_node_info, int32_t format_index,
                                      TensorInfoPtr &tensor_info_ptr, std::deque<NodeInfoPtr> &next_node_queue);

  /**
   * Propagate back ward from all output of sub graph DATA to the peer outputs'
   * owner node.
   * @param last_node: we will not traversing to the node where the heavy
   * format is from.
   * @param format_index: The heavy format index in ops kernel
   * @param heavy_format: The heavy format which is propagated on the main
   * route
   * @return SUCCESS or FAIL
   */
  Status PropagateSubDataBackwards(const NodeInfoPtr &curr_node_info, const TensorInfoPtr &tensor_info_ptr,
                                   std::deque<NodeInfoPtr> &next_node_queue);

  /* If origin format is different, not do propagation
   * Cause may create diff-shape with same format in this edge,
   * will insert more TransData op */
  bool CheckOriFormatEqual(const ge::NodePtr &next_node, const ge::DataAnchorPtr &next_data_anchor,
                           const ge::NodePtr &current_node, const TensorInfoPtr &tensor_info_ptr,
                           const bool &is_forwards) const;

  Status GetNextNodesInfoForwards(std::deque<NodeInfoPtr> &next_node_queue,
                                  const ge::InDataAnchorPtr &peer_in_data_anchor, const NodeInfoPtr &last_node_info,
                                  const TensorInfoPtr &tensor_info_ptr);

  void GetNextNodesInfoBackWards(std::deque<NodeInfoPtr> &next_node_queue,
                                 const ge::OutDataAnchorPtr &peer_out_anchor, const NodeInfoPtr &last_node_info,
                                 const TensorInfoPtr &tensor_info_ptr);

  bool JudgeIsFollowReshapeType(const ge::OpDescPtr &op_desc_ptr, const OpKernelInfoPtr &op_kernel_info_ptr) const;
  
  std::string GetPropagationReshapeType(const TensorInfoPtr &tensor_info_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                        const ge::GeTensorDescPtr &current_tensor) const;

  Status SetReshapeType(const ge::OpDescPtr &op_desc_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                        const ge::Format &ori_format, const TensorInfoPtr &tensor_info_ptr) const;

  bool IsInputAvailable(const ge::OpDescPtr &op_desc_ptr, const ge::InDataAnchorPtr &in_data_anchor_ptr) const;

  bool IsOutputAvailable(const ge::OpDescPtr &op_desc_ptr, const ge::OutDataAnchorPtr &out_data_anchor_ptr) const;

  Status CheckSpecificSubGraphDataOrNetoutput(const TensorInfoPtr &tensor_info_ptr,
                                              bool &is_sub_graph_data_or_net_output,
                                              std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections);

  void FindSameNameVariableNodes(const ge::NodePtr &node_ptr, std::vector<ge::NodePtr> &var_nodes);

  bool CheckBackwardPropagtionNecessary(const NodeInfoPtr &curr_node_info, const ge::InDataAnchorPtr &in_data_anchor,
                                        const TensorInfoPtr &tensor_info_ptr, uint32_t &count_anchor);

  bool CheckForwardPropagtionNecessary(const NodeInfoPtr &curr_node_info, const ge::OpDescPtr &op_desc_ptr,
                                       const ge::OutDataAnchorPtr &out_data_anchor,
                                       const TensorInfoPtr &tensor_info_ptr);

  Status SetHeavyFormat(const ge::InDataAnchorPtr &in_data_anchor, const NodeInfoPtr &curr_node_info,
                        const ge::OpDescPtr &op_desc_ptr, const int32_t &format_index,
                        const TensorInfoPtr &tensor_info_ptr) const;

  bool IsHeavyFormatSupported(const ge::NodePtr &current_node, const NodeInfoPtr &next_node_info) const;

  /* Check whether the propagation will bring benefits to the whole network.
   * For variable, if all the successors can support the heavy format, we consider it as a worthy case.
   * For other nodes, the number of successors which do not support this heavy format is less than or equal to
   * 1, we will consider it as a worthy case. */
  bool JudgePropagationWorthy(const ge::NodePtr &node, const std::deque<NodeInfoPtr> &pending_next_nodes,
                              std::deque<NodeInfoPtr> &next_nodes);

  void AddNodeInfoToQueue(NodeInfoPtr &node_info, std::deque<NodeInfoPtr> &next_node_queue);

  Status GetNodeReshapeType(const NodeInfoPtr &node_info, string &reshape_type, bool only_get_default_type) const;

  bool CheckIsNeedHeavy(const ge::NodePtr &last_node, const NodeInfoPtr &last_node_info,
                        const NodeInfoPtr &next_node_info, const std::string &next_node_reshape_type) const;

  void CreateNextNodeInfo(const ge::NodePtr &next_node, const NodeInfoPtr &last_node_info, ge::Format heavy_format,
                          int32_t sub_format, int32_t c0_format, PropagationInfo &propagation_info,
                          int32_t anchor_index, bool is_input, NodeInfoPtr &next_node_info,
                          std::deque<NodeInfoPtr> &next_node_queue);

  Status SetOpKernelAndTensorMap(const NodeInfoPtr &node_info);

  bool IsFormatAgnosticAnchor(const ge::NodePtr &current_node, const NodeInfoPtr &next_node) const;

  void SetFormatAgnosticType(const ge::OpDescPtr &op_desc_ptr, const NodeInfoPtr &node_info);

  Status UpdateForOneNode(const NodeInfoPtr &node_info) const;

  Status UpdateInputAndOutputForFormatContinuous();

  bool IsNextNodeQualified(const ge::NodePtr &current_node, const NodeInfoPtr &next_node);

  bool CheckSwitchQualified(const std::vector<NodeInfoPtr> &node_list);

  bool IsConstNodeInSubGraph(const ge::NodePtr &node_ptr) const;

  void RollBackFormatAndShape(const ge::NodePtr &node, bool is_sub_graph_weight) const;

  Status UpdateInputForPairType(const ge::NodePtr &current_node, const vector<int64_t> &input_non_format_agnostic_index,
                                const vector<int64_t> &output_non_format_agnostic_index) const;

  Status UpdateOutputForPairType(const ge::NodePtr &current_node,
                                 const vector<int64_t> &input_non_format_agnostic_index,
                                 const vector<int64_t> &output_non_format_agnostic_index) const;

  Status GetPeerInFormat(const vector<int64_t> &output_non_format_agnostic_index,
                          PeerFormatAndShapeInfo &peer_in_format_and_shape) const;

  Status GetPeerOutFormat(const vector<int64_t> &input_non_format_agnostic_index,
                          PeerFormatAndShapeInfo &peer_out_format_and_shape) const;

  Status UpdateInputFormatAndShape(ge::NodePtr &current_node,
                                   const vector<int64_t> &input_non_format_agnostic_index,
                                   const vector<int64_t> &output_non_format_agnostic_index) const;

  Status UpdateOutputFormatAndShape(const ge::NodePtr &current_node,
                                    const vector<int64_t> &input_non_format_agnostic_index,
                                    const vector<int64_t> &output_non_format_agnostic_index) const;

  bool CheckFormatCompatible(const ge::Format &primary_format, const ge::Format &origin_format) const;
};

}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_PROPAGATION_H_
