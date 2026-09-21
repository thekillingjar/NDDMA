/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_GRAPH_COMMON_H_
#define FUSION_ENGINE_INC_COMMON_GRAPH_COMMON_H_

#include <map>
#include <string>
#include <utility>
#include <mutex>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/aicore_util_types.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
using ScopeNodeMap = std::map<int64_t, std::vector<ge::NodePtr>>;
class GraphCommImpl;
using GraphCommImplPtr = std::unique_ptr<GraphCommImpl>;

class GraphComm {
public:
  explicit GraphComm(const string &engine_name);
  GraphComm(const string &engine_name, std::shared_ptr<std::mutex> &lock_ptr);
  virtual ~GraphComm();
  GraphComm(const GraphComm &in) = delete;
  GraphComm &operator=(const GraphComm &in) = delete;

  Status Initialize();

  Status GetscopeNodeMap(const ge::ComputeGraph &graph, ScopeNodeMap &fusion_map);

  void GetEdgeNode(const vector<FusionDataFlow> &fus_input_edge_list,
                   const vector<FusionDataFlow> &fus_output_edge_list,
                   vector<ge::NodePtr> &edge_nodes) const;
  Status CopyFusionOpNodes(const vector<FusionDataFlow> &fus_input_edge_list,
                           const vector<FusionDataFlow> &fus_output_edge_list,
                           const vector<ge::NodePtr> &fus_nodelist,
                           const ge::OpDescPtr &fusion_op_desc,
                           const ge::ComputeGraphPtr &fusion_graph) const;

  Status CopyFusionOpEdges(const ge::ComputeGraph &orig_graph, const ge::ComputeGraphPtr &fusion_graph);

  Status GetNodeDataFlowMap(const ge::NodePtr &fus_node,
                            std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> &fusion_op_anchors_map,
                            ge::kFusionDataFlowVec_t &fus_dataflow_list, const int &map_type);

  Status GetFusionNodeEdgeList(std::vector<ge::NodePtr> &fus_nodelist,
                               const std::unordered_set<ge::NodePtr> &node_set,
                               std::vector<FusionDataFlow> &fus_input_edge_list,
                               std::vector<FusionDataFlow> &fus_output_edge_list);
  void ClearFusionSrc();

  void ClearFusionDst();

  void AddFusionOutputSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor, const int32_t &fusion_src_index,
                          const std::pair<string, ge::AnchorPtr> &node_dataindex_pair);

  void AddFusionInputSrc(const uint32_t &src_op_id,
                         const ge::AnchorPtr &src_anchor,
                         const int32_t &fusion_dst_index,
                         const std::pair<string, ge::AnchorPtr> &node_dataindex_pair);

  void SaveFusionDst(const uint32_t &dst_op_id, const ge::AnchorPtr &dst_anchor);

  bool IsFusionDstExist(const uint32_t &dst_op_id,
                        const ge::AnchorPtr &dst_anchor);

  bool GetFusionSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor,
                    int32_t &fusion_src_index, int32_t &fusion_dst_index);

  Status GetFusionNodeCtrlEdgeList(vector<ge::NodePtr> &fus_nodelist,
                                   const std::unordered_set<ge::NodePtr> &node_set,
                                   vector<FusionDataFlow> &fus_input_ctrl_edge_list,
                                   vector<FusionDataFlow> &fus_output_ctrl_edge_list);

  Status MergeFusionNodeEdgeList(const ge::NodePtr &fus_node, const vector<ge::NodePtr> &fus_nodelist,
                                 const vector<FusionDataFlow> &fus_input_edge_list,
                                 const vector<FusionDataFlow> &fus_output_edge_list);

  Status MergeFunctionNodeEdgeList(const ge::NodePtr &fus_node, const std::vector<ge::NodePtr> &fus_nodelist,
                                   const std::vector<FusionDataFlow> &fus_input_edge_list,
                                   const std::vector<FusionDataFlow> &fus_output_edge_list);

  Status MergeFusionNodeCtrlEdgeList(const ge::NodePtr &fus_node,
                                     const vector<ge::NodePtr> &fus_nodelist,
                                     const vector<FusionDataFlow> &fus_input_ctrl_edge_list,
                                     const vector<FusionDataFlow> &fus_output_ctrl_edge_list);

  Status AddFunctionNodeInputDesc(const ge::OpDescPtr &fus_op, vector<fe::FusionDataFlow> &fus_input_edge_list);

  Status AddFunctionNodeOutputDesc(const ge::OpDescPtr &fus_op, vector<fe::FusionDataFlow> &fus_output_edge_list);

  ge::OpDescPtr CreateFunctionOp(vector<ge::NodePtr> &node_vec) const;

  Status AddOuterDataEdgeInputs(const vector<fe::FusionDataFlow> &input_edge_list,
                                ge::CompleteGraphBuilder &builder) const;

  Status AddOuterCtrlEdgeOutputs(const vector<fe::FusionDataFlow> &output_ctrl_edge_list,
                                 ge::CompleteGraphBuilder &builder) const;

  Status AddOuterDataEdgeOutputs(const vector<fe::FusionDataFlow> &output_edge_list,
                                 ge::CompleteGraphBuilder &builder) const;

  Status EstablishConnection(const ge::NodePtr &node, const std::unordered_set<std::string> &node_name_set,
                             ge::CompleteGraphBuilder &builder) const;

  Status ConnectionSubGraphToRootGraph(const ge::ComputeGraphPtr &sub_graph,
                                       ge::ComputeGraphPtr &root_graph,
                                       std::shared_ptr<std::mutex> &graph_common_lock_ptr);

  Status CreateFunctionOpSubGraph(const ge::NodePtr &function_node,
                                  std::vector<ge::NodePtr> &node_vec,
                                  vector<fe::FusionDataFlow> &input_edge_list,
                                  vector<fe::FusionDataFlow> &output_edge_list,
                                  vector<fe::FusionDataFlow> &output_ctrl_edge_list);

  ge::NodePtr TransSingleSubGraph(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &node_vec);

  string GetEngineName();

  Status ConvertFixpipePartitionCalledOp(ge::ComputeGraph& graph, const AICoreMode &ai_core_mode);
  Status UnfoldFuncOp(ge::ComputeGraph& graph);
  bool IsTuningMode() const;
private:
  Status MergeFusionNodeInputEdgeList(const ge::NodePtr &fus_node, const std::vector<ge::NodePtr> &fus_nodelist,
                                      const std::vector<FusionDataFlow> &fus_input_edge_list);

  Status MergeFunctionNodeInputEdgeList(const ge::NodePtr &fus_node, const vector<ge::NodePtr> &fus_nodelist,
                                        const vector<FusionDataFlow> &fus_input_edge_list);

  Status MergeFusionNodeOutputEdgeList(const ge::NodePtr &fus_node,
                                       const std::vector<FusionDataFlow> &fus_output_edge_list);

  Status CopyOutDataEdges(const ge::NodePtr &src_node,
                          const ge::NodePtr &node,
                          const ge::ComputeGraphPtr &fusion_graph);

  Status CopyControlEdges(const ge::NodePtr &src_node,
                          const ge::NodePtr &node,
                          const ge::ComputeGraphPtr &fusion_graph);

  void CommonCollectFixpipeRelativeNodes(const ge::NodePtr &node,
                                         std::vector<ge::NodePtr> &fixpipe_nodes) const;

  void UnlinkOldEdges(const vector<FusionDataFlow> &fus_output_edge_list);

  uint64_t GetAtomicId() const;

  Status SetFixpipeToFuncOp(ge::ComputeGraph& graph, const AICoreMode &ai_core_mode);

  void CollectSwitchMergeNodes(const ge::NodePtr &cube_node, std::vector<ge::NodePtr> &fixpipe_nodes) const;

  void CollectSwitchInMaxDeepth(const ge::NodePtr &cur_node, std::vector<ge::NodePtr> &to_fold_nodes,
    uint8_t cur_deepth, uint8_t &switch_deepth, bool &exist_switch) const;

  void CollectFixpipe(const ge::NodePtr &cube_node, std::vector<ge::NodePtr> &fixpipe_nodes) const;

  string engine_name_;

  uint64_t function_graph_id_;
  std::vector<FusionOpSrc> exist_fusion_src_list_;
  std::vector<FusionOpDst> exist_fusion_dst_list_;

  // std::vector<std::multimap<std::string, uint32_t>>
  ge::kFusionDataFlowVec_t fusion_input_dataflow_list_;

  // std::vector<std::multimap<std::string, ge::AnchorPtr>>
  ge::kFusionDataFlowVec_t fusion_output_dataflow_list_;

  GraphCommImplPtr graph_comm_impl_ptr_;
  std::shared_ptr<std::mutex> graph_common_lock_ptr_;
};

using GraphCommPtr = std::shared_ptr<GraphComm>;
} // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_GRAPH_COMMON_H_
