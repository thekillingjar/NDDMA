/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_MODEL_HYBRID_MODEL_BUILDER_H_
#define GE_HYBRID_MODEL_HYBRID_MODEL_BUILDER_H_

#include <vector>
#include <queue>
#include <memory>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/node.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/model/node_item.h"
#include "common/model/ge_model.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"

namespace ge {
namespace hybrid {
class HybridModelBuilder {
 public:
  explicit HybridModelBuilder(HybridModel &hybrid_model);
  ~HybridModelBuilder() = default;
  Status Build();
  Status BuildForSingleOp();
  static Status UnfoldSubgraphs(const ComputeGraphPtr &root_graph, ComputeGraphPtr &merged_graph);

 private:
  static Status RecoverShapeConsistency(const ComputeGraph &root_graph);
  static Status UpdateAnchorStatus(const NodePtr &node);
  static Status DoUnlinkDataAnchors(const OutDataAnchorPtr &out_data_anchor, const InDataAnchorPtr &in_data_anchor);
  static Status DoLinkDataAnchors(const OutDataAnchorPtr &out_data_anchor, const InDataAnchorPtr &in_data_anchor);
  static NodePtr GetPeerNode(const InDataAnchorPtr &in_data_anchor);

  static Status GetPeerNodeAcrossSubGraphs(const NodePtr &data_node, NodePtr &peer_node, int32_t &peer_out_index);
  Status CopyConstantData(const NodePtr &node, const GeTensor &tensor,
                          const std::unique_ptr<TensorValue> &var_tensor) const;
  static Status MergeInputNodes(ComputeGraph &compute_graph);
  static Status MergeInputInData(const NodePtr &node, const NodePtr &wrapped_node, std::set<NodePtr> &root_nodes);
  static Status MergeNetOutputNode(ComputeGraph &compute_graph);
  static Status MergeNetOutputInData(const NodePtr &parent_node, const OpDescPtr &net_output_desc,
                                     const InDataAnchorPtr &in_data_anchor);
  static Status UnfoldSubgraph(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &parent_graph,
                               ComputeGraph &sub_graph, const uint32_t depth = 0U);
  static Status BuildInputMapping(GraphItem &graph_item,
                                  const std::vector<NodeItem *> &input_nodes,
                                  const bool is_root_graph);
  static Status ResolveRefIo(NodeItem &node_item);
  Status InitOverflowAddr() const;
  Status InitNodeBinMode();
  Status BuildOutputMapping(GraphItem &graph_item, const NodeItem &node_item, const bool is_root_graph) const;
  Status ValidateParams() const;
  Status LoadGraph();
  Status CopyGraph() const;
  Status LoadGeModel(ComputeGraph &sub_graph, const GeModelPtr &ge_model);
  static Status InitHcclExecutorOnDemand(const GeModelPtr &ge_model);
  Status LoadTask(NodeItem &node_item) const;
  Status LoadTasks() const;
  Status IdentifyVariableOutputs(NodeItem &node_item, const ComputeGraphPtr &subgraph);
  Status BuildNodeItem(const NodePtr &node, NodeItem &node_item);
  Status GetOrCreateNodeItem(const NodePtr &node, NodeItem *&node_item) const;
  Status ParseForceInfershapeNodes(const NodePtr &node, NodeItem &node_item) const;
  Status CollectParallelGroups(NodeItem &node_item);
  Status ParseDependentInputNodes(NodeItem &node_item);
  Status ParseDependentInData(const NodeItem &node_item, std::set<NodePtr> &dependent_for_execution) const;
  Status ParseDependencies(NodeItem &node_item,
                           std::set<NodePtr> &dependent_for_shape_inference);
  Status ParseDependentForFusedSubgraph(const NodeItem &node_item,
                                        std::set<ge::NodePtr> &dependencies) const;
  Status ParseDependentByParallelGroup();
  Status IndexTaskDefs();
  Status IndexTaskDefs(const ComputeGraphPtr &sub_graph, const GeModelPtr &ge_model);
  static void LoadTbeKernelBinToOpDesc(const ModelTaskType task_type, const GeModelPtr &ge_model,
                                       const NodePtr &node);
  static void LoadSgtKernelBinToOpDesc(const OpDesc &op_desc, const ComputeGraphPtr &sub_graph,
                                       const GeModelPtr &ge_model);
  Status IndexSpecialNodes();
  Status InitRuntimeParams();
  Status InitModelMem();
  Status InitWeights() const;
  Status TransAllVarData() const;
  Status CopyVarData() const;
  Status VarNodeToTensor(const NodePtr &var_node, std::unique_ptr<TensorValue> &tensor) const;
  Status AssignUninitializedConstantOps() const;
  Status InitConstantOps() const;
  Status InitVariableTensors() const;
  void *GetOrCreateVarMem(const std::string &var_name,
                          const OpDescPtr &var_desc,
                          const rtMemType_t memory_type) const;
  Status LoadDynamicSubgraph(const ComputeGraphPtr &graph, const bool is_root_graph);
  Status LoadDynamicNodeItem(GraphItem &graph_item, const NodePtr &node, std::vector<NodeItem *> &input_nodes,
                             std::map<size_t, std::pair<uint32_t, uint32_t>> &profiling_nodes);
  Status SetStageCache(const ComputeGraph &graph, const GraphItem &stage_graph) const;
  Status CheckForObserver(const ComputeGraph &graph) const;
  Status ParseVarOutputs(NodeItem &node_item) const;
  Status LoadKnownShapedSubgraph(const ComputeGraph &graph, const NodeItem &parent_node_item);
  Status LoadKnownNodeItem(GraphItem &graph_item, const NodePtr &node, const OpDescPtr &wrapper_op_desc) const;
  Status RecoverGraphUnknownFlag() const;
  Status CheckAicpuOpList() const;
  Status CreateProfilingNodeBefore(GraphItem &graph_item, const NodePtr &node, uint32_t &prev_num) const;
  Status CreateProfilingNodeAfter(GraphItem &graph_item, const NodePtr &node, uint32_t &post_num) const;
  static Status GenerateFpProfilingTask(const OpDescPtr &op_desc, std::vector<domi::TaskDef> &task_def_list);
  static Status GenerateBpProfilingTask(const OpDescPtr &op_desc, std::vector<domi::TaskDef> &task_def_list);
  static Status GenerateEndProfilingTask(const OpDescPtr &op_desc, std::vector<domi::TaskDef> &task_def_list);
  static Status GenerateArProfilingTask(const OpDescPtr &op_desc, const int64_t profiling_log_id,
                                        std::vector<domi::TaskDef> &task_def_list);
  Status OptimizeDependenciesForConstantInputs();
  Status Convert2HostTensor(const NodePtr &node, const int64_t node_id, const uint32_t output_idx);

  Status RelinkNextIteration() const;
  Status BuildProfilingControl(GraphItem &graph_item,
                               const std::map<size_t, std::pair<uint32_t, uint32_t>> &nodes) const;
  Status BuildFrameGroupIndex(NodeItem &node_item);
  Status BuildFrameGroupIndexForEnter(NodeItem &node_item, const int64_t ctrl_flow_group);

  Status BuildControlFlowGroup(GraphItem &graph_item, const NodePtr &node, NodeItem &node_item) const;
  Status CreateNormalNodeGroup(const NodePtr &node, NodeItem &node_item) const;
  Status CreateMergeEnterGroup(const NodePtr &node, NodeItem &node_item) const;
  Status CreateMergeIterationGroup(const NodePtr &node, NodeItem &node_item) const;
  Status CreateStreamActiveGroup(const NodePtr &node, NodeItem &node_item) const;
  Status CreateStreamSwitchGroup(const NodePtr &node, NodeItem &node_item) const;
  Status CreateNextIterationGroup(const NodePtr &node, NodeItem &node_item) const;

  Status CreateSwitchGroup(const NodePtr &node, NodeItem &node_item) const;
  Status CreateNotImplement(const NodePtr &node, NodeItem &node_item) const;
  Status ValidateFftsPlusSubNodeItem(const NodeItem &node_item) const;

  const std::string &ModelName() const {
    return hybrid_model_.model_name_;
  }

  const NodeItem *GetNodeItem(const NodePtr &node) const;
  NodeItem *MutableNodeItem(const NodePtr &node) const;
  Status InitAippInfoAndType();
  Status InitFileConstantInDevice(const int64_t output_size, const OpDescPtr &op_desc,
                                  const TensorValue &tensor_value) const;
  Status InitFileConstantInHost(const int64_t output_size, const OpDescPtr &op_desc,
                                std::unique_ptr<TensorValue> &var_tensor) const;
  Status InitFileConstantOps();

  GeRootModelPtr ge_root_model_;
  std::map<std::string, GeModelPtr> subgraph_models_;
  std::map<std::string, NodePtr> constant_op_nodes_;
  std::map<std::string, NodePtr> stream_merge_op_nodes_;
  std::map<std::string, NodePtr> next_iteration_op_nodes_;
  std::map<int64_t, int64_t> parent_frame_group_;
  std::map<std::string, std::set<NodeItem *>> parallel_group_to_nodes_;
  std::map<NodeItem *, std::set<std::string>> node_to_parallel_groups_;

  HybridModel &hybrid_model_;
  std::map<NodePtr, std::vector<std::pair<int32_t, NodePtr>>> node_ref_inputs_;

  RuntimeParam &runtime_param_;
  std::shared_ptr<VarManager> var_manager_ = nullptr;

  // std::map<known_node_item, std::map<output_idx, constant_node>>
  std::map<NodeItem *, std::map<uint32_t, NodePtr>> known_subgraph_constant_output_refs_;

  // std::map<dst_node_item, std::vector<output_idx, src_node_item>>
  std::map<NodeItem *, std::vector<std::pair<uint32_t, NodeItem *>>> host_input_value_dependencies_;

  // Establish the relationship between the file ID and the file path.
  std::map<std::string, std::string> file_id_and_path_map_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_MODEL_HYBRID_MODEL_BUILDER_H_
