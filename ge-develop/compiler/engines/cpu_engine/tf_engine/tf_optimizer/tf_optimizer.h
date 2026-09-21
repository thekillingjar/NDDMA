/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TF_OPTIMIZER_H_
#define AICPU_TF_OPTIMIZER_H_

#include <unordered_map>

#include "proto/tensorflow/function.pb.h"
#include "common/aicpu_graph_optimizer/optimizer.h"
#include "aicpu_ops_kernel_info_store/op_struct.h"

namespace aicpu {
using OptimizerPtr = std::shared_ptr<Optimizer>;
struct SubGraphInfo
{
  // record node in new sub graph
  std::unordered_map<string, ge::NodePtr> new_node_map;
  // record all input data anchors for new fused node
  std::vector<ge::InDataAnchorPtr> in_data_anchors;
  // record all output data anchors for new fused node
  std::vector<ge::OutDataAnchorPtr> out_data_anchors;
  // record map of (output data anchors of new fused node) with (peer input data anchors belong to other nodes)
  std::map<ge::OutDataAnchorPtr, std::vector<ge::InDataAnchorPtr>> out_data_anchor_map;
  // record map of (input control anchors of new fused node) with (peer out control anchors belong to other nodes)
  std::map<ge::InControlAnchorPtr, std::vector<ge::OutControlAnchorPtr>> in_control_anchor_map;
  // record map of (out control anchors of new fused node) with (peer in control anchors belong to other nodes)
  std::map<ge::OutControlAnchorPtr, std::vector<ge::InControlAnchorPtr>> out_control_anchor_map;
};

class TfOptimizer : public Optimizer {
public:
  /**
   * Destructor
   */
  virtual ~TfOptimizer() = default;

  /**
   * @return optimizer object
   */
  static OptimizerPtr Instance();

  /**
   * Optimize original graph, using in graph preparation stage
   * @param graph Computation graph
   * @param all_op_info, map used to store op full information
   * @return status whether this operation success
   */
  ge::Status OptimizeOriginalGraph(ge::ComputeGraph &graph,
                                   const std::map<std::string, OpFullInfo> &all_op_info) const override;

  /**
   * optimizer fused graph, find ops can be fused in the graph and fuse it
   * @param graph, Compute graph
   * @param all_op_info, map used to store op full information
   * @return status whether this operation success
   */
  ge::Status OptimizeFusedGraph(ge::ComputeGraph &graph,
                                const std::map<std::string, OpFullInfo> &all_op_info) const override;

  /**
   * init optimizer
   * @return status whether this operation success
   */
  ge::Status Initialize() override;

private:
  /**
   * Contructor
  */
  TfOptimizer() = default;

  // Copy prohibited
  TfOptimizer(const TfOptimizer& tf_optimizer) = delete;

  // Move prohibited
  TfOptimizer(const TfOptimizer&& tf_optimizer) = delete;

  // Copy prohibited
  TfOptimizer& operator=(const TfOptimizer& tf_optimizer) = delete;

  // Move prohibited
  TfOptimizer& operator=(TfOptimizer&& tf_optimizer) = delete;

  /**
   * mark node can be fused
   * @param graph, Compute graph
   * @param tf_node_cluster_map, node cluster
   * @param tf_isolated_node_map, node can not be fused
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden")))
  ge::Status MarkNodeForFusion(const ge::ComputeGraph &graph,
                               const std::map<std::string, OpFullInfo> &all_op_info,
                               std::unordered_map<std::string, std::vector<ge::NodePtr>> &tf_node_cluster_map,
                               std::unordered_map<std::string, ge::NodePtr> &tf_isolated_node_map) const;
   /**
   * mark node can be fused
   * @param graph, Compute graph
   * @param cluster_node_map, node cluster
   * @param isolated_node_map, node can not be fused
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden"))) ge::Status MarkNodeForFusionOfFfts(
      const ge::ComputeGraph &graph, const std::map<std::string, OpFullInfo> &all_op_info,
      std::unordered_map<std::string, std::vector<ge::NodePtr>> &tf_node_cluster_map,
      std::unordered_map<std::string, ge::NodePtr> &tf_isolated_node_map) const;

  /**
   * Check op is fusion or not
   * @param op_desc Op desc ptr
   * @param all_op_info op all info
   * @param fuse_flag bool if Can be fused
   * @return status whether this operation success
   */
  ge::Status CheckOpsFusion(ge::OpDescPtr &op_desc_ptr, const std::map<std::string, OpFullInfo> &all_op_info,
                            bool &fuse_flag) const;

  /**
   * Fuse node for graph
   * @param tf_node_cluster Node cluster
   * @param tf_node_cluster_map Node cluster map
   * @param isolated_node_map, node can not be fused
   * @return status whether this operation success
   */
  ge::Status SeparateDiffOpsToMap(const std::vector<ge::NodePtr> &tf_node_cluster,
                                  std::unordered_map<std::string, std::vector<ge::NodePtr>> &tf_node_cluster_map,
                                  std::unordered_map<std::string, ge::NodePtr> &tf_isolated_node_map) const;

  /**
   * Get no safe edge
   * @param node Ge node
   * @param no_safe_flag no safe flag
   * @return status whether this operation success
   */
  ge::Status GetOutNoSafeEdgeList(const ge::NodePtr &node, bool &no_safe_flag) const;

  /**
   * init debug mode of tf ops
  */
  void InitTfDebugMode();

  /**
   * init the min num of fused ops
  */
  void InitOpFusionMinNum();

  /**
   * init the parser used to convert ir to tf ops
  */
  ge::Status InitializeIr2TfParser() const;

  /**
   * Check op is function op or not
   * @param op_desc Op desc ptr
   * @return bool if is function op
  */
  bool CheckIsFunctionOp(ge::OpDescPtr &op_desc) const;

  ge::Status CheckSubGraphSupportFuse(std::vector<ge::NodePtr> &node_cluster,
                                      const SubGraphInfo &sub_graph_info,
                                      std::unordered_map<std::string, ge::NodePtr> &isolated_node_map) const;
  
  /**
   * Optimize node cluster
   * @param graph Computation graph
   * @param node_cluster Node cluster
   * @return status whether this operation success
  */
  ge::Status OptimizeNodeCluster(ge::ComputeGraph &graph,
                                 std::vector<ge::NodePtr> &node_cluster,
                                 std::unordered_map<std::string, ge::NodePtr> &isolated_node_map) const;

  /**
   * Fuse node for graph
   * @param graph Compute graph
   * @param node_cluster Node cluster
   * @return status whether this operation success
  */
  ge::Status FuseNodesForGraph(ge::ComputeGraph &graph,
                               std::vector<ge::NodePtr> &node_cluster,
                               std::unordered_map<std::string, ge::NodePtr> &isolated_node_map) const;

  /**
   * Optimize isolated node
   * @param node Ge node
   * @return status whether this operation success
  */
  ge::Status OptimizeIsolatedNode(ge::NodePtr &node) const;

  /**
   * Create node def for ge node
   * @param node Ge node
   * @return status whether this operation success
  */
  ge::Status CreateNodeDefForGeNode(ge::NodePtr &node) const;

  /**
   * set asyncFlag attr for ge node
   * @param node ge node
   * @return status whether this operation success
  */
  ge::Status SetAsyncFlag(ge::NodePtr &node) const;

  /**
   * Insert nodes to graph
   * @param sub_graph sub graph
   * @param node_cluster nodes need to be fused
   * @param sub_graph_info Sub graph info
   * @return status whether this operation success
  */
  ge::Status InsertNodesToSubGraph(ge::ComputeGraphPtr &sub_graph,
                                   std::vector<ge::NodePtr> &node_cluster,
                                   SubGraphInfo &sub_graph_info) const;

  /**
   * Link nodes in sub graph according to node connection in original graph
   * @param original_graph Original graph
   * @param new_node_map all node in new sub graph
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden")))
  ge::Status LinkInnerAnchorsForSubGraph(ge::ComputeGraph &original_graph,
                                         std::unordered_map<std::string, ge::NodePtr> &new_node_map) const;

  /**
   * Collect node function
   * @param node_cluster Node cluster
   * @param library Function def library
   * @return status whether this operation success
  */
  ge::Status CollectNodeFuncs(const std::vector<ge::NodePtr> &node_cluster,
                              domi::tensorflow::FunctionDefLibrary *library) const;

  /**
   * Rebuild output desc for fused op
   * @param out_data_anchors Output data anchors
   * @param fused_op_desc Fused op desc
   * @return status whether this operation success
  */
  ge::Status RebuildOutputDesc(const std::vector<ge::OutDataAnchorPtr> &out_data_anchors,
                               ge::OpDescPtr &fused_op_desc) const;

  /**
   * Rebuild input desc for fused op
   * @param in_data_anchors Input data anchors
   * @param fused_op_desc Fused op desc
   * @return status whether this operation success
  */
  ge::Status RebuildInputDesc(const std::vector<ge::InDataAnchorPtr> &in_data_anchors,
                              ge::OpDescPtr &fused_op_desc) const;

  /**
   * Rebuild fusion node
   * @param sub_graph_info Sub graph info
   * @param fused_node Fused node
   * @return status whether this operation success
  */
  ge::Status RebuildFusionNode(SubGraphInfo &sub_graph_info,
                               ge::NodePtr &fused_node) const;

  /**
   * Add attributes by scence
   * @param node_cluster Node cluster
   * @param fused_node Fused node
   * @return status whether this operation success
  */
  ge::Status LabelAttributesByScence(const std::vector<ge::NodePtr> &node_cluster,
                                     const ge::NodePtr &fused_node) const;

  /**
   * Save fusion node mapping relations
   * @param node_cluster Cluster nodes befor optimizer
   * @param out_data_anchors Output data anchors
   * @param mapping_op_desc op desc
   * @return status whether this operation success
  */
  ge::Status SaveFusionNodeMappingRelations(const std::vector<ge::NodePtr> &node_cluster,
                                            const std::vector<ge::OutDataAnchorPtr> &out_data_anchors,
                                            ge::OpDescPtr &mapping_op_desc) const;

  /**
   * Get tf_op_flag & op_async_flag
   * @param op_desc_ptr Op desc ptr
   * @param all_op_info All op info
   * @param tf_op_flag Indicates whether a TF op
   * @param op_async_flag Indicates whether a async op
   * @return status whether this operation success
  */
  ge::Status GetOpFlagAndAsyncFlag(ge::OpDescPtr &op_desc_ptr,
                                   const std::map<std::string, OpFullInfo> &all_op_info,
                                   bool &tf_op_flag, bool &op_async_flag) const;

  void UpdataOpInfos(ge::ComputeGraph &graph,
                    const std::map<std::string, OpFullInfo> &all_op_info) const;

  ge::Status SetExceptionAbortAttr(ge::NodePtr &node) const;
  bool IsTfDebugFusion(ge::OpDescPtr &op_desc) const;

  /**
   * Rebuild tf node cluster map
   * @param graph compute graph
   * @param tf_node_cluster_map original tf node cluster map
   * @return unordered_map rebuild tf node cluster map
  */
  std::unordered_map<std::string, std::vector<ge::NodePtr>> RebuildTfNodeClusterMap(
    const ge::ComputeGraph &graph,
    std::unordered_map<std::string, std::vector<ge::NodePtr>> &tf_node_cluster_map) const;

 private:

  // singleton instance
  static OptimizerPtr instance_;
  // min op fusion number
  uint64_t op_fusion_min_num_ = 2U;
  // tf debug mode(0: off, 1: on)
  bool tf_debug_mode_ = false;
};
} // namespace aicpu
#endif // AICPU_TF_OPTIMIZER_H_
