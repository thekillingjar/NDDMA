
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_tile_fusion_pass.h
 * \brief
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONCAT_TILE_FUSION_PASS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONCAT_TILE_FUSION_PASS_H_

#include <map>
#include <memory>
#include <string>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/op_info_common.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "transfer_shape_utils.h"


namespace fe {
class ConcatTileFusionPass : public PatternFusionBasePass {
 public:
  ConcatTileFusionPass();
  ~ConcatTileFusionPass() override;
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;
  
private:
  bool GetFusionNodeMap(const ge::NodePtr &concat_node,
                        std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs);
  bool GetFusionNodes(const ge::InDataAnchorPtr &in_data_anchor,
                      std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs);
  void GetTileFusionNodes(const ge::NodePtr &tile_node,
                          std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs);
  bool GetTileBroadcastDims(const ge::NodePtr &tile_node, std::vector<int32_t> &tile_broadcast_dims) const;
  bool CheckTileBroadcastDimsAndInputShape(const std::vector<int64_t> &input_shape,
                                           const std::vector<int32_t> &tile_broadcast_dims) const;
  Status RemoveInputEdges(vector<ge::NodePtr> &src_nodes, vector<ge::NodePtr> &dst_nodes) const;
  Status RemoveConcatEdges(const ge::NodePtr &concat_node, vector<ge::NodePtr> &input_nodes) const;
  Status DoFusion(ge::NodePtr &concat_node, ge::ComputeGraph &graph,
                  std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs);
  Status AddConcatEdges(const ge::NodePtr &concat_node, vector<ge::NodePtr> &input_nodes) const;
  Status InsertConcatOp(ge::OpDescPtr &new_concat_op_desc, const vector<ge::NodePtr> &inputs,
                        ge::NodePtr &tile_node);
  Status DoTileFusion(const ge::NodePtr &concat_node,
                      ge::ComputeGraph &graph, const ge::OpDescPtr &replace_concat_node,
                      const std::vector<std::vector<ge::NodePtr>> &sorted_nodes,
                      std::vector<std::pair<int32_t, ge::NodePtr>> &concat_node_peer_output_nodes);                                          
  bool GetConcatDim(const ge::NodePtr &concat_node, const int32_t &indx);
  Status ParseConcatNode(const ge::NodePtr &concat_node);
  Status ReplaceConcatNodeAddOutputs(ge::OpDescPtr &replace_concat_node, const ge::NodePtr &concat_node,
                                     bool &is_only_op,
                                     std::vector<ge::OutDataAnchorPtr> &concat_node_peer_output_anchors,
                                     std::vector<ge::InDataAnchorPtr> &concat_node_peer_input_anchors) const;
  Status RelinkReplaceConcatNode(const ge::NodePtr &replace_concat_node,
                                 std::vector<ge::OutDataAnchorPtr> &concat_node_peer_output_anchors,
                                 std::vector<ge::InDataAnchorPtr> &concat_node_peer_input_anchors) const;
  Status GetShapeLimited(const ge::DataType &data_type);
  bool GetConcatConstantNode(const ge::NodePtr &concat_node, int32_t &indx);
  bool GetConcatV2ConstantNode(const ge::NodePtr &concat_node, int32_t &indx);
  Status RemoveTileEdge(vector<ge::NodePtr> &src_nodes, vector<ge::NodePtr> &dst_nodes,
                        const ge::NodePtr &concat_node) const;
  Status ConcatAddConcatDimsInfo(const ge::NodePtr &new_concat_node, std::vector<ge::NodePtr> &tile_inputs);
  Status AddConcatDim(const ge::OpDescPtr &replace_concat_node);
  Status AddConcatNode(const ge::NodePtr &concat_node, const ge::OpDescPtr &concat_tile,
                       const ge::NodePtr &tile_node, const std::vector<ge::NodePtr> tile_inputs);
  bool IsValidTileNode(const ge::NodePtr &tile_node, std::vector<int32_t> &tile_broadcast_dims);
  bool IsNeedToFusion(std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs,
                      std::vector<std::vector<ge::NodePtr>> &sorted_nodes) const;
  Status UpdateConcatTileTensor(ge::OpDescPtr &new_concat_op_desc, ge::NodePtr &tile_node, const int64_t &concat_shape,
                                const int64_t &concat_ori_shape, ge::GeTensorDesc &tile_input_output_tensor) const;
  Status GetConcatValueByConcatdim(const vector<ge::NodePtr> &inputs, int64_t &concat_shape_dim,
                                   int64_t &concat_ori_shape_dim, ge::OpDescPtr &new_concat_op_desc) const;
  Status DoAddTileEdges(ge::NodePtr &tile_node, ge::NodePtr &new_concat_node,
                        std::vector<ge::NodePtr> &tile_inputs);
  Status DoReplaceConcatFusion(ge::NodePtr &concat_node, ge::OpDescPtr &replace_concat,
                               ge::ComputeGraph &graph,
                               std::vector<std::pair<int32_t, ge::NodePtr>> &concat_node_peer_output_nodes,
                               std::vector<ge::InDataAnchorPtr> &concat_node_peer_input_anchors);

  Status GetConcatOriginInputInfo(const ge::NodePtr &concat_node,
                                  std::vector<std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>>>
                                      &concat_node_peer_output_info) const;

  Status ReplaceConcatNodeAddInputs(const ge::NodePtr &concat_node,
      ge::OpDescPtr &replace_concat_node, std::vector<std::pair<int32_t, ge::NodePtr>> &concat_node_peer_output_nodes,
      std::vector<ge::OutDataAnchorPtr> &concat_node_peer_output_anchors);
  void GetSortedTiledNodes(std::vector<std::vector<ge::NodePtr>> &sorted_nodes,
                           const std::vector<pair<int32_t, ge::NodePtr>> &tmp_nodes) const;

  void SortContinuityTileNodes(std::vector<ge::NodePtr> &tile_nodes,
                               std::vector<std::vector<ge::NodePtr>> &sorted_nodes) const;

  static Status AddContrEdge(ge::NodePtr &tile_node, ge::NodePtr &new_concat_node);

  int32_t concat_dim_{-1};
  ge::NodePtr concat_dim_input_node_{nullptr};
  uint64_t shape_limited_{1};
};
}  // namespace fe
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONCAT_C_OPTIMIZE_FUSION_PASS_H_
