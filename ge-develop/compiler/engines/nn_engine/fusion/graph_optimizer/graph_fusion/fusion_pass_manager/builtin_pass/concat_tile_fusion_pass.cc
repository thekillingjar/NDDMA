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
 * \file concat_tile_fusion_pass.cc
 * \brief
 */


#include "concat_tile_fusion_pass.h"
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/anchor.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "common/string_util.h"
#include "platform/platform_info.h"
#include "common/math_util.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "common/fe_utils.h"


namespace fe{
namespace {
  const std::string CONCAT_ILEFUSION_PASS = "ConcatTileFusionPass";
  const std::string kPatternConcat = "concat";
  const std::string CONCATV2 = "ConcatV2";
  const std::string CONCAT = "Concat";
  const std::string VectorCoreNum = "vector_core_cnt";
  const std::string AiCoreNum = "ai_core_cnt";
  const std::string VecCalcSize = "vec_calc_size";
  const std::string AICoreSpec = "AICoreSpec";
  const std::string SocInfo = "SoCInfo";
  const uint32_t Float16Size = 2;
  const uint32_t MinTileNum = 2;
  std::set<string> ConcatOp = {CONCAT, CONCATD, CONCATV2D, CONCATV2};
  bool CheckControlEdge(const ge::NodePtr &concat_node) {
    bool res = (concat_node->GetInControlNodes().size() != 0) || (concat_node->GetOutControlNodes().size() != 0);
    FE_LOGD("Check node[%s, %s] has_ctrl_edge is[%d].", concat_node->GetNamePtr(), concat_node->GetTypePtr(), res);
    return res;
  }

  bool CompareNodeIndx(const std::pair<int32_t, ge::NodePtr> &node1, const std::pair<int32_t, ge::NodePtr> &node2) {
    return node1.first < node2.first;
  }
  bool CompareNodeAnchorIndx(const std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>> &node1, 
                             const std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>> &node2) {
    return node1.first < node2.first;
  }
}

ConcatTileFusionPass::ConcatTileFusionPass() {}

ConcatTileFusionPass::~ConcatTileFusionPass() {}

vector<FusionPattern *> ConcatTileFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(CONCAT_ILEFUSION_PASS);
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphFusion][ConcatElemwise] Failed to create a new object."),
           return patterns);
  FE_LOGD("Start to define %s pattern.", CONCAT_ILEFUSION_PASS.c_str());
	pattern->AddOpDesc(kPatternConcat, {CONCAT, CONCATD, CONCATV2D, CONCATV2})
    .SetOutput(kPatternConcat);
  patterns.emplace_back(pattern);
  FE_LOGD("Finished defining %s pattern.", CONCAT_ILEFUSION_PASS.c_str());
  return patterns;
}

Status ConcatTileFusionPass::GetShapeLimited(const ge::DataType &data_type) {
  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != SUCCESS) {
    FE_LOGW("Failed to obtain platform information due to missing SoC version.");
    return FAILED;
  }
  auto data_type_size = ge::GetSizeByDataType(data_type);
  FE_LOGD("Current data_type is %d, which size is [%d].", data_type, data_type_size);
  FE_CHECK(data_type_size == 0, FE_LOGW("The current dtype is not supported."), return FAILED);
  string val_size_str = "";
  string val_core_str = "";
  platform_infos.GetPlatformRes(AICoreSpec, VecCalcSize, val_size_str);
  platform_infos.GetPlatformRes(SocInfo, VectorCoreNum, val_core_str);
  uint64_t val_size = static_cast<uint64_t>(std::atoi(val_size_str.c_str())) * static_cast<uint64_t>(Float16Size);
  uint64_t val_core = static_cast<uint64_t>(std::atoi(val_core_str.c_str()));
  if (val_core == 0) {
    platform_infos.GetPlatformRes(SocInfo, AiCoreNum, val_core_str);
    val_core = static_cast<uint64_t>(std::atoi(val_core_str.c_str()));
  }
  FE_MUL_OVERFLOW(val_size, val_core, shape_limited_);
  shape_limited_ /= data_type_size;
  FE_LOGD("Current soc has vecor_core_num is [%s], vector_calculate_size is [%s], shape_limited_size is [%lu].",
          val_core_str.c_str(), val_size_str.c_str(), shape_limited_);
  return SUCCESS;
}

bool ConcatTileFusionPass::GetTileBroadcastDims(const ge::NodePtr &tile_node,
                                                std::vector<int32_t> &tile_broadcast_dims) const {
  auto input_tensor = tile_node->GetOpDesc()->MutableInputDesc(0);
  auto output_tensor = tile_node->GetOpDesc()->MutableOutputDesc(0);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    return false;
  }
  std::vector<int64_t> input_shape = input_tensor->GetShape().GetDims();
  std::vector<int64_t> output_shape = output_tensor->GetShape().GetDims();
  if (input_shape.size() != output_shape.size()) {
    FE_LOGW("Node [%s, %s] has an input size that does not match the output size and cannot be supported.",
            tile_node->GetNamePtr(), tile_node->GetTypePtr());
    return false;
  }
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (input_shape[i] != output_shape[i] && input_shape[i] != 0) {
      tile_broadcast_dims.emplace_back(output_shape[i] / input_shape[i]);
    } else {
      tile_broadcast_dims.emplace_back(1);
    }
  }
  return true;
}

bool ConcatTileFusionPass::CheckTileBroadcastDimsAndInputShape(const std::vector<int64_t> &input_shape,
                                                               const std::vector<int32_t> &tile_broadcast_dims) const {
  FE_LOGD("Tile node broadcast_dims is [%s], concat_dims is [%d].",
          fe::StringUtils::IntegerVecToString(tile_broadcast_dims).c_str(), concat_dim_);
  for (size_t i = 0; i < tile_broadcast_dims.size(); i++) {
    // tile_broad_dim 1: broadcast
    if (tile_broadcast_dims[i] != 1 && static_cast<int64_t>(i) == concat_dim_) {
      FE_LOGW("Tile node broadcast_dim is equal concat_dim, couldn't be supported.");
      return false;
    }
  }

  uint64_t acc = 1;
  for (auto input : input_shape) {
    FE_CHECK(CheckInt64MulOverflow(acc, input) == FAILED, FE_LOGW("Input shape is too large to be supported."),
             return false);
    acc *= input;
  }
  FE_LOGD("Tile node's input element is [%lu].", acc);
  return acc <= shape_limited_;
}

bool ConcatTileFusionPass::IsValidTileNode(const ge::NodePtr &tile_node, std::vector<int32_t> &tile_broadcast_dims) {
  if (UnknownShapeUtils::IsUnknownShapeOp(*tile_node->GetOpDesc())) {
    FE_LOGD("Node [%s, %s] dynamic is not supported.", tile_node->GetNamePtr(), tile_node->GetTypePtr());
    return false;
  }

  if (tile_node->GetType() == TILE) {
    FE_LOGW("Node [%s, %s] without attr multiples.", tile_node->GetNamePtr(), tile_node->GetTypePtr());
    if (!GetTileBroadcastDims(tile_node, tile_broadcast_dims)) {
      return false;
    }
  } else {
    FE_CHECK(!ge::AttrUtils::GetListInt(tile_node->GetOpDesc(), "multiples", tile_broadcast_dims),
             FE_LOGW("TileD node without multiples attr, couldn't be supported"), return false);
  }
  if (tile_node->GetOpDesc()->MutableInputDesc(0) == nullptr) {
    FE_LOGW("Tile node input[0] is null");
    return false;
  }
  auto input_shape = tile_node->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims();
  if (!CheckTileBroadcastDimsAndInputShape(input_shape, tile_broadcast_dims)) {
    FE_LOGW("Tile input shape or multiple is invalid.");
    return false;
  }
  return true;
}

void ConcatTileFusionPass::GetTileFusionNodes(const ge::NodePtr &tile_node,
                                              std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs) {
  std::vector<int32_t> tile_broadcast_dims;
  if (!IsValidTileNode(tile_node, tile_broadcast_dims)) {
    return;
  }

  if (concat_tile_inputs.count(tile_broadcast_dims) > 0) {
    concat_tile_inputs[tile_broadcast_dims].emplace_back(tile_node);
  } else {
    concat_tile_inputs[tile_broadcast_dims] = {tile_node};
  }
}

bool ConcatTileFusionPass::GetFusionNodes(const ge::InDataAnchorPtr &in_data_anchor,
                                          std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    FE_LOGW("The [%d]peer out anchor of node concat is null", in_data_anchor->GetIdx());
    return false;
  }
  ge::NodePtr peer_out_node = peer_out_anchor->GetOwnerNode();
  FE_CHECK(peer_out_node == nullptr, FE_LOGW("Concat node input[%d]'s peer node is null.", in_data_anchor->GetIdx()),
            return FAILED);
  std::string op_type = peer_out_node->GetOpDesc()->GetType();
  FE_LOGD("Selected OpType is [%s, %s].", peer_out_node->GetOpDesc()->GetNamePtr(), op_type.c_str());
  if (ConcatOp.count(op_type) > 0) {
    FE_LOGW("Couldn't support concat node's input is concat");
    return false;
  }
  if (op_type == TILE || op_type == TILED) {
    GetTileFusionNodes(peer_out_node, concat_tile_inputs);
  }
  return true;
}

void ConcatTileFusionPass::GetSortedTiledNodes(std::vector<std::vector<ge::NodePtr>> &sorted_nodes,
                                               const std::vector<pair<int32_t, ge::NodePtr>> &tmp_nodes) const {
  if (tmp_nodes.size() < 1) {
    FE_LOGW("No support fusion tile nodes");
    return;
  }
  int32_t start_indx = tmp_nodes[0].first;
  std::vector<ge::NodePtr> fusion_tile_nodes = {tmp_nodes[0].second};
  for (size_t i = 1; i < tmp_nodes.size(); i++) {
    if (tmp_nodes[i].first == start_indx + 1) {
      fusion_tile_nodes.emplace_back(tmp_nodes[i].second);
    } else {
      if (fusion_tile_nodes.size() > MinTileNum) {
        sorted_nodes.emplace_back(fusion_tile_nodes);
      }
      fusion_tile_nodes.clear();
      fusion_tile_nodes.shrink_to_fit();
      fusion_tile_nodes.emplace_back(tmp_nodes[i].second);
    }
    start_indx = tmp_nodes[i].first;
  }
  if (fusion_tile_nodes.size() > MinTileNum) {
    sorted_nodes.emplace_back(fusion_tile_nodes);
  }
}

void ConcatTileFusionPass::SortContinuityTileNodes(std::vector<ge::NodePtr> &tile_nodes,
                                                   std::vector<std::vector<ge::NodePtr>> &sorted_nodes) const {
  std::vector<std::pair<int32_t, ge::NodePtr>> tmp_nodes;
  for (auto tile_node : tile_nodes) {
    FE_CHECK(tile_node->GetOutDataAnchor(0) == nullptr, REPORT_FE_ERROR("tile_node->GetOutDataAnchor(0) is nullptr."), return);
    auto peer_in_anchors = tile_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
    if (peer_in_anchors.size() > 1) {
      FE_LOGW("Node[%s, %s] with multi outputs.", tile_node->GetNamePtr(), tile_node->GetTypePtr());
      continue;
    }
    int32_t indx = peer_in_anchors.at(0)->GetIdx();
    std::pair<int32_t, ge::NodePtr> p_ind_tile(indx, tile_node);
    tmp_nodes.emplace_back(p_ind_tile);
  }
  std::sort(tmp_nodes.begin(), tmp_nodes.end(), CompareNodeIndx);
  GetSortedTiledNodes(sorted_nodes, tmp_nodes);
}

bool ConcatTileFusionPass::IsNeedToFusion(std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs,
                                          std::vector<std::vector<ge::NodePtr>> &sorted_nodes) const {
  auto it = concat_tile_inputs.begin();
  while (it != concat_tile_inputs.end()) {
    if (it->second.size() <= MinTileNum) {
      FE_LOGD("Tile node num is less than 2 don't need to fusion.");
      it = concat_tile_inputs.erase(it);
    } else {
      SortContinuityTileNodes(it->second, sorted_nodes);
      ++it;
    }
  }
  return sorted_nodes.size() > 0;
}

bool ConcatTileFusionPass::GetFusionNodeMap(const ge::NodePtr &concat_node,
                                    std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs) {
  for (auto data_anchor : concat_node->GetAllInDataAnchors()) {
    if (!GetFusionNodes(data_anchor, concat_tile_inputs)) {
      return false;
    }
  }
  return true;
}

Status ConcatTileFusionPass::AddConcatEdges(const ge::NodePtr &concat_node, vector<ge::NodePtr> &input_nodes) const {
  for (size_t i = 0; i < input_nodes.size(); i++) {
    FE_CHECK(ge::GraphUtils::AddEdge(input_nodes[i]->GetOutDataAnchor(0), concat_node->GetInDataAnchor(i)) ==
             ge::FAILED, FE_LOGE("ConcatTileFusionPass failed to add edge from input [%s] to new_concat_node [%s]",
             input_nodes[i]->GetNamePtr(), concat_node->GetNamePtr()), return FAILED);
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::RemoveInputEdges(vector<ge::NodePtr> &src_nodes, vector<ge::NodePtr> &dst_nodes) const {
  for (size_t i = 0; i < dst_nodes.size(); i++) {
    FE_CHECK(dst_nodes[i]->GetInDataAnchor(0) == nullptr, FE_LOGW("Tile node's input is null."), return FAILED);
    auto tile_peer_out_anchor = dst_nodes[i]->GetInDataAnchor(0)->GetPeerOutAnchor();
    if (tile_peer_out_anchor != nullptr) {
      FE_CHECK(tile_peer_out_anchor->GetOwnerNode() == nullptr, FE_LOGE("Tile input's node is null."), return FAILED);
      FE_CHECK(ge::GraphUtils::RemoveEdge(tile_peer_out_anchor, dst_nodes[i]->GetInDataAnchor(0)) == ge::FAILED,
               FE_LOGE("ConcatTileFusionPass failed to remove edge from input [%s] to fusion_node [%s]",
                       tile_peer_out_anchor->GetOwnerNode()->GetNamePtr(), dst_nodes[i]->GetNamePtr()),
               return FAILED);
      src_nodes.emplace_back(tile_peer_out_anchor->GetOwnerNode());
    } else {
      FE_LOGW("Node[%s, %s] peer output anchor is null", dst_nodes[i]->GetNamePtr(), dst_nodes[i]->GetTypePtr());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::RemoveConcatEdges(const ge::NodePtr &concat_node, vector<ge::NodePtr> &input_nodes) const {
  for (size_t i = 0; i < input_nodes.size(); i++) {
    FE_CHECK(input_nodes[i]->GetOutDataAnchor(0) == nullptr, FE_LOGW("Tile node's output is null."), return FAILED);
    auto peer_in_anchors = input_nodes[i]->GetOutDataAnchor(0)->GetPeerInDataAnchors();
    // can't support tile multi output
    if (peer_in_anchors.size() > 1) {
      FE_LOGW("ConcatTileFusionPass can't support tile multi output.");
      return FAILED;
    }
    int32_t indx = peer_in_anchors.at(0)->GetIdx();
    FE_CHECK(ge::GraphUtils::RemoveEdge(input_nodes[i]->GetOutDataAnchor(0), concat_node->GetInDataAnchor(indx)) ==
                 ge::FAILED,
             FE_LOGE("ConcatTileFusionPass failed to remove edge input [%s] to concat_node [%s]",
                     input_nodes[i]->GetNamePtr(), concat_node->GetNamePtr()),
             return FAILED);
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::UpdateConcatTileTensor(ge::OpDescPtr &new_concat_op_desc, ge::NodePtr &tile_node,
                                                    const int64_t &concat_shape, const int64_t &concat_ori_shape,
                                                    ge::GeTensorDesc &tile_input_output_tensor) const {
  std::vector<int64_t> data_shape = tile_input_output_tensor.GetShape().GetDims();
  std::vector<int64_t> ori_data_shape = tile_input_output_tensor.GetOriginShape().GetDims();
  data_shape[concat_dim_] = concat_shape;
  ori_data_shape[concat_dim_] = concat_ori_shape;
  tile_input_output_tensor.SetShape(ge::GeShape(data_shape));
  tile_input_output_tensor.SetOriginShape(ge::GeShape(ori_data_shape));
  // new_concat_op output need use inputs concat_dim
  new_concat_op_desc->AddOutputDesc(tile_input_output_tensor);
  // concat anchor_in
  ge::GeTensorDescPtr tile_output_tensor = tile_node->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK(tile_output_tensor == nullptr, REPORT_FE_ERROR("tile_output_tensor is nullptr."), return FAILED);
  std::vector<int64_t> tile_shape = tile_output_tensor->GetShape().GetDims();
  std::vector<int64_t> tile_ori_shape = tile_output_tensor->GetOriginShape().GetDims();
  tile_shape[concat_dim_] = concat_shape;
  tile_ori_shape[concat_dim_] = concat_ori_shape;

  tile_output_tensor->SetShape(ge::GeShape(tile_shape));
  tile_output_tensor->SetOriginShape(ge::GeShape(tile_ori_shape));
  // update tile input output
  tile_node->GetOpDesc()->UpdateInputDesc(0, tile_input_output_tensor);
  tile_node->GetOpDesc()->UpdateOutputDesc(0, *tile_output_tensor);
  return SUCCESS;
}

Status ConcatTileFusionPass::GetConcatValueByConcatdim(const vector<ge::NodePtr> &inputs, int64_t &concat_shape_dim,
                                                       int64_t &concat_ori_shape_dim,
                                                       ge::OpDescPtr &new_concat_op_desc) const {
  for (size_t i = 0; i < inputs.size(); i++) {
    ge::GeTensorDescPtr tile_input_output_tensor = inputs[i]->GetOpDesc()->MutableOutputDesc(0);
    FE_CHECK(tile_input_output_tensor == nullptr, FE_LOGW("Tile input tensor is null."), return FAILED);
    std::vector<int64_t> data_shape = tile_input_output_tensor->GetShape().GetDims();
    std::vector<int64_t> ori_data_shape = tile_input_output_tensor->GetOriginShape().GetDims();
    FE_CHECK((data_shape.size() < 1 || ori_data_shape.size() < 1),FE_LOGE("Tile inputs tensor is not complete."),
             return FAILED);
    size_t shape_len = data_shape.size();
    if (static_cast<int32_t>(shape_len) <= concat_dim_) {
      FE_LOGW("The data_shape dim[%d] is larger than data_shape size[%lu], couldn't support", concat_dim_, shape_len);
      return FAILED;
    }
    concat_shape_dim += data_shape[concat_dim_];
    concat_ori_shape_dim += ori_data_shape[concat_dim_];
    // new concat node input is inputs output
    FE_CHECK(new_concat_op_desc->AddInputDesc("x" + to_string(i), *tile_input_output_tensor) != ge::GRAPH_SUCCESS,
             FE_LOGW("Node [%s, %s] failed to add input descriptor", new_concat_op_desc->GetNamePtr(),
             new_concat_op_desc->GetTypePtr()), return FAILED);
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::InsertConcatOp(ge::OpDescPtr &new_concat_op_desc, const vector<ge::NodePtr> &inputs,
                                            ge::NodePtr &tile_node) {
  // Create new concat node 
  int64_t concat_shape_dim = 0;
  int64_t concat_ori_shape_dim = 0;
  if (new_concat_op_desc->GetType() == CONCAT || new_concat_op_desc->GetType() == CONCATD) {
    FE_CHECK(AddConcatDim(new_concat_op_desc) != SUCCESS, FE_LOGE("Failed to add concat_dim."), return FAILED);
  }
  FE_CHECK(GetConcatValueByConcatdim(inputs, concat_shape_dim, concat_ori_shape_dim, new_concat_op_desc) != SUCCESS,
           FE_LOGE("New concat node couldn't get concat value"), return FAILED);
  
  FE_LOGD("New concat_tile op[%s, %s] with inputs is [%zu].",
          new_concat_op_desc->GetNamePtr(), new_concat_op_desc->GetTypePtr(), inputs.size());
  ge::AttrUtils::SetInt(new_concat_op_desc, "N", static_cast<int64_t>(inputs.size()));
  // add concat dim for new concat_node
  if (new_concat_op_desc->GetType() == CONCATV2 || new_concat_op_desc->GetType() == CONCATV2D) {
    FE_CHECK(AddConcatDim(new_concat_op_desc) != SUCCESS, FE_LOGE("Failed to add concat_dim."), return FAILED);
  }
  ge::GeTensorDesc tile_input_output_tensor = inputs[0]->GetOpDesc()->GetOutputDesc(0);
  FE_CHECK(UpdateConcatTileTensor(new_concat_op_desc, tile_node, concat_shape_dim,
                                  concat_ori_shape_dim, tile_input_output_tensor) != SUCCESS,
                                  FE_LOGE("Update tile tensor Failed"), return FAILED);
  return SUCCESS;
}

Status ConcatTileFusionPass::RemoveTileEdge(vector<ge::NodePtr> &src_nodes, vector<ge::NodePtr> &dst_nodes,
                                            const ge::NodePtr &concat_node) const {
  FE_CHECK(RemoveInputEdges(src_nodes, dst_nodes) != SUCCESS,
          REPORT_FE_ERROR("Failed to remove input->tile edges"), return FAILED);
  FE_CHECK(RemoveConcatEdges(concat_node, dst_nodes) != SUCCESS,
           REPORT_FE_ERROR("Failed to remove edges between tile and concat_node"), return FAILED);                                              
  return SUCCESS;
}

Status ConcatTileFusionPass::ConcatAddConcatDimsInfo(const ge::NodePtr &new_concat_node,
                                                     std::vector<ge::NodePtr> &tile_inputs) {
  // Addedge concat_dim ->new_tile_node
  if (concat_dim_input_node_ != nullptr) {
    FE_LOGD("Adding new concat node to include concat dimension.");
    if (new_concat_node->GetType() == CONCAT) {
      tile_inputs.insert(tile_inputs.begin(), concat_dim_input_node_);
    } else {
      tile_inputs.emplace_back(concat_dim_input_node_);
    }
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::AddContrEdge(ge::NodePtr &tile_node, ge::NodePtr &new_concat_node) {
  for (auto contrl_node : tile_node->GetInControlNodes()) {
    FE_LOGD("Begin adding an edge between node [%s, %s] and control node [%s, %s].",
            tile_node->GetNamePtr(), tile_node->GetTypePtr(), contrl_node->GetNamePtr(), contrl_node->GetTypePtr());
    FE_CHECK(ge::GraphUtils::RemoveEdge(contrl_node->GetOutControlAnchor(), tile_node->GetInControlAnchor()), 
            REPORT_FE_ERROR("Remove contrl edges inputs[%s] to new_concat_node[%s] failed",
            contrl_node->GetNamePtr(), tile_node->GetNamePtr()), return FAILED);
    FE_CHECK(ge::GraphUtils::AddEdge(contrl_node->GetOutControlAnchor(), new_concat_node->GetInControlAnchor()), 
            REPORT_FE_ERROR("Add control edges from inputs[%s] to new_concat_node[%s] failed",
            contrl_node->GetNamePtr(), new_concat_node->GetNamePtr()), return FAILED);
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::DoAddTileEdges(ge::NodePtr &tile_node, ge::NodePtr &new_concat_node,
                                            std::vector<ge::NodePtr> &tile_inputs) {
  FE_CHECK(ConcatAddConcatDimsInfo(new_concat_node, tile_inputs) != SUCCESS,
            REPORT_FE_ERROR("New_concat_node[%s] failed to add concat dimension", new_concat_node->GetNamePtr()), return FAILED);
  FE_CHECK(AddConcatEdges(new_concat_node, tile_inputs), 
            REPORT_FE_ERROR("Failed to add edges inputs to new_concat_node"), return FAILED);
  FE_CHECK(AddContrEdge(tile_node, new_concat_node) != SUCCESS,
           REPORT_FE_ERROR("Failed to add input control edge to new_concat_node"), return FAILED);
  // Add new_concat ->tilenode -> concat
  FE_CHECK(ge::GraphUtils::AddEdge(new_concat_node->GetOutDataAnchor(0), tile_node->GetInDataAnchor(0)) ==
           ge::FAILED, FE_LOGE("ConcatTileFusionPass failed to add edge from input [%s] to tile_node [%s]",
           new_concat_node->GetNamePtr(), tile_node->GetNamePtr()), return FAILED);
  return SUCCESS;
}

Status ConcatTileFusionPass::DoTileFusion(const ge::NodePtr &concat_node,
                            ge::ComputeGraph &graph, const ge::OpDescPtr &replace_concat_node,
                            const std::vector<std::vector<ge::NodePtr>> &sorted_nodes,
                            std::vector<std::pair<int32_t, ge::NodePtr>> &concat_node_peer_output_nodes) {
  (void)replace_concat_node;
  FE_LOGD("Start to do tile fusion.");
  uint32_t new_concat_num = 0;
  for (auto sorted_tile_node :sorted_nodes) {
    std::vector<ge::NodePtr> tile_inputs;
    ge::NodePtr tile_node = sorted_tile_node[0];
    FE_CHECK(tile_node->GetOutDataAnchor(0) == nullptr, REPORT_FE_ERROR("tile_node->GetOutDataAnchor(0) is nullptr."), return FAILED);
    auto peer_in_anchors = tile_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
    int32_t indx = peer_in_anchors.at(0)->GetIdx();

    FE_CHECK(RemoveTileEdge(tile_inputs, sorted_tile_node, concat_node) !=SUCCESS,
            REPORT_FE_ERROR("remove edge from tile node failed"), return FAILED);    
    // remove rudant tile node
    for (size_t i = 1; i < sorted_tile_node.size(); i++) {
      graph.RemoveNode(sorted_tile_node[i]);
    }
    std::string node_name = concat_node->GetName();
    node_name = node_name + "_tile_" + std::to_string(new_concat_num);
    ge::OpDescPtr new_concat_tile= nullptr;
    FE_MAKE_SHARED(new_concat_tile = std::make_shared<ge::OpDesc>(node_name, concat_node->GetType()), return FAILED);
    new_concat_tile->CopyAttrsFrom(*concat_node->GetOpDesc());
    FE_CHECK(InsertConcatOp(new_concat_tile, tile_inputs, tile_node) != SUCCESS,
             FE_LOGE("Fusion add concat node failed"), return FAILED);

    ge::NodePtr new_concat_node = graph.AddNode(new_concat_tile);
    FE_CHECK_NOTNULL(new_concat_node);

    FE_CHECK(DoAddTileEdges(tile_node, new_concat_node, tile_inputs) != SUCCESS, FE_LOGE("Fusion edges failed"),
             return FAILED);
    FE_LOGD("After fusion Tile Node[%s, %s] with output shape is [%s].", tile_node->GetNamePtr(),
            tile_node->GetTypePtr(), tile_node->GetOpDesc()->GetOutputDesc(0).GetShape().ToString().c_str());
    // save tile node for replace_concat_node
    std::pair<int32_t, ge::NodePtr> tmp_peer_out_info(indx, tile_node);
    concat_node_peer_output_nodes.emplace_back(tmp_peer_out_info);
    new_concat_num++;
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::AddConcatDim(const ge::OpDescPtr &replace_concat_node) {
  if (concat_dim_input_node_ == nullptr) {
    FE_CHECK(!ge::AttrUtils::SetInt(replace_concat_node, "concat_dim", concat_dim_),
             REPORT_FE_ERROR("Node [%s, %s] failed to add concat_dim attribute", replace_concat_node->GetNamePtr(),
                             replace_concat_node->GetTypePtr()), return FAILED);
  } else {
    FE_LOGD("Concat node not find concat dim attr, replace concat node need to include constant node.");
    auto constant_output_tensor = concat_dim_input_node_->GetOpDesc()->MutableOutputDesc(0);
    FE_CHECK(constant_output_tensor == nullptr, REPORT_FE_ERROR("constant_output_tensor is nullptr."), return FAILED);
    FE_CHECK(replace_concat_node->AddInputDesc("concat_dim", *constant_output_tensor) == ge::GRAPH_FAILED,
             REPORT_FE_ERROR("Node [%s, %s] failed to add concat_dim node", replace_concat_node->GetNamePtr(),
                              replace_concat_node->GetTypePtr()), return FAILED);
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::GetConcatOriginInputInfo(
    const ge::NodePtr &concat_node,
    std::vector<std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>>> &concat_node_peer_output_info)
    const {
  auto data_in_anchors = concat_node->GetAllInDataAnchors();
  for (auto data_in_anchor : data_in_anchors) {
    if (data_in_anchor == nullptr) {
      continue;
    }
    auto peer_out_anchor = data_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      FE_CHECK(ge::GraphUtils::RemoveEdge(peer_out_anchor, data_in_anchor) == ge::FAILED,
               FE_LOGE("ConcatTileFusionPass failed to remove edge input [%s] to concat node [%s]",
                       peer_out_anchor->GetOwnerNode()->GetNamePtr(), concat_node->GetNamePtr()),
               return FAILED);
      FE_LOGD("Set concat_node inputs info.");
      ge::NodePtr peer_out_node = peer_out_anchor->GetOwnerNode();
      if (peer_out_node == nullptr) {
        FE_LOGE("Concat peer out node is null");
        continue;
      }
      if (peer_out_node == concat_dim_input_node_) {
        continue;
      }
      int32_t indx = data_in_anchor->GetIdx();
      std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>> tmp_info;
      tmp_info.first = indx;
      tmp_info.second.first = peer_out_anchor;
      tmp_info.second.second = peer_out_node->GetOpDesc()->MutableOutputDesc(peer_out_anchor->GetIdx());
      concat_node_peer_output_info.emplace_back(tmp_info);
    }
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::ReplaceConcatNodeAddInputs(const ge::NodePtr &concat_node,
    ge::OpDescPtr &replace_concat_node, std::vector<std::pair<int32_t, ge::NodePtr>> &concat_node_peer_output_nodes,
    std::vector<ge::OutDataAnchorPtr> &concat_node_peer_output_anchors) {
  FE_LOGD("After fusion, the number of concat_peer_out_node is [%zu].", concat_node_peer_output_nodes.size());
  std::vector<std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>>> concat_node_peer_output_info;
  for (auto concat_peer_out_node : concat_node_peer_output_nodes) {
    std::pair<int32_t, std::pair<ge::OutDataAnchorPtr, ge::GeTensorDescPtr>> tmp_info;
    tmp_info.first = concat_peer_out_node.first;
    tmp_info.second.first = concat_peer_out_node.second->GetOutDataAnchor(0);
    tmp_info.second.second = concat_peer_out_node.second->GetOpDesc()->MutableOutputDesc(0);
    concat_node_peer_output_info.emplace_back(tmp_info);
  }

  FE_CHECK(GetConcatOriginInputInfo(concat_node, concat_node_peer_output_info) != SUCCESS,
           FE_LOGE("Get concat node[%s]'s inputs failed", concat_node->GetNamePtr()), return FAILED);

  std::sort(concat_node_peer_output_info.begin(), concat_node_peer_output_info.end(), CompareNodeAnchorIndx);
  int32_t indx = 0;
  for (auto peer_out_node_info : concat_node_peer_output_info) {
    replace_concat_node->AddInputDesc("x" + to_string(indx), *peer_out_node_info.second.second);
    concat_node_peer_output_anchors.emplace_back(peer_out_node_info.second.first);
    indx++;
  }
  FE_LOGD("Replace concat node[%s] with inputs is [%zu].", replace_concat_node->GetNamePtr(),
          concat_node_peer_output_info.size());
  ge::AttrUtils::SetInt(replace_concat_node, "N", static_cast<int64_t>(concat_node_peer_output_info.size()));
  return SUCCESS;
}

Status ConcatTileFusionPass::ReplaceConcatNodeAddOutputs(
    ge::OpDescPtr &replace_concat_node, const ge::NodePtr &concat_node, bool &is_only_op,
    std::vector<ge::OutDataAnchorPtr> &concat_node_peer_output_anchors,
    std::vector<ge::InDataAnchorPtr> &concat_node_peer_input_anchors) const {
  FE_LOGD("Replace concat_node add output.");
  auto data_output_anchor = concat_node->GetOutDataAnchor(0);
  FE_CHECK(data_output_anchor == nullptr, REPORT_FE_ERROR("data_output_anchor is nullptr."), return FAILED);
  auto concat_peer_in_anchors = data_output_anchor->GetPeerInDataAnchors();
  auto concat_output_tensor = concat_node->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK(concat_output_tensor == nullptr, REPORT_FE_ERROR("concat_output_tensor is nullptr."), return FAILED);
  replace_concat_node->AddOutputDesc(*concat_output_tensor);
  for (auto concat_peer_in_anchor : concat_peer_in_anchors) {
    if (concat_peer_in_anchor == nullptr) {
      FE_LOGW("Concat_peer_in_anchor is null");
      continue;
    }
    FE_CHECK(ge::GraphUtils::RemoveEdge(data_output_anchor, concat_peer_in_anchor) ==
             ge::FAILED, FE_LOGE("ConcatTileFusionPass failed to remove edge input from concat node"),
             return FAILED);        
    concat_node_peer_input_anchors.emplace_back(concat_peer_in_anchor);
  }
  int64_t concat_input_num = 0;
  (void)ge::AttrUtils::GetInt(replace_concat_node, "N", concat_input_num);
  FE_LOGD("The current input_size is[%lld], anchors is [%zu].", concat_input_num,
          concat_node_peer_output_anchors.size());
  if (concat_input_num == 1) {
    FE_LOGD("Starting to fuse node->output.");
    int32_t idx = 0;
    if (static_cast<size_t>(concat_input_num) < concat_node_peer_output_anchors.size()) {
      idx = 1;
    }
    for (size_t i = 0; i < concat_node_peer_input_anchors.size(); i++) {
      FE_CHECK(ge::GraphUtils::AddEdge(concat_node_peer_output_anchors[idx], concat_node_peer_input_anchors[i]) ==
              ge::FAILED, FE_LOGE("ConcatTileFusionPass failed to add edge input to replace_concat_node"),
              return FAILED);
    }
    is_only_op = true;
    return SUCCESS;
  }
  FE_LOGD("Finished replacing concat node and added output.");
  return SUCCESS;
}

Status ConcatTileFusionPass::RelinkReplaceConcatNode(
    const ge::NodePtr &replace_concat_node, std::vector<ge::OutDataAnchorPtr> &concat_node_peer_output_anchors,
    std::vector<ge::InDataAnchorPtr> &concat_node_peer_input_anchors) const {
  auto data_in_anchors = replace_concat_node->GetAllInDataAnchors();
  if (data_in_anchors.size() != concat_node_peer_output_anchors.size()) {
    FE_LOGW("Replace concat node in anchor size is[%d], not equal concat_node peer out anchor's [%d]",
            data_in_anchors.size(), concat_node_peer_output_anchors.size());
    return FAILED;
  }
  // Add edge concat node peer output anchor to replace concat node
  for (size_t i = 0; i < data_in_anchors.size(); i++) {
    FE_CHECK(ge::GraphUtils::AddEdge(concat_node_peer_output_anchors[i], data_in_anchors.at(i)) ==
             ge::GRAPH_FAILED, FE_LOGE("ConcatTileFusionPass failed to add edge input to replace_concat_node"),
             return FAILED);
  }
  // Add edge concat node  to peer in anchor
  for (size_t i = 0; i < concat_node_peer_input_anchors.size(); i++) {
    FE_CHECK(ge::GraphUtils::AddEdge(replace_concat_node->GetOutDataAnchor(0), concat_node_peer_input_anchors[i]) ==
             ge::FAILED, FE_LOGE("ConcatTileFusionPass failed to add edge input to replace_concat_node"),
             return FAILED);
  }
  return SUCCESS;
}

bool ConcatTileFusionPass:: GetConcatDim(const ge::NodePtr &concat_node, const int32_t &indx) {
  auto input_tensor = concat_node->GetOpDesc()->MutableInputDesc(indx);
  auto output_tensor = concat_node->GetOpDesc()->MutableOutputDesc(0);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    FE_LOGW("Concat input[%d] is null, could not be supported.", indx);
    return false;
  }
  std::vector<int64_t> input_shape = input_tensor->GetShape().GetDims();
  std::vector<int64_t> output_shape = output_tensor->GetShape().GetDims();
  if (input_shape.size() != output_shape.size()) {
    FE_LOGW("Node [%s, %s] has an input size that does not match the output size and cannot be supported.",
            concat_node->GetNamePtr(), concat_node->GetTypePtr());
    return false;
  }
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (input_shape[i] != output_shape[i]) {
      concat_dim_ = static_cast<int32_t>(i);
    }
  }
  return true;
}

// for concat node, concat_dim is first const node
bool ConcatTileFusionPass::GetConcatConstantNode(const ge::NodePtr &concat_node, int32_t &indx) {
  FE_LOGD("Starting to get constant node.");
  auto in_anchor = concat_node->GetInDataAnchor(0);
  FE_CHECK(in_anchor == nullptr, FE_LOGW("Concat node first input anchor is None."), return false);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode() != nullptr &&
     (peer_out_anchor->GetOwnerNode()->GetType() == "Constant" ||
      peer_out_anchor->GetOwnerNode()->GetType() == "Const")) {
    concat_dim_input_node_ = peer_out_anchor->GetOwnerNode();
  }

  for (auto data_in_anchor : concat_node->GetAllInDataAnchors()) {
    FE_CHECK(data_in_anchor == nullptr, FE_LOGW("Concat node input exists null anchor."), return false);
    auto peer_out_anchor = data_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode() != nullptr &&
        (peer_out_anchor->GetOwnerNode()->GetType() == "Constant" ||
         peer_out_anchor->GetOwnerNode()->GetType() == "Const")) {
      continue;
    } else {
      auto input_tensor = concat_node->GetOpDesc()->MutableInputDesc(data_in_anchor->GetIdx());
      if (input_tensor != nullptr && input_tensor->GetShape().GetDimNum() != 0) {
        indx = data_in_anchor->GetIdx();
      }
    }
  }
  return concat_dim_input_node_ != nullptr;
}

// for concatv2 node, concat_dim is last const node
bool ConcatTileFusionPass::GetConcatV2ConstantNode(const ge::NodePtr &concat_node, int32_t &indx) {
  FE_LOGD("Starting to get constant node.");
  for (auto data_in_anchor : concat_node->GetAllInDataAnchors()) {
    FE_CHECK(data_in_anchor == nullptr, FE_LOGW("Concat node input exists null anchor."), return false);
    auto peer_out_anchor = data_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode() != nullptr &&
        (peer_out_anchor->GetOwnerNode()->GetType() == "Constant" ||
         peer_out_anchor->GetOwnerNode()->GetType() == "Const")) {
      concat_dim_input_node_ = peer_out_anchor->GetOwnerNode();
    } else {
      auto input_tensor = concat_node->GetOpDesc()->MutableInputDesc(data_in_anchor->GetIdx());
      if (input_tensor != nullptr && input_tensor->GetShape().GetDimNum() != 0) {
        indx = data_in_anchor->GetIdx();
      }
    }
  }
  return concat_dim_input_node_ != nullptr;
}

Status ConcatTileFusionPass::ParseConcatNode(const ge::NodePtr &concat_node) {
  FE_CHECK(CheckControlEdge(concat_node), FE_LOGW("Concat node with control edge is not supported."),
           return NOT_CHANGED);
  FE_CHECK(UnknownShapeUtils::IsUnknownShapeOp(*concat_node->GetOpDesc()),
           FE_LOGW("Not Support dynamic"), return NOT_CHANGED);
  ge::GeTensorDescPtr concat_output_tensor = concat_node->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK(concat_output_tensor == nullptr, REPORT_FE_ERROR("concat_output_tensor is nullptr."), return NOT_CHANGED);
  ge::DataType concat_data_type = concat_output_tensor->GetDataType();
  FE_CHECK(GetShapeLimited(concat_data_type) != SUCCESS, FE_LOGW("Can't get shape_limited."), return NOT_CHANGED);
  if (concat_node->GetType() == CONCAT || concat_node->GetType() == CONCATV2) {
    int32_t indx = 0;
    FE_LOGD("Node[%s, %s] without concat_dim attr.", concat_node->GetNamePtr(), concat_node->GetTypePtr());
    if (concat_node->GetType() == CONCAT) {
      FE_CHECK(!GetConcatConstantNode(concat_node, indx), FE_LOGW("Can't find const node"), return NOT_CHANGED);
    } else {
      FE_CHECK(!GetConcatV2ConstantNode(concat_node, indx), FE_LOGW("Can't find const node"), return NOT_CHANGED);
    }
    FE_CHECK(CheckControlEdge(concat_dim_input_node_),
            FE_LOGW("Concat node's concat_dim with conctrl edges, Couldn't be supported."), return NOT_CHANGED);
    if (!GetConcatDim(concat_node, indx)) {
      return NOT_CHANGED;
    }
    FE_LOGD("Concat node's concat dim is [%d].", concat_dim_);
  } else {
    ge::AttrUtils::GetInt(concat_node->GetOpDesc(), "concat_dim", concat_dim_);
    FE_CHECK(concat_node->GetOpDesc()->MutableOutputDesc(0) == nullptr, REPORT_FE_ERROR("concat_node->GetOpDesc()->MutableOutputDesc(0) is nullptr."),
           return NOT_CHANGED);
    int32_t shape_len =static_cast<int32_t>(concat_node->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDimNum());
    concat_dim_  = concat_dim_ < 0 ? shape_len + concat_dim_ : concat_dim_;
    FE_CHECK(concat_dim_ < 0, FE_LOGW("Concat attr concat_dim is invalid"), return NOT_CHANGED);
    FE_CHECK(concat_dim_ >= shape_len, FE_LOGW("Concat attr concat_dim[%d] is invalid", concat_dim_),
             return NOT_CHANGED); 
  }
  return SUCCESS;
}

Status ConcatTileFusionPass::DoReplaceConcatFusion(ge::NodePtr &concat_node, ge::OpDescPtr &replace_concat,
                                        ge::ComputeGraph &graph,
                                        std::vector<std::pair<int32_t, ge::NodePtr>> &concat_node_peer_output_nodes,
                                        std::vector<ge::InDataAnchorPtr> &concat_node_peer_input_anchors) {
  FE_LOGD("Start to do replace concat node fusion.");
  bool is_only_op = false;
  std::vector<ge::OutDataAnchorPtr> concat_node_peer_output_anchors;
  if (concat_node->GetType() == CONCAT || concat_node->GetType() == CONCATD) {
    FE_CHECK(AddConcatDim(replace_concat) != SUCCESS, FE_LOGE("Failed to add concat_dim."), return FAILED);
    if (concat_dim_input_node_ != nullptr) {
      concat_node_peer_output_anchors.emplace_back(concat_dim_input_node_->GetOutDataAnchor(0));
    }
  }
  FE_CHECK(ReplaceConcatNodeAddInputs(concat_node, replace_concat, concat_node_peer_output_nodes,
           concat_node_peer_output_anchors) != SUCCESS,
           FE_LOGE("Replace concat node add input tensor failed"), return FAILED);
          
  FE_CHECK(ReplaceConcatNodeAddOutputs(replace_concat, concat_node, is_only_op,
           concat_node_peer_output_anchors, concat_node_peer_input_anchors) != SUCCESS,
           FE_LOGE("Replace concat node add outputs failed"), return FAILED);
  
  graph.RemoveNode(concat_node);
  if (is_only_op) {
    FE_LOGD("There is only one op, don't to relink replace_concat_node.");
    return SUCCESS;
  }

  if (concat_node->GetType() == CONCATV2 || concat_node->GetType() == CONCATV2D) {
    FE_CHECK(AddConcatDim(replace_concat) != SUCCESS, FE_LOGE("Failed to add concat_dim."), return FAILED);
    if (concat_dim_input_node_ != nullptr) {
      concat_node_peer_output_anchors.emplace_back(concat_dim_input_node_->GetOutDataAnchor(0));
    }
  }
  
  ge::NodePtr replace_concat_node = graph.AddNode(replace_concat);
  FE_CHECK(replace_concat_node == nullptr, FE_LOGE("Create replace concat node failed."), return FAILED);
  FE_CHECK(RelinkReplaceConcatNode(replace_concat_node, concat_node_peer_output_anchors,
           concat_node_peer_input_anchors) != SUCCESS, FE_LOGE("Relink replace concat node failed"), return FAILED);
  return SUCCESS;
}

Status ConcatTileFusionPass::DoFusion(ge::NodePtr &concat_node, ge::ComputeGraph &graph,
                                      std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> &concat_tile_inputs) {
  std::vector<std::vector<ge::NodePtr>> sorted_nodes;
  FE_CHECK(!IsNeedToFusion(concat_tile_inputs, sorted_nodes), FE_LOGW("No nodes need to be fused."), return NOT_CHANGED);
  // creat replace concat node
  ge::OpDescPtr replace_concat = nullptr;
  std::string node_name = concat_node->GetName() + "_replace";
  FE_MAKE_SHARED(replace_concat = std::make_shared<ge::OpDesc>(node_name, concat_node->GetType()), return FAILED);
  replace_concat->CopyAttrsFrom(*concat_node->GetOpDesc());
  std::vector<std::pair<int32_t, ge::NodePtr>> concat_node_peer_output_nodes;
  std::vector<ge::InDataAnchorPtr> concat_node_peer_input_anchors;
  FE_CHECK(DoTileFusion(concat_node, graph, replace_concat, sorted_nodes, concat_node_peer_output_nodes) != SUCCESS,
          REPORT_FE_ERROR("Tile Fusion Failed"), return FAILED);
  FE_CHECK(DoReplaceConcatFusion(concat_node, replace_concat, graph, concat_node_peer_output_nodes,
          concat_node_peer_input_anchors) != SUCCESS,
          REPORT_FE_ERROR("Do replace concat fusion failed"), return FAILED);
  return SUCCESS;
}

Status ConcatTileFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, 
                                    vector<ge::NodePtr> &fusion_nodes) {
  (void)fusion_nodes;
  FE_LOGD("Start to ConcatTileFusionPass.");
  ge::NodePtr concat_node = GetNodeFromMapping(kPatternConcat, mapping);
  FE_CHECK(concat_node == nullptr, FE_LOGW("[GraphOpt][ConcatTileFusion] Concat node is nullptr."),
           return NOT_CHANGED);
  FE_LOGD("The concat node [%s, %s].", concat_node->GetNamePtr(), concat_node->GetTypePtr());
  FE_CHECK(ParseConcatNode(concat_node) != SUCCESS, FE_LOGW("ConcatTileFusionPass current concatNode is not supported."),
         return NOT_CHANGED);
  std::map<std::vector<int32_t>, std::vector<ge::NodePtr>> concat_tile_inputs;
  FE_CHECK(!GetFusionNodeMap(concat_node, concat_tile_inputs), 
           FE_LOGW("[GraphOpt][ConcatTile][Fus] Couldn't get fusion nodes."), return NOT_CHANGED);
  return DoFusion(concat_node, graph, concat_tile_inputs);
}

REG_PASS("ConcatTileFusionPass", BUILT_IN_GRAPH_PASS,
		 ConcatTileFusionPass, FE_PASS);
} // usenamepace fe