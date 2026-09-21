/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/fusion_graph_merge/fusion_graph_merge.h"

#include <memory>
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/fe_utils.h"
#include "common/fusion_op_comm.h"
#include "common/graph/fe_graph_utils.h"
#include "common/lxfusion_json_util.h"
#include "common/thread_slice_info_utils.h"
#include "common/op_info_common.h"
#include "common/util/constants.h"
#include "common/util/op_info_util.h"
#include "common/scope_allocator.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "common/sgt_slice_type.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "param_calculate/tensor_compute_util.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
namespace {
constexpr char kAttrRefPortIndex[] = "ref_port_index";
const std::string kMaxDumpGraphLevel = "3";
const std::string kStageMergeFusNode = "[SubGraphOpt][PostProcess][MergeFusNode]";

string RangeVecToString(const std::vector<ffts::DimRange> &integer_vec) {
  string temp;
  for (auto &ele : integer_vec) {
    temp += "{";
    temp += std::to_string(ele.lower);
    temp += ",";
    temp += std::to_string(ele.higher);
    temp += "}, ";
  }
  return temp;
}

void PrintTensorSliceInfo(const std::vector<std::vector<std::vector<ffts::DimRange>>> &tensor_slice) {
  for (size_t i = 0; i < tensor_slice.size(); i++) {
    FE_LOGD("Thread %zu:", i);
    for (size_t j = 0; j < tensor_slice[i].size(); j++) {
      FE_LOGD("input/output index : %zu", j);
      FE_LOGD("%s", RangeVecToString(tensor_slice[i][j]).c_str());
    }
  }
}

void PrintSliceInfo(const ffts::ThreadSliceMap &obj) {
  FE_LOGD("Print op slice info for op %s.", obj.original_node.c_str());
  FE_LOGD("%u %u %u %zu %zu %zu %zu %zu %zu %zu %zu", obj.thread_scope_id, obj.slice_instance_num,
          obj.thread_mode, obj.ori_input_tensor_shape.size(), obj.ori_output_tensor_shape.size(),
          obj.input_tensor_slice.size(), obj.output_tensor_slice.size(), obj.ori_input_tensor_slice.size(),
          obj.ori_output_tensor_slice.size(), obj.input_tensor_indexes.size(), obj.output_tensor_indexes.size());
  FE_LOGD("ori_input_tensor_shape:");
  for (size_t i = 0; i < obj.ori_input_tensor_shape.size(); i++) {
    FE_LOGD("Thread %zu:", i);
    for (size_t j = 0; j < obj.ori_input_tensor_shape[i].size(); j++) {
      FE_LOGD("ori input %zu:", j);
      FE_LOGD("%s", StringUtils::IntegerVecToString(obj.ori_input_tensor_shape[i][j]).c_str());
    }
  }
  FE_LOGD("END ori_input_tensor_shape.");

  FE_LOGD("ori_output_tensor_shape:");
  for (size_t i = 0; i < obj.ori_output_tensor_shape.size(); i++) {
    FE_LOGD("Thread %zu:", i);
    for (size_t j = 0; j < obj.ori_output_tensor_shape[i].size(); j++) {
      FE_LOGD("ori output %zu:", j);
      FE_LOGD("%s", StringUtils::IntegerVecToString(obj.ori_output_tensor_shape[i][j]).c_str());
    }
  }
  FE_LOGD("END ori_output_tensor_shape");

  FE_LOGD("input_tensor_slice:");
  PrintTensorSliceInfo(obj.input_tensor_slice);
  FE_LOGD("END input_tensor_slice.");

  FE_LOGD("output_tensor_slice:");
  PrintTensorSliceInfo(obj.output_tensor_slice);
  FE_LOGD("END output_tensor_slice.");

  FE_LOGD("ori_input_tensor_slice:");
  PrintTensorSliceInfo(obj.ori_input_tensor_slice);
  FE_LOGD("END ori_input_tensor_slice.");

  FE_LOGD("ori_output_tensor_slice:");
  PrintTensorSliceInfo(obj.ori_output_tensor_slice);
  FE_LOGD("END ori_output_tensor_slice.");
  FE_LOGD("input_tensor_indexes:");
  FE_LOGD("slice input index: %s", fe::StringUtils::IntegerVecToString(obj.input_tensor_indexes).c_str());
  FE_LOGD("End input_tensor_indexes.");
  FE_LOGD("output_tensor_indexes:");
  FE_LOGD("slice output index: %s.", fe::StringUtils::IntegerVecToString(obj.output_tensor_indexes).c_str());
  FE_LOGD("End output_tensor_indexes.");
}

void UpdateInputSgtSliceInfo(const ge::OpDescPtr &dst_op, uint32_t dst_anchor_index, ge::OpDescPtr &fus_op) {
  ffts::ThreadSliceMapPtr ori_slice_info = nullptr;
  ori_slice_info = dst_op->TryGetExtAttr(ffts::kAttrSgtStructInfo, ori_slice_info);

  ffts::ThreadSliceMapPtr fused_slice_info = nullptr;
  fused_slice_info = fus_op->TryGetExtAttr(ffts::kAttrSgtStructInfo, fused_slice_info);

  if (ori_slice_info != nullptr) {
    if (fused_slice_info == nullptr) {
      FE_LOGI("fused_slice_info for fused op %s is nullptr; creating a new one.", fus_op->GetName().c_str());
      FE_MAKE_SHARED(fused_slice_info = std::make_shared<ffts::ThreadSliceMap>(), return);
      if (fused_slice_info == nullptr) {
        REPORT_FE_ERROR("[SubGraphOpt][Merge][UpdateSliceInfo]Failed to make slice info pointer.");
        return;
      }
      FE_LOGD("Copying all slice info from [%s] to [%s].", dst_op->GetName().c_str(), fus_op->GetName().c_str());
      (void)ge::AttrUtils::SetInt(fus_op, "_thread_scope_id", ori_slice_info->thread_scope_id);
      (void)ge::AttrUtils::SetInt(fus_op, "_thread_mode", ori_slice_info->thread_mode);

      uint32_t thread_id = 0;
      if (ge::AttrUtils::GetInt(dst_op, "_thread_id", thread_id)) {
        (void)ge::AttrUtils::SetInt(fus_op, "_thread_id", thread_id);
        FE_LOGD("Set fusion op[%s] thread id from attr thread id %u.", fus_op->GetName().c_str(), thread_id);
      }

      if (!ori_slice_info->thread_mode) {
        (void)ge::AttrUtils::SetBool(fus_op, kTypeFFTSPlus, true);
      }
      CopyBasicInfo(*ori_slice_info.get(), fused_slice_info);
    }

    FE_LOGD("Copy input slice info from [%s] to [%s].", dst_op->GetName().c_str(), fus_op->GetName().c_str());
    CopyInputSliceInfo(*ori_slice_info.get(), dst_anchor_index, fused_slice_info);
    PrintSliceInfo(*fused_slice_info.get());
    fus_op->SetExtAttr(ffts::kAttrSgtStructInfo, fused_slice_info);
  }
}
} // namespace
FusionGraphMerge::FusionGraphMerge(const std::string &scope_attr, const GraphCommPtr &graph_comm_ptr)
    : scope_attr_(scope_attr), graph_comm_ptr_(graph_comm_ptr) {}

FusionGraphMerge::~FusionGraphMerge() {}

Status FusionGraphMerge::MergeFusionGraph(ge::ComputeGraph &fusion_graph) {
  FE_LOGD("Begin to merge graph[%s] after fusion.", fusion_graph.GetName().c_str());
  Status ret = MergeFusionNodes(fusion_graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to merge fusion graph.");
    return FAILED;
  }

  ret = MergeFusionNodeL2Info(fusion_graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to merge l2 info for fusion node.");
    return FAILED;
  }

  ret = AfterMergeFusionGraph(fusion_graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to handle fusion nodes after merging the graph.");
    return FAILED;
  }
  FE_LOGD("Finishing merging graph[%s] after fusion.", fusion_graph.GetName().c_str());
  return SUCCESS;
}

const std::string &FusionGraphMerge::GetScopeAttr() const {
  return scope_attr_;
}

Status FusionGraphMerge::MergeFusionNodes(ge::ComputeGraph &fusion_graph) {
  // 1. Get Same scope nodelist
  ScopeNodeMap fusion_scope_node_map;
  if (GetScopeNodeMap(fusion_graph, fusion_scope_node_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to obtain scope_node_map.");
    return FAILED;
  }

  // 2. merge same node
  for (auto it = fusion_scope_node_map.begin(); it != fusion_scope_node_map.end(); it++) {
    if (MergeEachFusionNode(fusion_graph, it->second)) {
      REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to merge fusion nodes.");
      return FAILED;
    }
  }
  CalcSingleOpStridedSize(fusion_graph);
  if (!fusion_scope_node_map.empty() && fusion_graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] MergeEachFusionNode Topology Failed.");
    return FAILED;
  }
  FE_LOGD("Finish to merge fusion nodes.");
  return SUCCESS;
}

Status FusionGraphMerge::MergeFusionNodeL2Info(const ge::ComputeGraph &fusion_graph) {
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (build_mode_value == ge::BUILD_MODE_TUNING ||
      (Configuration::Instance(AI_CORE_NAME).EnableL2Fusion() && CheckL2FusionFusionStrategy(fusion_graph))) {
    int64_t scope_id = 0;
    Status ret;
    for (auto &node_ptr : fusion_graph.GetDirectNode()) {
      ge::OpDescPtr op_desc = node_ptr->GetOpDesc();
      if (ge::AttrUtils::GetInt(op_desc, scope_attr_, scope_id) && scope_id >= 0) {
        ret = SetL2TaskInfoToFusionOp(node_ptr);
        if (ret != SUCCESS) {
          REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to update L2 info.");
          return FAILED;
        }
      } else {
        L2FusionInfoPtr origin_l2_info = GetL2FusionInfoFromJson(op_desc);
        if (origin_l2_info == nullptr) {
          FE_LOGD("node %s does not has l2_info.", node_ptr->GetName().c_str());
          continue;
        }
        ret = SetL2NameAndIndexForUnfusNode(origin_l2_info);
        if (ret != SUCCESS) {
          REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusGraph] Failed to update unfusion node information.");
          return FAILED;
        }
        SetL2FusionInfoToNode(op_desc, origin_l2_info);
      }
    }
  }
  return SUCCESS;
}

Status FusionGraphMerge::AddRelatedThreadNode(ScopeNodeMap &fusion_scope_map) const {
  for (const auto &ele : fusion_scope_map) {
    auto &fusion_nodes = ele.second;
    std::vector<std::shared_ptr<std::vector<ge::NodePtr>>> related_fusion_nodes;
    size_t has_related_node_cnt = 0;
    size_t related_nodes_size = 0;
    string pass_name;
    for (const ge::NodePtr &node : fusion_nodes) {
      FE_LOGD("Check node %s.", node->GetName().c_str());
      std::shared_ptr<std::vector<ge::NodePtr>> related_thread_nodes = nullptr;
      related_thread_nodes = node->GetOpDesc()->TryGetExtAttr(kAttrRelatedThreadsNodes, related_thread_nodes);
      if (related_thread_nodes != nullptr && !related_thread_nodes->empty()) {
        ++has_related_node_cnt;
        related_fusion_nodes.emplace_back(related_thread_nodes);
        if (related_nodes_size == 0) {
          related_nodes_size = related_thread_nodes->size();
        } else {
          if (related_nodes_size != related_thread_nodes->size()) {
            FE_LOGE("related node size of %s is %zu, first %zu.",
                    node->GetName().c_str(), related_thread_nodes->size(), related_nodes_size);
            return FAILED;
          }
        }
        for (auto &node_ptr : *related_thread_nodes) {
          FE_LOGD("Related nodes are %s.", node_ptr->GetName().c_str());
        }
      } else {
        FE_LOGI("Node %s do not have related nodes.", node->GetName().c_str());
      }
      if (pass_name.empty()) {
        (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kPassNameUbAttr, pass_name);
      }
    }
    if (has_related_node_cnt == 0) {
      continue;
    }

    if (has_related_node_cnt != fusion_nodes.size()) {
      FE_LOGE("Some nodes do not have related nodes.");
      return FAILED;
    }

    vector<int64_t> related_scope_id;
    for (size_t i = 0; i < related_nodes_size; ++i) {
      related_scope_id.emplace_back(ScopeAllocator::Instance().AllocateScopeId());
    }

    for (auto &related_nodes_of_one_original_node : related_fusion_nodes) {
      for (size_t i = 0; i < related_nodes_size; ++i) {
        auto &related_node = (*related_nodes_of_one_original_node)[i];
        if (ScopeAllocator::SetScopeAttr(related_node->GetOpDesc(), related_scope_id[i])) {
          FE_LOGD("Node[%s]: set related scope_id[%ld] successfully.", related_node->GetName().c_str(),
                  related_scope_id[i]);
        }

        auto iter = fusion_scope_map.find(related_scope_id[i]);
        if (iter == fusion_scope_map.end()) {
          std::vector<ge::NodePtr> node_vec_tmp({related_node});
          fusion_scope_map.emplace(related_scope_id[i], node_vec_tmp);
        } else {
          iter->second.emplace_back(related_node);
        }
        if (ge::AttrUtils::SetStr(related_node->GetOpDesc(), kPassNameUbAttr, pass_name)) {
          FE_LOGD("Node[%s]: set pass_name[%s] successfully.", related_node->GetName().c_str(), pass_name.c_str());
        }
      }
    }
  }
  return SUCCESS;
}

Status FusionGraphMerge::GetScopeNodeMap(const ge::ComputeGraph &fusion_graph, ScopeNodeMap &fusion_scope_map) const {
  int64_t scope_id = 0;
  for (const ge::NodePtr &node : fusion_graph.GetDirectNode()) {
    FE_CHECK_NOTNULL(node);
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);
    // LXfusion after slice, not process COMPIED_FUSION_OP
    bool no_need_compile = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_IS_COMPIED_FUSION_OP, no_need_compile);
    if (no_need_compile) {
      FE_LOGD("Op[name:%s, type:%s] does not require optimization of the fused graph.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (ge::AttrUtils::GetInt(op_desc, scope_attr_, scope_id)) {
      if (scope_id < 0) {
        continue;
      }
      auto iter = fusion_scope_map.find(scope_id);
      if (iter == fusion_scope_map.end()) {
        std::vector<ge::NodePtr> node_vec;
        node_vec.push_back(node);
        fusion_scope_map.emplace(scope_id, node_vec);
      } else {
        iter->second.push_back(node);
      }
    }
  }

  AddRelatedThreadNode(fusion_scope_map);
  return SUCCESS;
}

void FusionGraphMerge::SetAtomicFlagAndOutputIndex(const ge::NodePtr &first_node, const ge::NodePtr &fus_node) const {
  std::vector<int64_t> parameters_index;
  std::vector<int64_t> output_index;
  std::vector<int64_t> workspace_index;
  std::vector<int32_t> dtypes_list;
  std::vector<int64_t> value_int64_list;
  std::vector<float> value_float_list;
  int64_t workspace_atomic_flag = 0;
  FE_LOGI_IF(!ge::AttrUtils::GetListInt(first_node->GetOpDesc(), "ub_atomic_params", parameters_index),
             "Get attr ub_atomic_params not success!");
  FE_LOGI_IF(!ge::AttrUtils::GetListInt(first_node->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, dtypes_list),
             "Get attr ub_atomic_dtypes_list not success!");
  FE_LOGI_IF(!ge::AttrUtils::GetListInt(first_node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, value_int64_list),
             "Get attr ub_atomic_value_int64_list not success!");
  FE_LOGI_IF(!ge::AttrUtils::GetListFloat(first_node->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, value_float_list),
             "Get attr ub_atomic_value_float_list not success!");
  size_t input_num = fus_node->GetOpDesc()->GetInputsSize();
  size_t workspace_num = fus_node->GetOpDesc()->GetWorkspaceBytes().size();
  size_t output_num = fus_node->GetOpDesc()->GetOutputsSize();
  size_t first_input_num = first_node->GetOpDesc()->GetInputsSize();
  size_t first_workspace_num = first_node->GetOpDesc()->GetWorkspaceBytes().size();
  size_t first_output_num = first_node->GetOpDesc()->GetOutputsSize();
  FE_LOGD("inputNum:%zu, output_num:%zu, workspace_size:%zu, para_size:%zu,"
          "firstNode name:%s, fus_node name:%s.",
          input_num, output_num, workspace_num, parameters_index.size(), first_node->GetName().c_str(),
          fus_node->GetName().c_str());
  if ((first_input_num + first_workspace_num + first_output_num) < parameters_index.size()) {
    // in parameters data sort as input->output->workspace
    for (size_t i = 0; i < workspace_num; ++i) {
      size_t index = input_num + output_num + i;
      if (index >= parameters_index.size()) {
        break;
      }
      workspace_index.push_back(parameters_index[index]);
      if (parameters_index[index] == 1) {
        workspace_atomic_flag = 1;
      }
    }
    for (size_t i = 0; i < output_num; ++i) {
      size_t index = input_num + i;
      if (index >= parameters_index.size()) {
        break;
      }
      output_index.push_back(parameters_index[index]);
    }
    (void)ge::AttrUtils::SetInt(fus_node->GetOpDesc(), TBE_OP_ATOMIC_WORKSPACE_FLAG, workspace_atomic_flag);
    (void)ge::AttrUtils::SetListInt(fus_node->GetOpDesc(), TBE_OP_ATOMIC_WORKSPACE_INDEX, workspace_index);
    (void)ge::AttrUtils::SetListInt(fus_node->GetOpDesc(), TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);
    if (!dtypes_list.empty()) {
      (void)ge::AttrUtils::SetListInt(fus_node->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, dtypes_list);
    }
    if (!value_int64_list.empty()) {
      (void)ge::AttrUtils::SetListInt(fus_node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, value_int64_list);
    }
    if (!value_float_list.empty()) {
      (void)ge::AttrUtils::SetListFloat(fus_node->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, value_float_list);
    }
  }
}

void FusionGraphMerge::SetFusionOpOutputInplaceAbility(ge::OpDescPtr fus_op_desc,
                                                       const vector<vector<ge::ConstGeTensorDescPtr>> &output_place) {
  // map fusion_op tensor ptr to indx
  vector<vector<int64_t>> output_inplace_ability_list;
  for (auto place : output_place) {
    vector<int64_t> inplace = {fusion_op_output_idx_map_[place[0]], fusion_op_input_idx_map_[place[1]]};
    output_inplace_ability_list.push_back(inplace);
  }
  ge::AttrUtils::SetListListInt(fus_op_desc, kAttrOutputInplaceAbility, output_inplace_ability_list);
  PrintOutputInplace(fus_op_desc, output_inplace_ability_list);
  FE_LOGD("Finish Set fusion op OutputInplaceAbility.");
}

Status FusionGraphMerge::MergeEachFusionNode(ge::ComputeGraph &fusion_graph, vector<ge::NodePtr> &fus_nodelist) {
  // 1. Create fusion node
  ge::OpDescPtr fus_op_desc = FusionOpComm::CreateFusionOp(fus_nodelist, graph_comm_ptr_->GetEngineName());
  FE_CHECK_NOTNULL(fus_op_desc);
  std::unordered_set<ge::NodePtr> node_set;
  for (auto &node : fus_nodelist) {
    node_set.emplace(node);
  }
  // 2. Get Input Output EdgeList
  vector<FusionDataFlow> fus_input_edge_list;
  vector<FusionDataFlow> fus_output_edge_list;
  vector<FusionDataFlow> fus_input_ctrl_edge_list;
  vector<FusionDataFlow> fus_output_ctrl_edge_list;
  vector<pair<uint32_t, ge::NodePtr>> src_op_out_index_in_fus_op;
  if (graph_comm_ptr_->GetFusionNodeEdgeList(fus_nodelist, node_set, fus_input_edge_list,
      fus_output_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] Failed to get FusionNode edge list.");
    return FAILED;
  }
  if (graph_comm_ptr_->GetFusionNodeCtrlEdgeList(fus_nodelist, node_set, fus_input_ctrl_edge_list,
      fus_output_ctrl_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] Failed to get Fusion Node Control Edge List.");
    return FAILED;
  }
  if (CreateFusionOpNodeGraph(fus_input_edge_list, fus_output_edge_list, fus_nodelist, fus_op_desc, fusion_graph) !=
      SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] Failed to CreateFusionOpNodeGraph.");
    return FAILED;
  }
  if (AddFusionNodeOpDesc(fus_op_desc, fus_input_edge_list, fus_output_edge_list, src_op_out_index_in_fus_op)) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] Failed to AddFusionNodeOpDesc");
    return FAILED;
  }
  
  vector<vector<ge::ConstGeTensorDescPtr>> output_place;
  if (FusionOpComm::GetOutputInplaceAbilityAttrs(fus_nodelist, output_place)) {
    SetFusionOpOutputInplaceAbility(fus_op_desc, output_place);
  }
  std::vector<ge::NodePtr> fus_node_vec = fusion_graph.FuseNodeKeepTopo(fus_nodelist, {fus_op_desc});
  if (fus_node_vec.empty()) {
    return FAILED;
  }
  ge::NodePtr fus_node = fus_node_vec.at(0);
  FE_CHECK_NOTNULL(fus_node);
  GraphPassUtil::InheritGraphRelatedAttr(fus_nodelist, {fus_node}, BackWardInheritMode::kFusedNode);

  // Merge Same scope node
  if (graph_comm_ptr_->MergeFusionNodeEdgeList(fus_node, fus_nodelist, fus_input_edge_list, fus_output_edge_list) !=
      SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] Failed to execute MergeFusionNodeEdgeList!");
    return FAILED;
  }

  if (fus_nodelist.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] The fused node list is empty!");
    return FAILED;
  }
  ge::NodePtr first_node = fus_nodelist[0];
  SetAtomicFlagAndOutputIndex(first_node, fus_node);

  if (graph_comm_ptr_->MergeFusionNodeCtrlEdgeList(fus_node, fus_nodelist, fus_input_ctrl_edge_list,
                                                   fus_output_ctrl_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] MergeFusionNodeCtrlEdgeList failed!");
    return FAILED;
  }

  if (RefreshFusionNodeDataFlow(fus_node, fusion_graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MergeFusNode] Failed to RefreshFusionNodeDataFlow.");
    return FAILED;
  }

  if (CalcStridedWriteOutSize(fus_node, fus_nodelist, src_op_out_index_in_fus_op) != SUCCESS) {
    REPORT_FE_ERROR("%s Failed to calculate the real output tensor size of the fusion node with stride write node.",
                    kStageMergeFusNode.c_str());
    return FAILED;
  }

  if (CalcStridedReadInSize(fus_node, fus_nodelist) != SUCCESS) {
    REPORT_FE_ERROR("%s Failed to calculate the real input tensor size for the fusion node with stride read node.",
                    kStageMergeFusNode.c_str());
    return FAILED;
  }

  CreateOriginalFusionOpGraph(fus_node, fus_nodelist);

  SetDataDumpRef(fus_node, fusion_graph);
  for (ge::NodePtr &node : fus_nodelist) {
    if (fusion_graph.RemoveNode(node) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][RmNode] Failed to remove node %s.", node->GetName().c_str());
      return FAILED;
    }
  }

  // 4. Set id
  int64_t id = 0;
  for (const ge::NodePtr &node : fusion_graph.GetDirectNode()) {
    node->GetOpDesc()->SetId(id);
    id++;
  }
  return SUCCESS;
}

Status FusionGraphMerge::SetL2TaskInfoToFusionOp(ge::NodePtr fus_node) const {
  std::map<std::string, std::map<std::int64_t, ge::NodePtr>>::const_iterator iter;
  for (iter = fusion_op_name_map_all_.begin(); iter != fusion_op_name_map_all_.end(); iter++) {
    std::string origin_name = iter->first;
    for (auto iter_vec = iter->second.begin(); iter_vec != iter->second.end(); iter_vec++) {
      std::string map_fus_name = iter_vec->second->GetName();
      if (fus_node->GetName() == map_fus_name) {
        vector<ge::NodePtr> fus_nodelist;
        fus_nodelist = fus_node->GetOpDesc()->TryGetExtAttr("ScopeNodeSet", fus_nodelist);
        if (fus_nodelist.empty()) {
          REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetL2TaskInfo] Failed to TryGetExtAttr for ScopeNodeSet.");
          return FAILED;
        }
        L2FusionInfoPtr origin_l2_info;
        for (ge::NodePtr &origin_node : fus_nodelist) {
          if (origin_node->GetName() == origin_name) {
            ge::OpDescPtr origin_desc = origin_node->GetOpDesc();
            origin_l2_info = GetL2FusionInfoFromJson(origin_desc);
          }
        }
        if (origin_l2_info == nullptr) {
          FE_LOGD("node %s does not has l2_info.", origin_name.c_str());
          continue;
        }
        auto fuse_node_output_desc = fus_node->GetOpDesc()->MutableOutputDesc(iter_vec->first);
        FE_CHECK_NOTNULL(fuse_node_output_desc);
        std::int64_t origin_output_index = 0;
        if (!ge::AttrUtils::GetInt(fuse_node_output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,
                                   origin_output_index)) {
          FE_LOGW("Can not get origin_output_index, origin name %s, fus_name %s, fusIndex %ld.", origin_name.c_str(),
                  map_fus_name.c_str(), iter_vec->first);
        }

        bool is_fus_node_l2_info_set = false;
        ge::OpDescPtr fus_desc = fus_node->GetOpDesc();
        L2FusionInfoPtr fusion_l2_info = GetL2FusionInfoFromJson(fus_desc);
        if (fusion_l2_info != nullptr) {
          is_fus_node_l2_info_set = true;
        }
        if (!is_fus_node_l2_info_set) {
          L2FusionInfoPtr new_l2_info = nullptr;
          FE_MAKE_SHARED(new_l2_info = std::make_shared<TaskL2FusionInfo_t>(), return FAILED);
          FE_CHECK_NOTNULL(new_l2_info);
          // update nodename
          new_l2_info->node_name = fus_node->GetName();
          // update output and l2ctrl
          Status ret = UpdateL2Info(origin_output_index, iter_vec->first, origin_l2_info, new_l2_info);
          if (ret != SUCCESS) {
            REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetL2TaskInfo] Failed to update l2_info's output.");
            return FAILED;
          }

          // update l2ctrl
          new_l2_info->l2_info.l2ctrl = origin_l2_info->l2_info.l2ctrl;

          // update name and outputindex
          ret = SetL2NameAndIndex(origin_l2_info, new_l2_info);
          if (ret != SUCCESS) {
            REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetL2TaskInfo] Set L2NameAndIndex failed.");
            return FAILED;
          }
          SetL2FusionInfoToNode(fus_desc, new_l2_info);
        } else {
          Status ret = UpdateL2Info(origin_output_index, iter_vec->first, origin_l2_info, fusion_l2_info);
          if (ret != SUCCESS) {
            REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetL2TaskInfo] Failed to update l2_info's output.");
            return FAILED;
          }
          SetL2FusionInfoToNode(fus_desc, fusion_l2_info);
        }
        break;
      }
    }
  }
  return SUCCESS;
}

Status FusionGraphMerge::UpdateL2Info(const int64_t &origin_index, const int64_t &fusion_index,
                                      const L2FusionInfoPtr &originl2_info,
                                      const L2FusionInfoPtr &fusion_l2_info) const {
  // update output
  L2FusionDataMap_t &origin_outputs = originl2_info->output;
  L2FusionDataMap_t &fusion_outputs = fusion_l2_info->output;
  auto iter = origin_outputs.find(origin_index);
  if (iter != origin_outputs.end()) {
    L2FusionData_t &origin_output = iter->second;
    fusion_outputs[fusion_index] = origin_output;
  }
  return SUCCESS;
}

Status FusionGraphMerge::SetL2NameAndIndex(const L2FusionInfoPtr &originl2_info,
                                           L2FusionInfoPtr &fusion_l2_info) const {
  for (uint32_t i = 0; i < L2_MAXDATANUM; i++) {
    std::string orgin_name = originl2_info->l2_info.node_name[i];
    if (!orgin_name.empty()) {
      std::map<std::int64_t, std::int64_t> out_index_map;
      ge::NodePtr fusion_node;
      Status ret = GetFusionAnchorInfo(orgin_name, out_index_map, fusion_node);
      if (ret == FAILED || fusion_node == nullptr) {
        fusion_l2_info->l2_info.node_name[i] = originl2_info->l2_info.node_name[i];
        fusion_l2_info->l2_info.output_index[i] = originl2_info->l2_info.output_index[i];
        FE_LOGD("Node %s is not a fusion node.", orgin_name.c_str());
        continue;
      }
      std::string name_temp = fusion_node->GetName();
      std::map<std::int64_t, std::int64_t>::const_iterator index_map =
          out_index_map.find(originl2_info->l2_info.output_index[i]);
      uint8_t output_index_temp = 0;
      if (index_map != out_index_map.end()) {
        output_index_temp = index_map->second;
      } else {
        FE_LOGD(
            "Can not find fusion outindex in map, orgin node %s, index %u, "
            "fusion node %s.",
            orgin_name.c_str(), i, fusion_node->GetName().c_str());
      }
      fusion_l2_info->l2_info.node_name[i] = name_temp;
      fusion_l2_info->l2_info.output_index[i] = output_index_temp;
      FE_LOGD("data index %u, node_name %s, outputindex %u.", i, name_temp.c_str(), output_index_temp);
    } else {
      fusion_l2_info->l2_info.node_name[i] = originl2_info->l2_info.node_name[i];
      fusion_l2_info->l2_info.output_index[i] = originl2_info->l2_info.output_index[i];
    }
  }
  return SUCCESS;
}

Status FusionGraphMerge::CreateFusionOpNodeGraph(vector<FusionDataFlow> &fus_input_edge_list,
                                                 vector<FusionDataFlow> &fus_output_edge_list,
                                                 vector<ge::NodePtr> &fus_nodelist, ge::OpDescPtr fusion_op_desc,
                                                 ge::ComputeGraph &orig_graph) {
  ge::ComputeGraphPtr fusion_graph = nullptr;
  FE_MAKE_SHARED((fusion_graph = std::make_shared<ge::ComputeGraph>("OpFusionGraph")), return FAILED);
  FE_CHECK(fusion_graph == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOpNodeGraph] fusionGraph is nullptr."),
           return PARAM_INVALID);

  // copy nodes to new graph
  if (graph_comm_ptr_->CopyFusionOpNodes(fus_input_edge_list, fus_output_edge_list, fus_nodelist, fusion_op_desc,
                                         fusion_graph) != SUCCESS) {
    return FAILED;
  }

  // add edge to new graph
  if (graph_comm_ptr_->CopyFusionOpEdges(orig_graph, fusion_graph) != SUCCESS) {
    return FAILED;
  }

  fusion_graph->Dump();

  return SUCCESS;
}

Status FusionGraphMerge::ConvertSgtToJsonAttr(ge::OpDescPtr &fus_op) const {
  ffts::ThreadSliceMapPtr slice_info = nullptr;
  slice_info = fus_op->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info);
  if (slice_info == nullptr) {
    FE_LOGW("fusion op [%s:%s] does not have any sgt struct info.", fus_op->GetName().c_str(), fus_op->GetType().c_str());
    return FAILED;
  }
  std::string json_str;
  try {
    GetSliceJsonInfoStr(*(slice_info.get()), json_str);
  } catch (std::exception &e) {
    FE_LOGW("fusion op[%s:%s] caught exception %s", fus_op->GetName().c_str(), fus_op->GetType().c_str(), e.what());
    return FAILED;
  }
  FE_LOGD("fusion op[%s:%s] json str is %s.", fus_op->GetName().c_str(), fus_op->GetType().c_str(), json_str.c_str());
  if (!ge::AttrUtils::SetStr(fus_op, ffts::kAttrSgtJsonInfo, json_str)) {
    FE_LOGW("fusion op [%s:%s] failed to set sgt json", fus_op->GetName().c_str(), fus_op->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FusionGraphMerge::AddFusionNodeOpDesc(ge::OpDescPtr &fus_op, std::vector<FusionDataFlow> &fus_input_edge_list,
                                             std::vector<FusionDataFlow> &fus_output_edge_list,
                                             vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op) {
  if (AddFusionNodeInputDesc(fus_op, fus_input_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][AddFusNodeDesc] Failed to AddFusionNodeInputDesc.");
    return FAILED;
  }

  if (AddFusionNodeOutputDesc(fus_op, fus_output_edge_list, src_op_out_index_in_fus_op) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][AddFusNodeDesc] Failed to AddFusionNodeOutputDesc.");
    return FAILED;
  }

  (void)ConvertSgtToJsonAttr(fus_op);
  return SUCCESS;
}

static Status CopySgtCutList(const std::vector<std::vector<int64_t>> &ori_cur_list,
                      std::vector<std::vector<int64_t>> &dst_cur_list) {
  if (ori_cur_list.empty()) {
    return FAILED;
  }
  if (dst_cur_list.empty()) {
    dst_cur_list = ori_cur_list;
  } else if (ori_cur_list != dst_cur_list) {
    return FAILED;
  }
  return SUCCESS;
}

Status CopyOutputSliceInfo(const ffts::ThreadSliceMap &a, size_t output_index, const ffts::ThreadSliceMapPtr &b,
                           bool &is_consistent) {
  bool basic_condition = b->thread_scope_id == a.thread_scope_id && b->thread_mode == a.thread_mode &&
                         b->slice_instance_num == a.slice_instance_num;
  uint32_t slice_num = 1;  // manual
  if (b->thread_mode) {    // auto
    slice_num = b->slice_instance_num;
    // dynamic no need, will adjust in runtime
    if (a.output_tensor_slice.empty() || a.input_tensor_slice.empty()) {
      return SUCCESS;
    }
  }
  bool consistent = basic_condition && slice_num == b->ori_output_tensor_shape.size() &&
                    slice_num == b->output_tensor_slice.size() && slice_num == a.ori_output_tensor_shape.size() &&
                    b->output_tensor_slice.size() == a.output_tensor_slice.size() &&
                    b->ori_output_tensor_shape.size() == a.ori_output_tensor_shape.size();
  if (!consistent) {
    PrintSliceInfo(a);
    PrintSliceInfo(*b.get());
    FE_LOGW("Two sgt thread slices are inconsistent.");
    return FAILED;
  }
  is_consistent = true;

  for (uint32_t t_id = 0; t_id < slice_num; t_id++) {
    if (output_index >= a.output_tensor_slice[t_id].size() || output_index >= a.ori_output_tensor_shape[t_id].size()) {
      FE_LOGW("output_index %zu is larger than thread %u's size %zu.", output_index, t_id,
              a.output_tensor_slice[t_id].size());
      return FAILED;
    }
    b->output_tensor_slice[t_id].emplace_back(a.output_tensor_slice[t_id].at(output_index));
    b->ori_output_tensor_shape[t_id].emplace_back(a.ori_output_tensor_shape[t_id].at(output_index));
  }
  if (CopySgtCutList(a.output_cut_list, b->output_cut_list) != SUCCESS) {
    FE_LOGW("Original node %s output cut list is either empty or differs from other nodes.", a.original_node.c_str());
  }
  return SUCCESS;
}

void FusionGraphMerge::SetMultiKernelOutPutOffsets(const ge::OpDescPtr &src_op, size_t src_out_idx,
                                                   const ge::OpDescPtr &fus_op,
                                                   std::vector<int64_t> &save_pre_output_offset) const {
  std::vector<int64_t> output_offset_temp;
  std::vector<int64_t> buffer_fusion_output_offset;
  (void)ge::AttrUtils::GetListInt(fus_op, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION,
                                  buffer_fusion_output_offset);

  if (!buffer_fusion_output_offset.empty()) {
    if (ge::AttrUtils::GetListInt(src_op, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, output_offset_temp)) {
      buffer_fusion_output_offset.push_back(output_offset_temp[src_out_idx]);
    } else {
      buffer_fusion_output_offset.push_back(0);
    }
    (void)ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, buffer_fusion_output_offset);
  } else {
    if (ge::AttrUtils::GetListInt(src_op, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, output_offset_temp)) {
      save_pre_output_offset.push_back(output_offset_temp[src_out_idx]);
      (void)ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, save_pre_output_offset);
    } else {
      save_pre_output_offset.push_back(0);
    }
  }
}

void FusionGraphMerge::CopySameAtomic(const ffts::ThreadSliceMapPtr &ori_slice_info,
                                      const ffts::ThreadSliceMapPtr &fused_slice_info) const {
  if (ori_slice_info->same_atomic_clean_nodes.empty()) {
    return;
  }
  for (const auto &iter : ori_slice_info->same_atomic_clean_nodes) {
    if (std::find(fused_slice_info->same_atomic_clean_nodes.begin(), fused_slice_info->same_atomic_clean_nodes.end(),
        iter) == fused_slice_info->same_atomic_clean_nodes.end()) {
      fused_slice_info->same_atomic_clean_nodes.emplace_back(iter);
    }
  }
}

void FusionGraphMerge::UpdateOutputSgtSliceInfo(const ge::OpDescPtr &src_op, size_t src_out_idx, ge::OpDescPtr &fus_op,
                                                std::vector<int64_t> &save_pre_output_offset) const {
  SetMultiKernelOutPutOffsets(src_op, src_out_idx, fus_op, save_pre_output_offset);
  ffts::ThreadSliceMapPtr ori_slice_info = nullptr;
  ori_slice_info = src_op->TryGetExtAttr(ffts::kAttrSgtStructInfo, ori_slice_info);
  ffts::ThreadSliceMapPtr fused_slice_info = nullptr;
  fused_slice_info = fus_op->TryGetExtAttr(ffts::kAttrSgtStructInfo, fused_slice_info);
  if (ori_slice_info == nullptr || fused_slice_info == nullptr) {
    return;
  }

  CopySameAtomic(ori_slice_info, fused_slice_info);
  FE_LOGD("Copy output slice info from [%s] to [%s].", src_op->GetName().c_str(), fus_op->GetName().c_str());

  bool is_consistent = false;
  CopyOutputSliceInfo(*ori_slice_info.get(), src_out_idx, fused_slice_info, is_consistent);
  if (is_consistent) {
    FE_LOGD("is_consistent from [%s] to [%s].", src_op->GetName().c_str(), fus_op->GetName().c_str());
    std::vector<std::vector<int64_t>> src_output_tensor_shape;
    (void)ge::AttrUtils::GetListListInt(src_op, "ori_output_tensor_shape", src_output_tensor_shape);
    std::vector<std::vector<int64_t>> fuse_output_tensor_shape;
    (void)ge::AttrUtils::GetListListInt(fus_op, "ori_output_tensor_shape", fuse_output_tensor_shape);
    if (src_out_idx < src_output_tensor_shape.size()) {
      fuse_output_tensor_shape.emplace_back(src_output_tensor_shape.at(src_out_idx));
      (void)ge::AttrUtils::SetListListInt(fus_op, "ori_output_tensor_shape", fuse_output_tensor_shape);
    }
  }
  PrintSliceInfo(*fused_slice_info.get());
  fus_op->SetExtAttr(ffts::kAttrSgtStructInfo, fused_slice_info);
}

void FusionGraphMerge::UpdateFusionOpRefPortIndex(const ge::OpDescPtr &fus_op) const {
  bool is_reference = false;
  (void)ge::AttrUtils::GetBool(fus_op, ge::ATTR_NAME_REFERENCE, is_reference);
  if (!is_reference) {
    return;
  }

  std::vector<int32_t> ref_port_index;
  uint32_t cur_output_index = static_cast<uint32_t>(fus_op->GetAllOutputsDescSize() - 1);
  const auto &output_desc = fus_op->MutableOutputDesc(cur_output_index);
  if (ge::AttrUtils::GetListInt(output_desc, kAttrRefPortIndex, ref_port_index) && !ref_port_index.empty()) {
    for (size_t index = 0; index < ref_port_index.size(); ++index) {
      int32_t ref_tmp = ref_port_index[index];
      ref_port_index[index] = ref_tmp < 0 ? -ref_tmp : ref_tmp;
    }
    FE_LOGD("Op[%s] output[%u] ref_port_index[%s].", fus_op->GetName().c_str(), cur_output_index,
            StringUtils::IntegerVecToString(ref_port_index).c_str());
    ge::AttrUtils::SetListInt(output_desc, kAttrRefPortIndex, ref_port_index);
  }
}

Status FusionGraphMerge::AddFusionNodeOutputDesc(ge::OpDescPtr fus_op, vector<FusionDataFlow> &fus_output_edge_list,
                                                 vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op) {
  vector<int> out_mem_type_fus_node;
  vector<int> out_mem_type_old_node;
  vector<int64_t> FusNodeOutputOffset;
  vector<int64_t> old_node_output_offset;

  out_mem_type_fus_node.clear();
  out_mem_type_old_node.clear();
  graph_comm_ptr_->ClearFusionSrc();
  graph_comm_ptr_->ClearFusionDst();

  int32_t fusion_src_index = 0;
  std::vector<int64_t> save_pre_output_offset;
  for (FusionDataFlow &dataflow : fus_output_edge_list) {
    auto outedge = dataflow.edge;
    std::pair<string, ge::AnchorPtr> &node_srcindex_pair = dataflow.node_dataindex_pair;
    if (outedge.second != nullptr) {
      FE_CHECK_NOTNULL(outedge.second->GetOwnerNode());
      int64_t dst_op_id = outedge.second->GetOwnerNode()->GetOpDesc()->GetId();
      ge::DataAnchorPtr out_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.second);
      FE_CHECK_NOTNULL(out_edge_dst_data_anchor_ptr);
      if (graph_comm_ptr_->IsFusionDstExist(dst_op_id, outedge.second)) {
        FE_LOGI("MergeFusionNodeOutputEdgeList Dstid %ld, DstIndex %u.", dst_op_id,
                static_cast<uint32_t>(out_edge_dst_data_anchor_ptr->GetIdx()));
        continue;
      }
      graph_comm_ptr_->SaveFusionDst(dst_op_id, outedge.second);
    }

    int32_t fusion_src_exist_index;
    int32_t fusion_dst_exist_index;

    ge::DataAnchorPtr out_edge_src_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.first);
    FE_CHECK_NOTNULL(out_edge_src_data_anchor_ptr);
    FE_CHECK_NOTNULL(out_edge_src_data_anchor_ptr->GetOwnerNode());
    ge::OpDescPtr out_edge_src_op_desc_ptr = out_edge_src_data_anchor_ptr->GetOwnerNode()->GetOpDesc();
    FE_CHECK(out_edge_src_op_desc_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][AddTensor] outEdgeSrcOpDescPtr is nullptr."), return FAILED);
    int64_t src_op_id = out_edge_src_op_desc_ptr->GetId();
    if (!graph_comm_ptr_->GetFusionSrc(src_op_id, outedge.first, fusion_src_exist_index, fusion_dst_exist_index)) {
      size_t src_out_idx = static_cast<size_t>(out_edge_src_data_anchor_ptr->GetIdx());
      if (src_out_idx < out_edge_src_op_desc_ptr->GetOutputsSize()) {
        auto output_desc_ptr = out_edge_src_op_desc_ptr->GetOutputDescPtr(src_out_idx);
        if (output_desc_ptr == nullptr) {
          continue;
        }
        Status ret = fus_op->AddOutputDesc(*output_desc_ptr);
        if (ret != ge::GRAPH_SUCCESS) {
          FE_LOGI("fusOp[%s, %s] AddOutputDesc not successfully", fus_op->GetName().c_str(), fus_op->GetType().c_str());
        } else {
          src_op_out_index_in_fus_op.push_back(std::make_pair(src_out_idx,
                                                              out_edge_src_data_anchor_ptr->GetOwnerNode()));
          int64_t current_output_idx = static_cast<int64_t>(fus_op->GetAllOutputsDescSize() - 1);
          fusion_op_output_idx_map_[output_desc_ptr] = current_output_idx;
        }
        UpdateFusionOpRefPortIndex(fus_op);
        graph_comm_ptr_->AddFusionOutputSrc(src_op_id, outedge.first, fusion_src_index, node_srcindex_pair);
        FE_LOGD("fusOp[%s], src_out_op[%s].", fus_op->GetName().c_str(), out_edge_src_op_desc_ptr->GetName().c_str());
        UpdateOutputSgtSliceInfo(out_edge_src_op_desc_ptr, src_out_idx, fus_op, save_pre_output_offset);
        bool need_precompile_node = false;
        (void)ge::AttrUtils::GetBool(fus_op, NEED_RE_PRECOMPILE, need_precompile_node);
        if (need_precompile_node) {
          (void)ge::AttrUtils::GetListInt(out_edge_src_op_desc_ptr, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                          out_mem_type_old_node);
          if (static_cast<size_t>(out_edge_src_data_anchor_ptr->GetIdx()) < out_mem_type_old_node.size()) {
            out_mem_type_fus_node.push_back(out_mem_type_old_node[out_edge_src_data_anchor_ptr->GetIdx()]);
          } else {
            FE_LOGI("outMemTypeOldNode size is %zu, less than idx %d.", out_mem_type_old_node.size(),
                    out_edge_src_data_anchor_ptr->GetIdx());
          }

          old_node_output_offset = out_edge_src_op_desc_ptr->GetOutputOffset();
          string offset_str;
          for (auto old_node_input : old_node_output_offset) {
            offset_str += std::to_string(old_node_input) + ", ";
          }
          FE_LOGI("%s, old_node_output_offset size is %zu, idx %d, offset %s.",
                  out_edge_src_op_desc_ptr->GetName().c_str(), old_node_output_offset.size(),
                  out_edge_src_data_anchor_ptr->GetIdx(), offset_str.c_str());
          if (static_cast<size_t>(out_edge_src_data_anchor_ptr->GetIdx()) < old_node_output_offset.size()) {
            FusNodeOutputOffset.push_back(old_node_output_offset[out_edge_src_data_anchor_ptr->GetIdx()]);
          } else {
            FE_LOGI("%s, out_offset_old_node size is %zu, less than idx %d.",
                    out_edge_src_op_desc_ptr->GetName().c_str(),
                    old_node_output_offset.size(), out_edge_src_data_anchor_ptr->GetIdx());
          }
        }
      } else {
        int32_t output_desc_size = out_edge_src_op_desc_ptr->GetOutputsSize();
        int32_t SrcIndex = out_edge_src_data_anchor_ptr->GetIdx();
        REPORT_FE_ERROR("[SubGraphOpt][Merge][AddTensor] Failed to add output tensor with size %u, src index %u.",
                        static_cast<uint32_t>(output_desc_size), static_cast<uint32_t>(SrcIndex));
        return FAILED;
      }

      fusion_src_index++;
    }
  }

  FE_LOGD("fusion node in_mem_type size %zu, outputoffset size %zu.", out_mem_type_fus_node.size(),
          FusNodeOutputOffset.size());
  if (out_mem_type_fus_node.size()) {
    (void)ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_mem_type_fus_node);
  }

  if (FusNodeOutputOffset.size()) {
    fus_op->SetOutputOffset(FusNodeOutputOffset);
  }

  return SUCCESS;
}

static Status SetL1Attr(ge::OpDescPtr &fus_op, const vector<int64_t> &op_input_l1_flag,
                 const vector<int64_t> &op_input_l1_addr, const vector<int64_t> &op_input_l1_valid_size) {
  bool need_set_l1_attr = false;
  for (int64_t l1_flag : op_input_l1_flag) {
    if (l1_flag >= 0) {
      need_set_l1_attr = true;
      break;
    }
  }

  if (need_set_l1_attr) {
    if (!ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_OP_INPUT_L1_FLAG, op_input_l1_flag)) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][SetL1Attr] Failed to set the op_input_l1_flag attribute on op [%s, %s].",
                      fus_op->GetName().c_str(), fus_op->GetType().c_str());
      return FAILED;
    }
    FE_LOGD("OpInputL1Flag of op[%s, %s] is %s.", fus_op->GetName().c_str(), fus_op->GetType().c_str(),
            StringUtils::IntegerVecToString(op_input_l1_flag).c_str());
    if (!ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_OP_INPUT_L1_ADDR, op_input_l1_addr)) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][SetL1Attr] Failed to set op_input_l1_addr attribute for op[%s, %s].",
                      fus_op->GetName().c_str(), fus_op->GetType().c_str());
      return FAILED;
    }
    FE_LOGD("OpInputL1Addr of op[%s, %s] is %s.", fus_op->GetName().c_str(), fus_op->GetType().c_str(),
            StringUtils::IntegerVecToString(op_input_l1_addr).c_str());
    if (!ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_OP_INPUT_L1_VALID_SIZE, op_input_l1_valid_size)) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][SetL1Attr] Failed to set the op_input_l1_valid_size attribute on op [%s, %s].",
                      fus_op->GetName().c_str(), fus_op->GetType().c_str());
      return FAILED;
    }
    FE_LOGD("OpInputL1ValidSize of op[%s, %s] is %s.", fus_op->GetName().c_str(), fus_op->GetType().c_str(),
            StringUtils::IntegerVecToString(op_input_l1_valid_size).c_str());
  }
  return SUCCESS;
}
void CopyBasicInfo(const ffts::ThreadSliceMap &a, const ffts::ThreadSliceMapPtr &b) {
  b->thread_scope_id = a.thread_scope_id;
  b->is_first_node_in_topo_order = a.is_first_node_in_topo_order;
  b->thread_mode = a.thread_mode;
  b->node_num_in_thread_scope = a.node_num_in_thread_scope;
  b->is_input_node_of_thread_scope = a.is_input_node_of_thread_scope;
  b->is_output_node_of_thread_scope = a.is_output_node_of_thread_scope;
  b->original_node = a.original_node;
  b->slice_instance_num = a.slice_instance_num;
  b->parallel_window_size = a.parallel_window_size;
  b->thread_id = a.thread_id;
  b->core_num = a.core_num;
  b->cut_type = a.cut_type;
  b->dependencies = a.dependencies;
  uint32_t slice_num = 1;  // manual
  if (b->thread_mode) {    // auto
    slice_num = b->slice_instance_num;
    // dynamic no need, will adjust in runtime
    if (a.input_tensor_slice.empty() || a.output_tensor_slice.empty()) {
      return;
    }
  }
  for (size_t i = 0; i < slice_num; i++) {
    b->input_tensor_slice.emplace_back(vector<vector<ffts::DimRange>>());
    b->output_tensor_slice.emplace_back(vector<vector<ffts::DimRange>>());
    b->ori_input_tensor_shape.emplace_back(std::vector<std::vector<int64_t>>());
    b->ori_output_tensor_shape.emplace_back(std::vector<std::vector<int64_t>>());
  }
}

Status CopyInputSliceInfo(const ffts::ThreadSliceMap &a, size_t input_index, const ffts::ThreadSliceMapPtr &b) {
  uint32_t slice_num = 1;  // manual
  if (b->thread_mode) {    // auto
    slice_num = b->slice_instance_num;
    // dynamic no need, will adjust in runtime
    if (a.input_tensor_slice.empty() || a.output_tensor_slice.empty()) {
      return SUCCESS;
    }
  }
  bool basic_condition = b->thread_scope_id == a.thread_scope_id && b->thread_mode == a.thread_mode &&
                         b->slice_instance_num == a.slice_instance_num;
  bool consistent = basic_condition && slice_num == b->ori_input_tensor_shape.size() &&
                    slice_num == b->input_tensor_slice.size() && slice_num == a.ori_input_tensor_shape.size() &&
                    b->input_tensor_slice.size() == a.input_tensor_slice.size() &&
                    b->ori_input_tensor_shape.size() == a.ori_input_tensor_shape.size();
  if (!consistent) {
    PrintSliceInfo(a);
    PrintSliceInfo(*b.get());
    FE_LOGW("Two sgt thread slices are inconsistent.");
    return FAILED;
  }
  for (uint32_t t_id = 0; t_id < slice_num; t_id++) {
    if (input_index >= a.input_tensor_slice[t_id].size() || input_index >= a.ori_input_tensor_shape[t_id].size()) {
      REPORT_FE_ERROR("input_index %zu is larger than thread %u's size %zu.", input_index, t_id,
                      a.input_tensor_slice[t_id].size());
      return FAILED;
    }
    b->input_tensor_slice[t_id].emplace_back(a.input_tensor_slice[t_id].at(input_index));

    b->ori_input_tensor_shape[t_id].emplace_back(a.ori_input_tensor_shape[t_id].at(input_index));
  }
  if (CopySgtCutList(a.input_cut_list, b->input_cut_list) != SUCCESS) {
    FE_LOGW("Original node %s input cut list is either empty or differs from other nodes", a.original_node.c_str());
  }
  return SUCCESS;
}

static void SetInputMemTypeAndOffset(ge::OpDescPtr &fus_op, const vector<int> &in_mem_type_fus_node,
                              const vector<int64_t> &FusNodeInputOffset) {
  if (!in_mem_type_fus_node.empty()) {
    (void)ge::AttrUtils::SetListInt(fus_op, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_mem_type_fus_node);
  }

  if (!FusNodeInputOffset.empty()) {
    fus_op->SetInputOffset(FusNodeInputOffset);
  }
}

void FusionGraphMerge::UpdateOutputRefPortIndex(ge::OpDescPtr &in_edge_dst_op_desc_ptr, const ge::OpDescPtr &fus_op,
                                                const uint32_t dst_anchor_index) const {
  bool is_reference;
  (void)ge::AttrUtils::GetBool(fus_op, ge::ATTR_NAME_REFERENCE, is_reference);
  if (!is_reference) {
    return;
  }
  bool has_update_index = false;
  for (size_t i = 0; i < in_edge_dst_op_desc_ptr->GetAllOutputsDescSize(); i++) {
    std::vector<int32_t> ref_port_index;
    if (ge::AttrUtils::GetListInt(in_edge_dst_op_desc_ptr->MutableOutputDesc(i), kAttrRefPortIndex,
                                  ref_port_index) && !ref_port_index.empty()) {
      for (auto &ref_port_index_tmp : ref_port_index) {
        if (ref_port_index_tmp == static_cast<int32_t>(dst_anchor_index)) {
          FE_LOGD("Op[%s] old ref_port_index is: %d.", in_edge_dst_op_desc_ptr->GetName().c_str(), ref_port_index_tmp);
          // first set a minus number, to avoid update ref_port_index repeatly
          // In function UpdateFusionOpRefPortIndex, flush the minus ref_port_index to positive ref_port_index
          ref_port_index_tmp = -(fus_op->GetAllInputsSize() - 1);
          FE_LOGD("Op[%s] ref_port_index has been refresh, new ref_port_index is: %d.",
                  in_edge_dst_op_desc_ptr->GetName().c_str(), ref_port_index_tmp);
          has_update_index = true;
          break;
        }
      }
      if (has_update_index) {
        (void)ge::AttrUtils::SetListInt(in_edge_dst_op_desc_ptr->MutableOutputDesc(i), kAttrRefPortIndex,
                                        ref_port_index);
      }
    }
  }
}

static Status SetInputParaTypeList(ge::OpDescPtr &op_desc, std::vector<uint32_t> &fusion_input_para_type,
                            const uint32_t &idx) {
  std::vector<uint32_t> dst_input_type_list;
  if (!ge::AttrUtils::GetListInt(op_desc, kInputParaTypeList, dst_input_type_list)) {
    FE_LOGW("Failed to get attribute input_para_type_list from node [%s].", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (idx >= dst_input_type_list.size()) {
    REPORT_FE_ERROR("Invalid attr input_para_type_list from op[%s], wrong index[%u].",
                    op_desc->GetName().c_str(), idx);
    return FAILED;
  }
  fusion_input_para_type.emplace_back(dst_input_type_list[idx]);
  return SUCCESS;
}

Status FusionGraphMerge::AddFusionNodeInputDesc(ge::OpDescPtr fus_op, vector<FusionDataFlow> &fus_input_edge_list) {
  int32_t fusion_dst_index = -1;
  graph_comm_ptr_->ClearFusionSrc();
  vector<bool> fusion_is_input_const_vector;
  vector<int> in_mem_type_fus_node;
  vector<int> in_mem_type_old_node;
  vector<int64_t> FusNodeInputOffset;
  std::vector<uint32_t> fusion_input_para_type;

  in_mem_type_fus_node.clear();
  in_mem_type_old_node.clear();

  // update _op_infer_depends
  vector<string> op_infer_depends;
  (void)ge::AttrUtils::GetListStr(fus_op, ATTR_NAME_OP_INFER_DEPENDS, op_infer_depends);

  vector<int64_t> op_input_l1_flag(fus_input_edge_list.size(), -1);
  vector<int64_t> op_input_l1_addr(fus_input_edge_list.size(), 0);
  vector<int64_t> op_input_l1_valid_size(fus_input_edge_list.size(), -1);
  uint32_t tensor_desc_index = 0;

  for (FusionDataFlow &dataflow : fus_input_edge_list) {
    auto inedge = dataflow.edge;
    std::pair<string, ge::AnchorPtr> &node_dstindex_pair = dataflow.node_dataindex_pair;

    bool invalid_opt_flag = true;
    int64_t src_op_id = -1;
    if (inedge.first != nullptr) {
      auto owner_node_first = inedge.first->GetOwnerNode();
      FE_CHECK_NOTNULL(owner_node_first);
      src_op_id = owner_node_first->GetOpDesc()->GetId();
      invalid_opt_flag = false;
    }

    // only support data edge
    fusion_dst_index = fusion_dst_index + 1;
    vector<int64_t> input_vector;
    input_vector = fus_op->GetInputOffset();
    if (static_cast<size_t>(fusion_dst_index) < input_vector.size()) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][AddTensor] tensor index %d is less than input size %zu", fusion_dst_index,
                      input_vector.size());
      return FAILED;
    }

    ge::DataAnchorPtr in_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(inedge.second);
    FE_CHECK_NOTNULL(in_edge_dst_data_anchor_ptr);
    size_t in_anchor_idx = in_edge_dst_data_anchor_ptr->GetIdx();

    auto owner_node_second = in_edge_dst_data_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(owner_node_second);
    ge::OpDescPtr in_edge_dst_op_desc_ptr = owner_node_second->GetOpDesc();
    FE_CHECK_NOTNULL(in_edge_dst_op_desc_ptr);
    vector<bool> is_input_const_vector = in_edge_dst_op_desc_ptr->GetIsInputConst();
    uint32_t dst_anchor_index = static_cast<uint32_t>(in_anchor_idx);
    if (dst_anchor_index < in_edge_dst_op_desc_ptr->GetAllInputsSize()) {
      auto input_desc_ptr = in_edge_dst_op_desc_ptr->GetInputDescPtrDfault(in_anchor_idx);
      FE_CHECK_NOTNULL(input_desc_ptr);
      if (fus_op->AddInputDesc(*input_desc_ptr) != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][Merge][AddTensor] Add input desc failed.");
        return FAILED;
      }
      int64_t current_input_idx = static_cast<int64_t>(fus_op->GetAllInputsSize() - 1);
      fusion_op_input_idx_map_[input_desc_ptr] = current_input_idx;
      if (SetInputParaTypeList(in_edge_dst_op_desc_ptr, fusion_input_para_type, in_anchor_idx) != SUCCESS) {
        return FAILED;
      }
      if (invalid_opt_flag) {
        tensor_desc_index++;
        continue;
      }
      (void)UpdateOutputRefPortIndex(in_edge_dst_op_desc_ptr, fus_op, dst_anchor_index);
      FE_LOGD("fusOp name %s, in_anchor_idx %zu.", fus_op->GetName().c_str(), in_anchor_idx);
      for (size_t idx = 0; idx < op_infer_depends.size(); idx++) {
        if (op_infer_depends[idx] ==
            in_edge_dst_op_desc_ptr->GetInputNameByIndex(in_anchor_idx)) {
          op_infer_depends[idx] = fus_op->GetInputNameByIndex(tensor_desc_index);
          FE_LOGD("Update _op_infer_depends attribute of fusion op[%s]: %zu, %s", fus_op->GetName().c_str(), idx,
                  op_infer_depends[idx].c_str());
        }
      }
      AddBuffFusionNodeInputDesc(in_mem_type_old_node, in_edge_dst_op_desc_ptr, in_edge_dst_data_anchor_ptr,
                                 FusNodeInputOffset, in_mem_type_fus_node);
      bool need_precompile_node = false;
      bool buff_fusion_status =
          ((ge::AttrUtils::GetBool(fus_op, NEED_RE_PRECOMPILE, need_precompile_node) && need_precompile_node));
      if (buff_fusion_status) {
        UpdateL1Attr(in_edge_dst_op_desc_ptr, ge::ATTR_NAME_OP_INPUT_L1_FLAG, dst_anchor_index, tensor_desc_index,
                     op_input_l1_flag);
        UpdateL1Attr(in_edge_dst_op_desc_ptr, ge::ATTR_NAME_OP_INPUT_L1_ADDR, dst_anchor_index, tensor_desc_index,
                     op_input_l1_addr);
        UpdateL1Attr(in_edge_dst_op_desc_ptr, ge::ATTR_NAME_OP_INPUT_L1_VALID_SIZE, dst_anchor_index, tensor_desc_index,
                     op_input_l1_valid_size);
      }

      if (dst_anchor_index < is_input_const_vector.size()) {
        fusion_is_input_const_vector.push_back(is_input_const_vector[in_edge_dst_data_anchor_ptr->GetIdx()]);
      } else {
        fusion_is_input_const_vector.push_back(false);
      }
      UpdateInputSgtSliceInfo(in_edge_dst_op_desc_ptr, dst_anchor_index, fus_op);
      tensor_desc_index++;
    } else {
      int32_t input_desc_size = in_edge_dst_op_desc_ptr->GetAllInputsSize();
      int32_t is_input_const_size = is_input_const_vector.size();
      int32_t DstIndex = in_edge_dst_data_anchor_ptr->GetIdx();
      REPORT_FE_ERROR(
          "[SubGraphOpt][Merge][AddTensor] Failed to add input tensor with size %u, const_size %u, dst index %u.",
          static_cast<uint32_t>(input_desc_size), static_cast<uint32_t>(is_input_const_size),
          static_cast<uint32_t>(DstIndex));
      return FAILED;
    }
    fus_op->SetIsInputConst(fusion_is_input_const_vector);
    graph_comm_ptr_->AddFusionInputSrc(src_op_id, inedge.first, fusion_dst_index, node_dstindex_pair);
    (void)ge::AttrUtils::SetListInt(fus_op, kInputParaTypeList, fusion_input_para_type);
    FE_LOGD("Succeeded in setting input_para_type_list[len: %zu] for op [%s, %s].", fusion_input_para_type.size(),
            fus_op->GetName().c_str(), fus_op->GetType().c_str());
  }

  FE_LOGD("fusion node[%s] in_mem_type size %zu, inputoffset size %zu.", fus_op->GetName().c_str(),
          in_mem_type_fus_node.size(), FusNodeInputOffset.size());
  SetInputMemTypeAndOffset(fus_op, in_mem_type_fus_node, FusNodeInputOffset);
  Status ret = SetL1Attr(fus_op, op_input_l1_flag, op_input_l1_addr, op_input_l1_valid_size);
  if (ret != SUCCESS) {
    return ret;
  }
  (void)ge::AttrUtils::SetListStr(fus_op, ATTR_NAME_OP_INFER_DEPENDS, op_infer_depends);
  return SUCCESS;
}

void FusionGraphMerge::UpdateL1Attr(ge::OpDescPtr op_desc_ptr, const string &attr_key, const uint32_t &anchor_index,
                                    const uint32_t &tensor_desc_index, vector<int64_t> &target_vec) const {
  vector<int64_t> attr_value_vec;
  if (ge::AttrUtils::GetListInt(op_desc_ptr, attr_key, attr_value_vec)) {
    FE_LOGD("The [%s] of op[%s, %s] is %s.", attr_key.c_str(), op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), StringUtils::IntegerVecToString(attr_value_vec).c_str());
    FE_LOGD("The input index of fused op is %u. The input index of fusion op is %u", anchor_index, tensor_desc_index);
    if (anchor_index < attr_value_vec.size()) {
      target_vec[tensor_desc_index] = attr_value_vec[anchor_index];
    }
  } else {
    FE_LOGD("op[%s, %s] does not have attr [%s].", op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
            attr_key.c_str());
  }
}

void FusionGraphMerge::AddBuffFusionNodeInputDesc(vector<int> &in_mem_type_old_node,
                                                  ge::OpDescPtr &in_edge_dst_op_desc_ptr,
                                                  const ge::DataAnchorPtr &in_edge_dst_data_anchor_ptr,
                                                  vector<int64_t> &FusNodeInputOffset,
                                                  vector<int> &in_mem_type_fus_node) const {
  (void)ge::AttrUtils::GetListInt(in_edge_dst_op_desc_ptr, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_mem_type_old_node);
  if (static_cast<size_t>(in_edge_dst_data_anchor_ptr->GetIdx()) < in_mem_type_old_node.size()) {
    in_mem_type_fus_node.push_back(in_mem_type_old_node[in_edge_dst_data_anchor_ptr->GetIdx()]);
  } else {
    FE_LOGI("inMemTypeOldNode size is %zu, less than idx %d.", in_mem_type_old_node.size(),
            in_edge_dst_data_anchor_ptr->GetIdx());
  }

  auto old_node_input_offset = in_edge_dst_op_desc_ptr->GetInputOffset();
  string offset_str;
  for (auto old_node_input : old_node_input_offset) {
    offset_str += std::to_string(old_node_input) + ", ";
  }
  FE_LOGI("%s, old_node_input_offset size is %zu, idx %d, offset %s.", in_edge_dst_op_desc_ptr->GetName().c_str(),
          old_node_input_offset.size(), in_edge_dst_data_anchor_ptr->GetIdx(), offset_str.c_str());
  if (static_cast<size_t>(in_edge_dst_data_anchor_ptr->GetIdx()) < old_node_input_offset.size()) {
    FusNodeInputOffset.push_back(old_node_input_offset[in_edge_dst_data_anchor_ptr->GetIdx()]);
  } else {
    FE_LOGI("%s, inoffset_old_node size is %zu, less than idx %d.", in_edge_dst_op_desc_ptr->GetName().c_str(),
            FusNodeInputOffset.size(), in_edge_dst_data_anchor_ptr->GetIdx());
  }
  return;
}

Status FusionGraphMerge::SetDataOutPutMapingAttr(
    std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_anchors_map) const {
  string original_name;
  for (auto iter = fusion_op_anchors_map.cbegin(); iter != fusion_op_anchors_map.cend(); iter++) {
    for (auto iter_vec = iter->second.cbegin(); iter_vec != iter->second.cend(); iter_vec++) {
      FE_CHECK(iter_vec->first == nullptr, FE_LOGW("iterVec->first is null"), return FAILED);
      if (iter_vec->second == nullptr) {
        continue;
      }
      std::int64_t anchor_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->first)->GetIdx();
      std::int64_t anchor_fusion_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second)->GetIdx();
      ge::NodePtr fus_node = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second)->GetOwnerNode();
      FE_CHECK_NOTNULL(fus_node);
      FE_CHECK(iter->first == nullptr, FE_LOGW("iter->first is null"), return FAILED);
      FE_CHECK(iter->first->GetOpDesc() == nullptr, FE_LOGW("iter->first opdesc is null"), return FAILED);
      if (iter->first->GetOpDesc()->MutableOutputDesc(anchor_idx) == nullptr) {
        FE_LOGW("iter->first outputdesc is null");
        return FAILED;
      }
      auto output_desc = iter->first->GetOpDesc()->MutableOutputDesc(anchor_idx);
      FE_CHECK(fus_node->GetOpDesc() == nullptr, FE_LOGW("fusNode OpDesc is null"), return FAILED);
      if (fus_node->GetOpDesc()->MutableOutputDesc(anchor_fusion_idx) == nullptr) {
        FE_LOGW("fusNode OpDesc is null");
        return FAILED;
      }
      auto fuse_node_output_desc = fus_node->GetOpDesc()->MutableOutputDesc(anchor_fusion_idx);
      std::vector<std::string> original_names;
      bool is_has_attr =
          ge::AttrUtils::GetListStr(iter->first->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
      if (is_has_attr && original_names.size() == 0) {
        continue;
      }

      if (is_has_attr && ge::AttrUtils::GetStr(output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, original_name)) {
        (void)ge::AttrUtils::SetStr(fuse_node_output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, original_name);

        std::int64_t origin_output_index = 0;
        if (ge::AttrUtils::GetInt(output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index)) {
          (void)ge::AttrUtils::SetInt(fuse_node_output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,
                                      origin_output_index);
        }
        if (GraphPassUtil::GetDataDumpOriginDataType(output_desc) != ge::DT_UNDEFINED) {
          (void)GraphPassUtil::SetDataDumpOriginDataType(GraphPassUtil::GetDataDumpOriginDataType(output_desc),
                                                         fuse_node_output_desc);
        }

        if (GraphPassUtil::GetDataDumpOriginFormat(output_desc) != ge::FORMAT_RESERVED) {
          (void)GraphPassUtil::SetDataDumpOriginFormat(GraphPassUtil::GetDataDumpOriginFormat(output_desc),
                                                       fuse_node_output_desc);
        }

        FE_LOGD(
            "Original node %s has an original attribute; we successfully copied it to the fused node %s.",
            iter->first->GetName().c_str(), fus_node->GetName().c_str());
      } else {
        (void)ge::AttrUtils::SetStr(fuse_node_output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, iter->first->GetName());
        (void)ge::AttrUtils::SetInt(fuse_node_output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, anchor_idx);
        (void)GraphPassUtil::SetDataDumpOriginDataType(output_desc->GetOriginDataType(), fuse_node_output_desc);
        (void)GraphPassUtil::SetDataDumpOriginFormat(output_desc->GetOriginFormat(), fuse_node_output_desc);
        FE_LOGD(
            "original node %s does not has original attr, we create attr for "
            "fuse node %s.",
            iter->first->GetName().c_str(), fus_node->GetName().c_str());
      }

      (void)ge::AttrUtils::SetStr(fuse_node_output_desc, ge::ATTR_NAME_FUSION_ORIGIN_NAME, iter->first->GetName());
      (void)ge::AttrUtils::SetInt(fuse_node_output_desc, ge::ATTR_NAME_FUSION_ORIGIN_OUTPUT_INDEX, anchor_idx);
      FE_LOGD("Set output anchor map, original node:%s, anchor idx:%ld, fusion node:%s.",
              iter->first->GetName().c_str(), anchor_idx, fus_node->GetName().c_str());
    }
  }
  return SUCCESS;
}

void FusionGraphMerge::SetDataDumpRef(ge::NodePtr fus_node, const ge::ComputeGraph &fusion_graph) const {
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  bool switch_on = Configuration::Instance(graph_comm_ptr_->GetEngineName()).EnableL1Fusion() ||
                   build_mode_value == ge::BUILD_MODE_TUNING ||
                   (Configuration::Instance(AI_CORE_NAME).EnableL2Fusion() &&
                    CheckL2FusionFusionStrategy(fusion_graph));
  if (!switch_on) {
    return;
  }

  SetDataDumpRefForOutputDataAnchors(fus_node);
  SetDataDumpRefForInDataAnchors(fus_node);
}

void FusionGraphMerge::SetDataDumpRefForOutputDataAnchors(ge::NodePtr fus_node) const {
  for (auto out_data_anchor : fus_node->GetAllOutDataAnchors()) {
    if (out_data_anchor == nullptr) {
      continue;
    }

    for (auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      if (next_in_data_anchor == nullptr) {
        continue;
      }
      ge::NodePtr next_node = next_in_data_anchor->GetOwnerNode();
      int next_anchor_idx = next_in_data_anchor->GetIdx();
      int fusion_node_anchor_idx = out_data_anchor->GetIdx();
      auto next_input_desc = next_node->GetOpDesc()->MutableInputDesc(next_anchor_idx);
      if (next_input_desc == nullptr) {
        continue;
      }
      std::string tmp_node_name_index;
      if (!ge::AttrUtils::GetStr(next_input_desc, fe::ATTR_DATA_DUMP_REF, tmp_node_name_index)) {
        FE_LOGD("nextNode:%s input_index:%d, ATTR_DATA_DUMP_REF is null.", next_node->GetName().c_str(),
                next_anchor_idx);
        continue;
      }

      std::string ref_index = std::to_string(fusion_node_anchor_idx);
      std::string node_name_index = fus_node->GetName() + ":output:" + ref_index;
      FE_LOGD("nextNode:%s input_index:%d, ATTR_DATA_DUMP_REF is %s", next_node->GetName().c_str(), next_anchor_idx,
              node_name_index.c_str());
      (void)ge::AttrUtils::SetStr(next_input_desc, fe::ATTR_DATA_DUMP_REF, node_name_index);
    }
  }
}

void FusionGraphMerge::SetDataDumpRefForInDataAnchors(ge::NodePtr fus_node) const {
  for (auto &in_data_anchor : fus_node->GetAllInDataAnchors()) {
    if (in_data_anchor == nullptr || in_data_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }

    auto pre_node_out_anchor = in_data_anchor->GetPeerOutAnchor();
    ge::NodePtr pre_node = pre_node_out_anchor->GetOwnerNode();
    if (pre_node->GetType() != OP_TYPE_PHONY_CONCAT) {
      int pre_anchor_idx = pre_node_out_anchor->GetIdx();
      int fusion_node_anchor_idx = in_data_anchor->GetIdx();
      auto pre_output_desc = pre_node->GetOpDesc()->MutableOutputDesc(pre_anchor_idx);
      if (pre_output_desc == nullptr) {
        continue;
      }
      std::string tmp_node_name_index;
      if (!ge::AttrUtils::GetStr(pre_output_desc, fe::ATTR_DATA_DUMP_REF, tmp_node_name_index)) {
        FE_LOGD("preNode:%s output_index:%d, ATTR_DATA_DUMP_REF is null", pre_node->GetName().c_str(), pre_anchor_idx);
        continue;
      }
      std::string ref_index = std::to_string(fusion_node_anchor_idx);
      std::string node_name_index = fus_node->GetName() + ":input:" + ref_index;
      FE_LOGD("preNode:%s output_index:%d, ATTR_DATA_DUMP_REF is %s", pre_node->GetName().c_str(), pre_anchor_idx,
              node_name_index.c_str());
      (void)ge::AttrUtils::SetStr(pre_output_desc, fe::ATTR_DATA_DUMP_REF, node_name_index);
    }
  }
}

Status FusionGraphMerge::SetL2NameAndIndexForUnfusNode(L2FusionInfoPtr &originl2_info) const {
  for (uint32_t i = 0; i < L2_MAXDATANUM; i++) {
    std::string orgin_name = originl2_info->l2_info.node_name[i];
    if (!orgin_name.empty()) {
      std::map<std::int64_t, std::int64_t> out_index_map;
      ge::NodePtr fusion_node;
      Status ret = GetFusionAnchorInfo(orgin_name, out_index_map, fusion_node);
      if (ret == FAILED) {
        continue;
      }
      std::string name_temp = fusion_node->GetName();
      std::map<std::int64_t, std::int64_t>::const_iterator index_map =
          out_index_map.find(originl2_info->l2_info.output_index[i]);
      uint8_t output_index_temp = 0;
      if (index_map != out_index_map.end()) {
        output_index_temp = index_map->second;
      } else {
        FE_LOGD(
            "Can not find fusion outindex in map, orgin node %s, index %u, "
            "fusion node %s.",
            orgin_name.c_str(), i, fusion_node->GetName().c_str());
      }
      originl2_info->l2_info.node_name[i] = name_temp;
      originl2_info->l2_info.output_index[i] = output_index_temp;
      FE_LOGD("data index %u, node_name %s, outputindex %u.", i, name_temp.c_str(), output_index_temp);
    }
  }
  return SUCCESS;
}

Status FusionGraphMerge::GetFusionAnchorInfo(const std::string &origin_name,
                                             std::map<std::int64_t, std::int64_t> &out_index_map,
                                             ge::NodePtr &fusion_node) const {
  auto iter = fusion_op_name_map_all_.find(origin_name);
  if (iter != fusion_op_name_map_all_.end()) {
    for (auto iter_vec = iter->second.begin(); iter_vec != iter->second.end(); iter_vec++) {
      fusion_node = iter_vec->second;

      auto fuse_node_output_desc = fusion_node->GetOpDesc()->MutableOutputDesc(iter_vec->first);
      if (fuse_node_output_desc == nullptr) {
        continue;
      }
      std::int64_t origin_output_index = 0;
      if (!ge::AttrUtils::GetInt(fuse_node_output_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,
                                 origin_output_index)) {
        FE_LOGW(
            "Can not get origin_output_index, origin name %s, fus_name %s, "
            "fusIndex %ld.",
            origin_name.c_str(), fusion_node->GetName().c_str(), iter_vec->first);
        continue;
      }
      out_index_map.emplace(std::make_pair(origin_output_index, iter_vec->first));
      FE_LOGD("node %s, index %ld, fus_node %s, index %ld.", origin_name.c_str(), origin_output_index,
              fusion_node->GetName().c_str(), iter_vec->first);
    }
  } else {
    FE_LOGW("Can not find node %s in fusion_op_name_map_all.", origin_name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status FusionGraphMerge::RefreshFusionNodeDataFlow(ge::NodePtr fus_node, const ge::ComputeGraph &fusion_graph) {
  std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_anchors_map;
  ge::kFusionDataFlowVec_t fus_dataflow_list;

  // get in edgelist
  if (graph_comm_ptr_->GetNodeDataFlowMap(fus_node, fusion_op_anchors_map, fus_dataflow_list, 0) == FAILED) {
    return FAILED;
  }
  fus_node->SetFusionInputFlowList(fus_dataflow_list);
  fus_node->GetOpDesc()->SetExtAttr("InEdgeAnchorMap", fusion_op_anchors_map);

  fusion_op_anchors_map.clear();
  fus_dataflow_list.clear();
  // get out edgelist
  if (graph_comm_ptr_->GetNodeDataFlowMap(fus_node, fusion_op_anchors_map, fus_dataflow_list, 1) == FAILED) {
    return FAILED;
  }
  fus_node->SetFusionOutputFlowList(fus_dataflow_list);
  fus_node->GetOpDesc()->SetExtAttr("OutEdgeAnchorMap", fusion_op_anchors_map);

  SetDataOutPutMapingAttr(fusion_op_anchors_map);
  if (FEContextUtils::GetBuildMode() == ge::BUILD_MODE_TUNING ||
      (Configuration::Instance(AI_CORE_NAME).EnableL2Fusion() && CheckL2FusionFusionStrategy(fusion_graph))) {
    for (auto iter = fusion_op_anchors_map.cbegin(); iter != fusion_op_anchors_map.cend(); iter++) {
      for (auto iter_vec = iter->second.begin(); iter_vec != iter->second.end(); iter_vec++) {
        FE_CHECK_NOTNULL((iter_vec->first)->GetOwnerNode());
        std::string original_name =
            std::static_pointer_cast<ge::DataAnchor>(iter_vec->first)->GetOwnerNode()->GetName();
        ge::NodePtr fusion_node = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second)->GetOwnerNode();
        FE_CHECK_NOTNULL(fusion_node);
        std::int64_t fusion_anchor_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second)->GetIdx();
        FE_LOGD("originalName:%s, fusion_name:%s, fusion_anchor_idx:%ld.", original_name.c_str(),
                fusion_node->GetName().c_str(), fusion_anchor_idx);
        auto map_temp = fusion_op_name_map_all_.find(original_name);
        if (map_temp == fusion_op_name_map_all_.end()) {
          fusion_op_name_map_all_[original_name].insert(std::make_pair(fusion_anchor_idx, fusion_node));
        } else {
          if (map_temp->second.empty()) {
            REPORT_FE_ERROR("[SubGraphOpt][PostProcess][RefreshFusNode] Node[%s]'s fusion op name is empty.",
                            fus_node->GetName().c_str());
            return FAILED;
          }
          if (fusion_node->GetName() == map_temp->second.begin()->second->GetName()) {
            map_temp->second.emplace(std::make_pair(fusion_anchor_idx, fusion_node));
            fusion_op_name_map_all_.insert(std::make_pair(original_name, map_temp->second));
          } else {
            FE_LOGW("Node %s belongs to two nodes: %s and %s.", original_name.c_str(), fusion_node->GetName().c_str(),
                    map_temp->first.c_str());
          }
        }
      }
    }
  }

  FE_LOGD("Node %s refreshed successfully.", fus_node->GetName().c_str());
  return SUCCESS;
}

void FusionGraphMerge::CalcPassStridedOutSize(const ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist,
    vector<std::pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op) const {
  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return);
  auto fus_opdesc = fus_node_ptr->GetOpDesc();

  for (uint32_t i = 0; i < fus_opdesc->GetAllOutputsDescSize(); i++) {
    if (std::find(fus_nodelist.begin(), fus_nodelist.end(), src_op_out_index_in_fus_op[i].second) !=
        fus_nodelist.end() && i < src_op_out_index_in_fus_op.size()) {
      ge::OpDescPtr op_desc = src_op_out_index_in_fus_op[i].second->GetOpDesc();
      GetL2ToOpStructFromJson(op_desc, optimize_info);
      if (optimize_info == nullptr) {
        GetStridedToOpStructFromJson(op_desc, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
      }
      if (optimize_info == nullptr) {
        continue;
      }
      int64_t tensor_size = 0;
      int32_t real_calc_flag = 1;
      ge::GeTensorDesc tmp_td = src_op_out_index_in_fus_op[i].second->GetOpDesc()->GetOutputDesc(
          src_op_out_index_in_fus_op[i].first);
      auto cur_fus_opdesc = fus_node_ptr->GetOpDesc();
      if (!optimize_info->slice_output_shape.empty() &&
          src_op_out_index_in_fus_op[i].first < optimize_info->slice_output_shape.size() &&
          !optimize_info->slice_output_shape[src_op_out_index_in_fus_op[i].first].empty()) {
        tmp_td.SetShape(ge::GeShape(optimize_info->slice_output_shape[src_op_out_index_in_fus_op[i].first]));
      }
      (void)TensorComputeUtil::CalcTensorSize(tmp_td, tensor_size, real_calc_flag);
      (void)ge::AttrUtils::SetInt(cur_fus_opdesc->MutableOutputDesc(i), ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size);
    }
  }
}

void FusionGraphMerge::CalcSingleOpStridedOutSize(const ge::NodePtr &node) const {
  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return);
  ge::OpDescPtr op_desc = node->GetOpDesc();
  GetL2ToOpStructFromJson(op_desc, optimize_info);
  if (optimize_info == nullptr) {
    GetStridedToOpStructFromJson(op_desc, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
  }
  if (optimize_info == nullptr) {
    return;
  }
  if (!optimize_info->slice_output_shape.empty() && !optimize_info->slice_output_shape[0].empty()) {
    ge::GeTensorDesc tmp_td = node->GetOpDesc()->GetOutputDesc(0);
    tmp_td.SetShape(ge::GeShape(optimize_info->slice_output_shape[0]));
    int64_t tensor_size = 0;
    int32_t real_calc_flag = 1;
    (void)TensorComputeUtil::CalcTensorSize(tmp_td, tensor_size, real_calc_flag);
    (void)ge::AttrUtils::SetInt(op_desc->MutableOutputDesc(0), ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size);
  }
}

void FusionGraphMerge::CalcSingleOpStridedSize(ge::ComputeGraph &graph) const {
  /* When StridedWrite is connected to NetOutput, we need
   * to set _special_output_size attr. The reason is:
   * The output size of the whole model(which is _special_output_size
   * exactly) is larger than the tensor size of StridedWrite. So
   * we set special_output_size attr to let GE know the real
   * output size of this model. */
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  bool need_special_output_size =
          (build_mode_value == ge::BUILD_MODE_TUNING && step_mode_value == ge::BUILD_STEP_AFTER_UB_MATCH);
  if (!need_special_output_size) {
    return;
  }
  auto all_nodes = graph.GetDirectNode();
  for (auto &node : all_nodes) {
    auto op_desc = node->GetOpDesc();
    int64_t scope_id = 0;
    // only single op need do this
    (void)ge::AttrUtils::GetInt(op_desc, SCOPE_ID_ATTR, scope_id);
    if (scope_id >= 0) {
      continue;
    }
    CalcSingleOpStridedOutSize(node);
  }
}

Status FusionGraphMerge::CalcStridedWriteOutSize(const ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist,
    vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op) const {
  /* When StridedWrite is connected to NetOutput, we need
   * to set _special_output_size attr. The reason is:
   * The output size of the whole model(which is _special_output_size
   * exactly) is larger than the tensor size of StridedWrite. So
   * we set special_output_size attr to let GE know the real
   * output size of this model. */
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  bool need_special_output_size =
      (build_mode_value == ge::BUILD_MODE_TUNING && step_mode_value == ge::BUILD_STEP_AFTER_UB_MATCH);
  if (!need_special_output_size) {
    return SUCCESS;
  }

  CalcPassStridedOutSize(fus_node_ptr, fus_nodelist, src_op_out_index_in_fus_op);
  bool has_stridedwrite = false;
  ge::NodePtr strided_node = nullptr;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (node->GetType() == STRIDEDWRITE) {
      strided_node = node;
      has_stridedwrite = true;
      break;
    }
  }
  if (!has_stridedwrite) {
    return SUCCESS;
  }
  auto strided_opdesc = strided_node->GetOpDesc();
  /* stridedwrite node output format is NC1HWC0, so we deal it according to this format */
  FE_CHECK_NOTNULL(strided_opdesc->GetInputDescPtr(0));
  auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(strided_opdesc->GetInputDescPtr(0)->GetFormat()));
  if (primary_format != ge::FORMAT_NC1HWC0) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Merge][Stride] Strided write node [%s] format must be 5HD to calculate the actual output size.",
        strided_node->GetName().c_str());
    return FAILED;
  }
  int64_t batch_stride = 0;
  (void)ge::AttrUtils::GetInt(strided_node->GetOpDesc(), "stride", batch_stride);
  FE_CHECK_NOTNULL(strided_node->GetOpDesc()->MutableOutputDesc(0));
  int64_t batch_size = strided_node->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDim(1);
  int64_t n_batch = strided_node->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDim(0);
  if ((batch_size == 0) || (n_batch == 0)) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][Stride] batch_size[%ld] or n_batch[%ld] should not be 0.", batch_size,
                    n_batch);
    return FAILED;
  }

  int64_t tensor_size;
  auto tensor_desc = strided_node->GetOpDesc()->GetOutputDesc(0);
  int32_t output_real_calc_flag = 1;
  (void)TensorComputeUtil::CalcTensorSize(tensor_desc, tensor_size, output_real_calc_flag);
  int64_t real_out_size = (batch_size + (n_batch - 1) * batch_stride) * (tensor_size / (n_batch * batch_size));
  auto fus_opdesc = fus_node_ptr->GetOpDesc();
  (void)ge::AttrUtils::SetInt(fus_opdesc->MutableOutputDesc(0), ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, real_out_size);
  return SUCCESS;
}

Status FusionGraphMerge::CalcStridedReadInSize(const ge::NodePtr &fus_node_ptr,
                                               vector<ge::NodePtr> &fus_nodelist) const {
  bool has_stridedread = false;
  ge::NodePtr strided_node = nullptr;
  for (auto node : fus_nodelist) {
    if (node->GetType() == STRIDEDREAD) {
      strided_node = node;
      has_stridedread = true;
      break;
    }
  }
  if (!has_stridedread) {
    return SUCCESS;
  }
  auto strided_opdesc = strided_node->GetOpDesc();
  /* stridedwrite node output format is NC1HWC0, so we deal it according to this format */
  FE_CHECK_NOTNULL(strided_opdesc->GetInputDescPtr(0));
  auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(strided_opdesc->GetInputDescPtr(0)->GetFormat()));
  if (primary_format != ge::FORMAT_NC1HWC0) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Merge][Stride] Strided read node [%s] format must be 5HD to calculate the real output size.",
        strided_node->GetName().c_str());
    return FAILED;
  }
  int64_t batch_stride = 0;
  (void)ge::AttrUtils::GetInt(strided_node->GetOpDesc(), "stride", batch_stride);
  FE_CHECK_NOTNULL(strided_node->GetOpDesc()->MutableInputDesc(0));
  int64_t batch_size = strided_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDim(1);
  int64_t n_batch = strided_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDim(0);
  if ((batch_size == 0) || (n_batch == 0)) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][Stride] batch_size[%ld] or n_batch[%ld] should not be 0.", batch_size,
                    n_batch);
    return FAILED;
  }

  int64_t tensor_size;
  auto tensor_desc = strided_node->GetOpDesc()->GetInputDesc(0);
  int32_t output_real_calc_flag = 1;
  (void)TensorComputeUtil::CalcTensorSize(tensor_desc, tensor_size, output_real_calc_flag);
  int64_t real_in_size = (batch_size + (n_batch - 1) * batch_stride) * (tensor_size / (n_batch * batch_size));
  auto fus_opdesc = fus_node_ptr->GetOpDesc();
  (void)ge::AttrUtils::SetInt(fus_opdesc->MutableInputDesc(0), ge::ATTR_NAME_SPECIAL_INPUT_SIZE, real_in_size);
  return SUCCESS;
}

void FusionGraphMerge::DelAttrsInOriginalFusionOpGraph(ge::NodePtr &node) const {
  auto node_op_desc = node->GetOpDesc();
  std::vector<std::string> names_prefix;
  (void)ge::AttrUtils::GetListStr(node_op_desc, ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  for (const auto &prefix : names_prefix) {
    (void)node_op_desc->DelAttr(prefix + ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  }
  node_op_desc->DelAttr(ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  node_op_desc->DelAttr(COMPILE_INFO_JSON);
}

static bool IsNeedCreateOriginalFusion() {
  std::string env_ge_graph = Configuration::Instance(AI_CORE_NAME).GetDumpGeGraph();
  std::string env_graph_level = Configuration::Instance(AI_CORE_NAME).GetDumpGraphLevel();
  FE_LOGD("The value of dump ge graph is [%s] and the value of dump graph level is [%s].",
          env_ge_graph.c_str(), env_graph_level.c_str());
  if ((env_ge_graph.empty()) || (env_graph_level.empty())) {
    FE_LOGD("The environment variable of DUMP_GE_GRAPH or DUMP_GRAPH_LEVEL does not exist, do not need to create "
            "original fusion graph.");
    return true;
  } else if (env_graph_level != kMaxDumpGraphLevel) {
    return true;
  }
  return false;
}

void FusionGraphMerge::CreateOriginalFusionOpGraph(ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist) const {
  if (!UnknownShapeUtils::IsUnknownShapeOp(*fus_node_ptr->GetOpDesc()) && IsNeedCreateOriginalFusion()) {
    FE_LOGD("Op[name:%s] is known shape op, and do not need to create original fusion graph.",
            fus_node_ptr->GetName().c_str());
    return;
  }
  ge::CompleteGraphBuilder builder("fused_nodes");
  vector<std::string> node_name_list;
  for (auto &node : fus_nodelist) {
    node_name_list.emplace_back(node->GetName());
  }
  std::vector<ge::NodePtr> output_nodes;
  for (auto &node : fus_nodelist) {
    /* Adds a node to the builder. Currently, only the input parameter op_desc
       is supported, and the connection information will be lost. Therefore,
       the AddDataLink interface needs to be invoked to re-establish the
       connection. */
    DelAttrsInOriginalFusionOpGraph(node);
    builder.AddNode(node->GetOpDesc());
    FE_LOGD("Begin re-establish connection of node[name:%s]. out data size is %u.", node->GetName().c_str(),
            node->GetAllOutDataAnchorsSize());
    // Re-establish connection
    const auto &src_node_name = node->GetName();
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      if (out_data_anchor == nullptr) {
        return;
      }
      auto src_idx = out_data_anchor->GetIdx();
      if (out_data_anchor->GetPeerInDataAnchors().empty()) {
        FE_LOGD("Node[name:%s] is out op, skip.", node->GetName().c_str());
        output_nodes.emplace_back(node);
        continue;
      }
      for (auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        if (in_data_anchor == nullptr) {
          return;
        }
        auto dst_node = in_data_anchor->GetOwnerNode();
        if (dst_node == nullptr) {
          return;
        }
        auto iter = std::find(node_name_list.cbegin(), node_name_list.cend(), dst_node->GetName());
        if (iter == node_name_list.cend()) {
          FE_LOGD("Node[name:%s] is out op, skip.", node->GetName().c_str());
          continue;
        }
        builder.AddDataLink(src_node_name, src_idx, dst_node->GetName(), in_data_anchor->GetIdx());
      }
    }
  }

  ge::OpDescPtr fusion_op_desc = fus_node_ptr->GetOpDesc();
  std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_input_anchors_map;
  shared_ptr<std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>>> fusion_op_input_anchors_map_ptr = nullptr;
  FE_MAKE_SHARED((fusion_op_input_anchors_map_ptr =
                 std::make_shared<std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>>>(
                 fusion_op_desc->TryGetExtAttr("InEdgeAnchorMap", fusion_op_input_anchors_map))), return);
  FE_CHECK(fusion_op_input_anchors_map_ptr == nullptr,
           FE_LOGE("Null pointer is unexpected, current op[name:%s].", fus_node_ptr->GetName().c_str()), return );
  FE_LOGD("fusionOpInputAnchorsMap size is %zu.", fusion_op_input_anchors_map_ptr->size());
  std::map<uint32_t, uint32_t> input_mapping;
  for (auto iter = fusion_op_input_anchors_map_ptr->cbegin(); iter != fusion_op_input_anchors_map_ptr->cend(); iter++) {
    FE_LOGD("input data anchor size is %zu", iter->second.size());
    for (auto iter_vec = iter->second.begin(); iter_vec != iter->second.end(); iter_vec++) {
      FE_CHECK(iter_vec->first == nullptr, FE_LOGW("iterVec->first is null"), return );
      FE_CHECK(iter_vec->second == nullptr, FE_LOGW("iterVec->second is null"), return );
      std::uint32_t src_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->first)->GetIdx();
      std::uint32_t fusion_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second)->GetIdx();
      ge::NodePtr src_node = std::static_pointer_cast<ge::DataAnchor>(iter_vec->first)->GetOwnerNode();
      ge::NodePtr fusion_node = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second)->GetOwnerNode();
      FE_LOGD("srcIdx is %u, fusion_idx is %u. src_node is %s, fusion_node is %s.", src_idx, fusion_idx,
              src_node->GetName().c_str(), fusion_node->GetName().c_str());
      // add relation: input index of new graph to input index of original graph.
      std::vector<std::string> input_node_names{src_node->GetName()};
      std::vector<uint32_t> dst_indexes{src_idx};
      builder.SetInput(fusion_idx, input_node_names, dst_indexes);
      input_mapping.emplace(fusion_idx, fusion_idx);
    }
  }
  builder.SetInputMapping(input_mapping);

  std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_output_anchors_map;
  fusion_op_output_anchors_map = fusion_op_desc->TryGetExtAttr("OutEdgeAnchorMap", fusion_op_output_anchors_map);
  FE_LOGD("fusionOpOutputAnchorsMap size is %zu.", fusion_op_output_anchors_map.size());
  std::map<uint32_t, uint32_t> output_mapping;
  std::uint32_t output_mapping_index = 0;
  for (auto out_iter = fusion_op_output_anchors_map.cbegin(); out_iter != fusion_op_output_anchors_map.cend();
       out_iter++) {
    FE_LOGD("output data anchor size is %zu", out_iter->second.size());
    std::map<int, std::pair<ge::AnchorPtr, ge::AnchorPtr>> output_anchor_order_map;
    for (const std::pair<const ge::AnchorPtr, ge::AnchorPtr> &anchor_pair : out_iter->second) {
      int fusion_anchor_idx = anchor_pair.second->GetIdx();
      output_anchor_order_map.emplace(fusion_anchor_idx, anchor_pair);
    }
    for (auto iter_vec = output_anchor_order_map.cbegin(); iter_vec != output_anchor_order_map.cend(); iter_vec++) {
      FE_CHECK(iter_vec->second.first == nullptr, FE_LOGW("iterVec->first is null"), return );
      FE_CHECK(iter_vec->second.second == nullptr, FE_LOGW("iterVec->second is null"), return );
      std::uint32_t src_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second.first)->GetIdx();
      std::uint32_t fusion_idx = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second.second)->GetIdx();
      ge::NodePtr src_node = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second.first)->GetOwnerNode();
      ge::NodePtr fusion_node = std::static_pointer_cast<ge::DataAnchor>(iter_vec->second.second)->GetOwnerNode();
      FE_LOGD("out src_idx is %u, out fusion_idx is %u outputmapping_idx is %u. src_node is %s, fusion_node is %s.",
              src_idx, fusion_idx, output_mapping_index, src_node->GetName().c_str(), fusion_node->GetName().c_str());
      // add relation: output index of new graph to output index of original graph.
      std::vector<std::string> input_node_names{src_node->GetName()};
      std::vector<uint32_t> dst_indexes{src_idx};
      builder.AddOutput(src_node->GetName(), src_idx);
      output_mapping.emplace(output_mapping_index, fusion_idx);
      output_mapping_index++;
    }
  }
  if (output_mapping.empty()) {
    FE_LOGD("The size of output nodes is %zu.", output_nodes.size());
    for (auto &node : output_nodes) {
      builder.AddOutput(node->GetName(), 0);
      std::uint32_t fusion_idx = output_mapping_index;
      output_mapping.emplace(output_mapping_index, fusion_idx);
      output_mapping_index++;
    }
  }
  builder.SetOutputMapping(output_mapping);
  builder.SetParentNode(fus_node_ptr);

  ge::graphStatus status = ge::GRAPH_SUCCESS;
  string err_msg;
  auto graph = builder.Build(status, err_msg);
  FE_LOGD("Build graph ret is %u, err_msg is %s", status, err_msg.c_str());
  (void)ge::AttrUtils::SetGraph(fusion_op_desc, "_original_fusion_graph", graph);
}
}  // namespace fe
