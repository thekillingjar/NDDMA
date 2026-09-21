/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/calc_slice_utils.h"
#include "common/fe_log.h"
#include "common/aicore_util_attr_define.h"
#include "common/lxfusion_json_util.h"
#include "ub_pass_slice_info/ub_pass_slice_info_manager.h"

namespace fe {
namespace {
bool CompareByNodeId(const ge::NodePtr &left_node, const ge::NodePtr &right_node) {
  if (left_node == nullptr || right_node == nullptr) {
    return false;
  }
  return left_node->GetOpDesc()->GetId() < right_node->GetOpDesc()->GetId();
}
bool TopoSortForFusionNodes(const std::vector<ge::NodePtr> &fusion_nodes,
                            std::vector<ge::NodePtr> &sorted_fusion_nodes) {
  for (auto &fusion_node : fusion_nodes) {
    // check opDesc to ensure that there exists no nullptr when sorting
    if (fusion_node == nullptr || fusion_node->GetOpDesc() == nullptr) {
      return false;
    }
    sorted_fusion_nodes.push_back(fusion_node);
  }
  std::sort(sorted_fusion_nodes.begin(), sorted_fusion_nodes.end(), CompareByNodeId);
  return true;
}
}
void CalcSliceUtils::CalcSliceInfo(const BufferFusionPassBasePtr &buffer_fusion_pass_base_ptr,
                                   std::vector<ge::NodePtr> &fusion_nodes) {
  ge::NodePtr first_node = fusion_nodes.at(0);
  if (first_node == nullptr) {
    return;
  }

  std::string slice_info_str;
  (void)ge::AttrUtils::GetStr(first_node->GetOpDesc(), FUSION_OP_SLICE_INFO, slice_info_str);
  if (!slice_info_str.empty()) {
    FE_LOGD("FusionOp's slice info has been set, no need to calculate again.");
    return;
  }

  OpCalcInfo op_calc_info;
  if (!op_calc_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UB][CalcSliceInfo] op_calc_info initialize failed");
    return;
  }

  // adopt TopologicalSorting first to ensure that input/output indexes for fusionOp can be calculated correctlly
  vector<ge::NodePtr> sorted_fusion_nodes;
  bool enable_stratege1 = TopoSortForFusionNodes(fusion_nodes, sorted_fusion_nodes);
  // CalcFusionOpSliceInfo() may be implemented in fusion passes to calculate slice info,
  // mainlly for passes defined in cann
  bool enable_stratege2 = false;
  if (enable_stratege1) {
    enable_stratege2 = (!Stratege1(buffer_fusion_pass_base_ptr, sorted_fusion_nodes, op_calc_info));
  }
  // if stratege1 runs failed, adopt default fusionOpSliceInfo calculation process, mainlly for passes defined in fe
  bool enable_stratege3 = true;
  if (enable_stratege2) {
    enable_stratege3 = (!Stratege2(sorted_fusion_nodes, op_calc_info));
  }
  // if above strateges run failed, adopt default fusionOpSliceInfo calculation process,
  // only for passes containing conv2d node
  if (enable_stratege3) {
    UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusion_nodes);
  }
}

bool CalcSliceUtils::Stratege1(const BufferFusionPassBasePtr &buffer_fusion_pass_base_ptr,
                               vector<ge::NodePtr> &sorted_fusion_nodes, OpCalcInfo &op_calc_info) {
  if (buffer_fusion_pass_base_ptr != nullptr  &&
      buffer_fusion_pass_base_ptr->CalcFusionOpSliceInfo(sorted_fusion_nodes, op_calc_info) != SUCCESS) {
    vector<AxisSplitMap> empty_map;
    op_calc_info.SetAxisSplitMaps(empty_map);
  }
  // set sliceinfo for fusionOp
  if (op_calc_info.GetAxisSplitMaps().size() != 0) {
    SetOpSliceInfoForFusionOp(sorted_fusion_nodes, op_calc_info);
    FE_LOGD("Successfully calculated op_slice info using Strategy1.");
    return true;
  } else {
    FE_LOGD("Calculate op_slice info using Strategy1 unsuccessful.");
    return false;
  }
}

bool CalcSliceUtils::Stratege2(std::vector<ge::NodePtr> &sorted_fusion_nodes, OpCalcInfo &op_calc_info) {
  for (auto &fusion_node : sorted_fusion_nodes) {
    if (fusion_node == nullptr) {
      return false;
    }
    std::string op_pattern;
    (void)ge::AttrUtils::GetStr(fusion_node->GetOpDesc(), kOpPattern, op_pattern);
    bool condition = (UbPassSliceInfoManager::CheckOpPatternSupport(op_pattern) &&
                      UpdateOpSliceInfoForSpecificOp(fusion_node, op_pattern));
    if (!condition) {
      FE_LOGD("Not support op_pattern [%s], Strategy2 run unsuccessful.", op_pattern.c_str());
      return false;
    }
  }
  if (UbPassSliceInfoManager::CalcSliceInfoForFusionOp(sorted_fusion_nodes, op_calc_info) != SUCCESS) {
    vector<AxisSplitMap> empty_map;
    op_calc_info.SetAxisSplitMaps(empty_map);
  }
  // set sliceinfo for fusionOp
  if (op_calc_info.GetAxisSplitMaps().size() != 0) {
    SetOpSliceInfoForFusionOp(sorted_fusion_nodes, op_calc_info);
    FE_LOGD("Succeeded in calculating op_slice info using Strategy2.");
    return true;
  } else {
    FE_LOGD("Calculate op_slice info by Strategy2 unsuccessful.");
    return false;
  }
}

void CalcSliceUtils::SetOpSliceInfoForFusionOp(const std::vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_calc_info) {
  std::string op_calc_info_str;
  SetFusionOpSliceInfoToJson(op_calc_info, op_calc_info_str);
  for (const ge::NodePtr &fusion_node : fusion_nodes) {
    (void)ge::AttrUtils::SetStr(fusion_node->GetOpDesc(), FUSION_OP_SLICE_INFO, op_calc_info_str);
  }
}

bool CalcSliceUtils::UpdateOpSliceInfoForSpecificOp(const ge::NodePtr &fusion_node, const std::string &op_pattern) {
  if (UbPassSliceInfoManager::CheckOpPatternSliceInfoUpdate(op_pattern)) {
    // refine op_slice_info for some specified op_pattern
    auto iter = kOpPatternUbMatchedTypeMap.find(op_pattern);
    UbMatchedType matched_pattern =
            ((iter == kOpPatternUbMatchedTypeMap.end()) ? UbMatchedType::MATCHED_RESERVED : iter->second);
    size_t tmp_val = 0;
    auto slice_info_base_ptr =
            UbPassSliceInfoManager::SwitchSliceInfoPtrByPattern(matched_pattern, fusion_node, tmp_val);
    if (slice_info_base_ptr == nullptr || slice_info_base_ptr->ModifySliceInfoByPattern(fusion_node) != SUCCESS) {
      return false;
    }
  }
  return true;
}
}
