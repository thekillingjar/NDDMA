/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/conv_requants16_slice_info.h"
#include "common/fe_log.h"

namespace fe {
static const int32_t OUTPUT_INDEX_ONE = 1;
static const size_t FIRST_INDEX = 1;
static const size_t SECOND_INDEX = 2;

Status ConvRequantS16SliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                         const vector<ge::NodePtr> &fusion_nodes,
                                                         OpCalcInfo &op_calc_info, size_t &input_size,
                                                         const bool &is_head_fusion) {
  (void) fusion_nodes;
  size_t req_index = -1;
  size_t x1_index = -1;
  if (is_head_fusion) {
    FE_LOGD("Node [%s] is in head fusion mode.", fusion_node->GetName().c_str());
    req_index = input_size - SECOND_INDEX;
    x1_index = input_size - FIRST_INDEX;
  } else {
    FE_LOGD("Node[%s] fusion mode is tail fusion.", fusion_node->GetName().c_str());
    req_index = input_size - FIRST_INDEX;
    x1_index = input_size - SECOND_INDEX;
  }
  // 2. set requant(trible input) input split info
  InputSplitInfo requant_input_split_info;
  if (!requant_input_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] requant_input_split_info initialize failed");
    return FAILED;
  }
  requant_input_split_info.SetIndex(req_index);
  // requant(double input) can only split 0 axis
  std::vector<int64_t> axis = {1};
  std::vector<int64_t> x1_axis = {0};
  requant_input_split_info.SetAxis(axis);
  // requant(double input)'s overlap must be minus one
  std::vector<int64_t> over_lap = {-1};
  requant_input_split_info.SetHeadOverLap(over_lap);
  requant_input_split_info.SetTailOverLap(over_lap);

  // 2. add requant(double input) input split info for each split map
  std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
  for (auto &axis_split_map : axis_split_maps) {
    std::vector<InputSplitInfo> input_split_infos = axis_split_map.GetInputSplitInfoVec();
    for (auto &input_split_info : input_split_infos) {
      std::vector<int64_t> axes = input_split_info.GetAxis();
      if (axes == axis) {
        axis_split_map.AddInputSplitInfo(requant_input_split_info);
      }
      // 3. set requant(trible input) input split info
      InputSplitInfo requant_x1_input_split_info;
      if (!requant_x1_input_split_info.Initialize()) {
        REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] requant_x1_input_split_info initialization failed");
        return FAILED;
      }
      requant_x1_input_split_info.SetIndex(x1_index);
      // requant(trible input) can only split 0 axis
      requant_x1_input_split_info.SetAxis(x1_axis);
      // requant(trible input)'s overlap must be minus one
      requant_x1_input_split_info.SetHeadOverLap(over_lap);
      requant_x1_input_split_info.SetTailOverLap(over_lap);
      axis_split_map.AddInputSplitInfo(requant_x1_input_split_info);
    }
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  if (fusion_node->GetAllOutDataAnchorsSize() > 1) {
    SetOutputSliceInfoForRequantS16(op_calc_info);
  }
  return SUCCESS;
}

Status ConvRequantS16SliceInfo::SetOutputSliceInfoForRequantS16(OpCalcInfo &op_calc_info) const {
  // 2. set elemwise(double output) output split info
  OutputSplitInfo elemwise_output_split_info;
  if (!elemwise_output_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][SetOutSliceInfo] elemwise_output_split_info initialization failed");
    return FAILED;
  }
  elemwise_output_split_info.SetIndex(OUTPUT_INDEX_ONE);
  // elemwise(double output) can only split 0 axis
  std::vector<int64_t> axis = {0};
  elemwise_output_split_info.SetAxis(axis);

  // 3. add elemwise(double output) output split info for each split map
  std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
  for (auto &axis_split_map : axis_split_maps) {
    axis_split_map.AddOutputSplitInfo(elemwise_output_split_info);
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}
}  // namespace fe
