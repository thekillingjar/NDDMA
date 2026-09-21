/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/conv_dequants16_slice_info.h"
#include "common/fe_log.h"
#include "common/lxfusion_json_util.h"

namespace fe {
static const size_t MIN_INPUT_NUM = 1;
static const size_t FIRST_INDEX = 1;
static const size_t SECOND_INDEX = 2;
Status ConvDequantS16SliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                         const vector<ge::NodePtr> &fusion_nodes,
                                                         OpCalcInfo &op_calc_info, size_t &input_size,
                                                         const bool &is_head_fusion) {
  // if dequant node's deq_scale input is vector, no need to add split info
  (void) fusion_nodes;
  (void) is_head_fusion;
  ge::GeTensorDesc deq_scale_tensor = fusion_node->GetOpDesc()->GetInputDesc(FIRST_INDEX);
  if (deq_scale_tensor.GetOriginShape().GetDims().size() <= MIN_INPUT_NUM) {
    return SUCCESS;
  }
  // 2. set dequant(trible input) input split info
  InputSplitInfo dequant_input_split_info;
  if (!dequant_input_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] dequant_input_split_info initialize failed");
    return FAILED;
  }
  dequant_input_split_info.SetIndex(input_size - SECOND_INDEX);
  // dequant(double input) can only split 0 axis
  std::vector<int64_t> axis = {1};
  dequant_input_split_info.SetAxis(axis);
  // dequant(double input)'s overlap must be minus one
  std::vector<int64_t> over_lap = {-1};
  dequant_input_split_info.SetHeadOverLap(over_lap);
  dequant_input_split_info.SetTailOverLap(over_lap);

  // 2. set dequant(trible input) input split info
  InputSplitInfo dequant_x1_input_split_info;
  if (!dequant_x1_input_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] dequant_x1_input_split_info initialization failed");
    return FAILED;
  }
  dequant_x1_input_split_info.SetIndex(input_size - FIRST_INDEX);
  // dequant(double input) can only split 0 axis
  dequant_x1_input_split_info.SetAxis(axis);
  // dequant(double input)'s overlap must be minus one
  dequant_x1_input_split_info.SetHeadOverLap(over_lap);
  dequant_x1_input_split_info.SetTailOverLap(over_lap);

  // 3. add dequant(double input) input split info for each split map
  std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
  for (auto &axis_split_map : axis_split_maps) {
    std::vector<InputSplitInfo> input_split_infos = axis_split_map.GetInputSplitInfoVec();
    for (auto input_split_info : input_split_infos) {
      std::vector<int64_t> axes = input_split_info.GetAxis();
      if (axes == axis) {
        axis_split_map.AddInputSplitInfo(dequant_input_split_info);
        axis_split_map.AddInputSplitInfo(dequant_x1_input_split_info);
      }
    }
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}
}  // namespace fe
