/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/conv_requant_slice_info.h"
#include "common/fe_log.h"

namespace fe {
Status ConvRequantSliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                      const vector<ge::NodePtr> &fusion_nodes,
                                                      OpCalcInfo &op_calc_info, size_t &input_size,
                                                      const bool &is_head_fusion) {
  // if requant node's req_scale input is vector, no need to add split info
  (void) fusion_nodes;
  (void) is_head_fusion;
  ge::GeTensorDesc req_scale_tensor = fusion_node->GetOpDesc()->GetInputDesc(1);
  if (req_scale_tensor.GetOriginShape().GetDims().size() <= 1) {
    FE_LOGI("The requested scale shape dimension is less than 2.");
    return SUCCESS;
  }
  // 2. set requant(double input) input split info
  InputSplitInfo requant_input_split_info;
  if (!requant_input_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] requant_input_split_info initialization failed");
    return FAILED;
  }
  requant_input_split_info.SetIndex(input_size - 1);
  // requant(double input) can only split 0 axis
  std::vector<int64_t> axis = {0};
  requant_input_split_info.SetAxis(axis);
  // requant(double input)'s overlap must be minus one
  std::vector<int64_t> over_lap = {-1};
  requant_input_split_info.SetHeadOverLap(over_lap);
  requant_input_split_info.SetTailOverLap(over_lap);

  // 3. add requant(double input) input split info for each split map
  std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
  for (auto &axis_split_map : axis_split_maps) {
    axis_split_map.AddInputSplitInfo(requant_input_split_info);
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}
}  // namespace fe
