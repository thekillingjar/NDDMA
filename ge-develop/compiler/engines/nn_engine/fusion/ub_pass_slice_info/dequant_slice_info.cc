/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/dequant_slice_info.h"
#include "common/fe_log.h"
#include "graph/utils/type_utils.h"

namespace fe {
Status DequantSliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node) {
  FE_CHECK_NOTNULL(fusion_node);
  auto op_desc = fusion_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  // check if format is NC1HWC0
  auto input_desc = op_desc->MutableInputDesc(0);
  FE_CHECK_NOTNULL(input_desc);
  ge::Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc->GetFormat()));
  if (input_format != ge::FORMAT_NC1HWC0) {
    FE_LOGI("Input format not supported: current is %s, expected %s.",
            ge::TypeUtils::FormatToSerialString(input_format).c_str(),
            ge::TypeUtils::FormatToSerialString(ge::FORMAT_NC1HWC0).c_str());
   return SUCCESS;
  }

  // op_slice_info set by op does not contain Cout axis split, see impl file ascend_dequant.py get_op_support_info()
  AxisSplitMap new_axis_split_map;
  if (!new_axis_split_map.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] axis_split_map initialize failed");
    return FAILED;
  }
  // set dequant input split info
  InputSplitInfo dequant_input0_split_info;
  if (!dequant_input0_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] dequant_input_split_info initialization failed");
    return FAILED;
  }
  vector<int64_t> cout_axis_cut = vector<int64_t> {kNc1hwc0CoutCutAxis[0]};
  dequant_input0_split_info.SetIndex(0);
  dequant_input0_split_info.SetAxis(cout_axis_cut); // shape = NC1HWC0, adopt Cout cut
  std::vector<int64_t> over_lap = {-1};
  dequant_input0_split_info.SetHeadOverLap(over_lap);
  dequant_input0_split_info.SetTailOverLap(over_lap);
  new_axis_split_map.AddInputSplitInfo(dequant_input0_split_info);
  // if dequant node's deq_scale input is vector, add split info
  auto deq_scale_desc = op_desc->MutableInputDesc(1);
  FE_CHECK_NOTNULL(deq_scale_desc);
  auto deq_scale_dim = deq_scale_desc->GetOriginShape().GetDims();
  if (deq_scale_dim.size() > 0 && deq_scale_dim[0] > 1) {
    InputSplitInfo dequant_input1_split_info;
    if (!dequant_input1_split_info.Initialize()) {
      REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] dequant_input_split_info initialization failed");
      return FAILED;
    }
    dequant_input1_split_info.SetIndex(1);
    dequant_input1_split_info.SetAxis(cout_axis_cut); // shape = NC1HWC0, adopt Cout cut
    dequant_input1_split_info.SetHeadOverLap(over_lap);
    dequant_input1_split_info.SetTailOverLap(over_lap);
    new_axis_split_map.AddInputSplitInfo(dequant_input1_split_info);
  }
  // set dequant output split info
  OutputSplitInfo dequant_output_split_info;
  if (!dequant_output_split_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] dequant_input_split_info initialization failed");
    return FAILED;
  }
  dequant_output_split_info.SetIndex(0);
  dequant_output_split_info.SetAxis(cout_axis_cut);
  new_axis_split_map.AddOutputSplitInfo(dequant_output_split_info);

  OpCalcInfo op_calc_info;
  if (!op_calc_info.Initialize()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbSliceInfo][MdfSliceInfo] Failed to initialize op_calc_info");
    return FAILED;
  }
  if (GetOpSliceInfoForSingleOp(fusion_node, op_calc_info) == FAILED) {
    FE_LOGD("Node [%s] cannot be cut; check unsuccessful.", fusion_node->GetName().c_str());
    return FAILED;
  }
  op_calc_info.AddAxisSplitMap(new_axis_split_map);
  SetOpSliceInfoForSingleOp(fusion_node, op_calc_info);

  return SUCCESS;
}
} // namespace fe
