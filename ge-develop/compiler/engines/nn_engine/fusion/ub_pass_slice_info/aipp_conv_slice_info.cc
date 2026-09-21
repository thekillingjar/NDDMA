/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/aipp_conv_slice_info.h"
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/lxfusion_json_util.h"

namespace fe {
Status AippConvSliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                   const vector<ge::NodePtr> &fusion_nodes,
                                                   OpCalcInfo &op_calc_info, size_t &input_size,
                                                   const bool &is_head_fusion) {
  (void) fusion_nodes;
  (void) input_size;
  (void) is_head_fusion;
  ge::GeTensorDesc input_desc = fusion_node->GetOpDesc()->GetInputDesc(0);
  ge::Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc.GetFormat()));
  if (input_format != ge::FORMAT_NCHW && input_format != ge::FORMAT_NHWC) {
    FE_LOGD("Node[%s] format is neither NCHW nor NHWC, will not modify slice info.", fusion_node->GetName().c_str());
    return SUCCESS;
  }

  // delete H axis slice info
  int32_t axis_h = GetAxisIndexByFormat(input_format, "H");
  if (axis_h >= 0) {
    std::vector<int64_t> axis_cut = {axis_h};
    std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
    op_calc_info.DelAxisSplitMapBaseAxis(axis_cut);
  }
  // delete W axis slice info
  int32_t axis_w = GetAxisIndexByFormat(input_format, "W");
  if (axis_w >= 0) {
    std::vector<int64_t> axis_cut = {axis_w};
    std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
    op_calc_info.DelAxisSplitMapBaseAxis(axis_cut);
  }
  return SUCCESS;
}
}  // namespace fe
