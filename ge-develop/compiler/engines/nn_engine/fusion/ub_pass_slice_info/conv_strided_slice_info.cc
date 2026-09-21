/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/conv_strided_slice_info.h"
#include "common/fe_log.h"

namespace fe {
Status ConvStridedSliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                      const vector<ge::NodePtr> &fusion_nodes,
                                                      OpCalcInfo &op_calc_info, size_t &input_size,
                                                      const bool &is_head_fusion) {
  (void) fusion_nodes;
  (void) is_head_fusion;
  (void) input_size;

  // 1. get axis from strided node
  int32_t axis = -1;
  (void)ge::AttrUtils::GetInt(fusion_node->GetOpDesc(), "axis", axis);

  // 2. Del axis cut info from StridedRead split info
  if (axis >= 0) {
    std::vector<int64_t> axis_cut = {axis};
    std::vector<AxisSplitMap> axis_split_maps = op_calc_info.GetAxisSplitMapVec();
    op_calc_info.DelAxisSplitMapBaseAxis(axis_cut);
  }
  return SUCCESS;
}
}  // namespace fe
