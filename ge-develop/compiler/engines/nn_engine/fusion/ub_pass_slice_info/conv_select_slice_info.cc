/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/conv_select_slice_info.h"
#include "common/fe_log.h"

namespace fe {
Status ConvSelectSliceInfo::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                     const vector<ge::NodePtr> &fusion_nodes,
                                                     OpCalcInfo &op_calc_info, size_t &input_size,
                                                     const bool &is_head_fusion) {
  (void) fusion_node;
  (void) input_size;
  (void) is_head_fusion;
  (void) fusion_nodes;
  std::vector<int64_t> axis_cut_h = {2};
  std::vector<int64_t> axis_cut_w = {3};
  // Del Hcut info and Wcut info from select split info
  op_calc_info.DelAxisSplitMapBaseAxis(axis_cut_h);
  op_calc_info.DelAxisSplitMapBaseAxis(axis_cut_w);
  return SUCCESS;
}
}  // namespace fe
