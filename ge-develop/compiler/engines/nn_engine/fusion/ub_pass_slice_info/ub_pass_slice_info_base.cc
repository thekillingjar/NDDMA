/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_slice_info/ub_pass_slice_info_base.h"
#include "common/fe_log.h"
#include "common/lxfusion_json_util.h"

namespace fe {
Status UbPassSliceInfoBase::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node,
                                                     const vector<ge::NodePtr> &fusion_nodes,
                                                     OpCalcInfo &op_calc_info, size_t &input_size,
                                                     const bool &is_head_fusion) {
  (void)is_head_fusion;
  (void)input_size;
  (void)op_calc_info;
  (void)fusion_nodes;
  (void)fusion_node;
  
  return SUCCESS;
}

Status UbPassSliceInfoBase::ModifySliceInfoByPattern(const ge::NodePtr &fusion_node) {
  (void)fusion_node;
  return SUCCESS;
}

Status UbPassSliceInfoBase::GetOpSliceInfoForSingleOp(const ge::NodePtr &node, OpCalcInfo &op_slice_info) const {
  FE_CHECK_NOTNULL(node);
  if (op_slice_info.IsPtrNull()) {
    return FAILED;
  }
  std::string op_calc_info_str;
  if (!ge::AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_calc_info_str) || op_calc_info_str.empty()) {
    return FAILED;
  }
  GetOpSliceInfoFromJson(op_slice_info, op_calc_info_str);
  return SUCCESS;
}

Status UbPassSliceInfoBase::SetOpSliceInfoForSingleOp(const ge::NodePtr &node, OpCalcInfo &op_slice_info) const {
  FE_CHECK_NOTNULL(node);
  if (op_slice_info.IsPtrNull()) {
    return FAILED;
  }
  if (op_slice_info.GetAxisSplitMaps().size() != 0) {
    string op_calc_info_str;
    SetOpSliceInfoToJson(op_slice_info, op_calc_info_str);
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), OP_SLICE_INFO, op_calc_info_str);
  }
  return SUCCESS;
}
}  // namespace fe
