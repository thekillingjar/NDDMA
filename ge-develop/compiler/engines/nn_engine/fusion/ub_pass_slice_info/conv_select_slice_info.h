/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_UB_PASS_SLICE_INFO_CONV_SELECT_SLICE_INFO_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_UB_PASS_SLICE_INFO_CONV_SELECT_SLICE_INFO_H_

#include "ub_pass_slice_info/ub_pass_slice_info_base.h"

namespace fe {
class ConvSelectSliceInfo : public UbPassSliceInfoBase {
 protected:
 using UbPassSliceInfoBase::ModifySliceInfoByPattern;
  Status ModifySliceInfoByPattern(const ge::NodePtr &fusion_node, const vector<ge::NodePtr> &fusion_nodes,
                                  OpCalcInfo &op_calc_info, size_t &input_size, const bool &is_head_fusion) override;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_UB_PASS_SLICE_INFO_CONV_SELECT_SLICE_INFO_H_
