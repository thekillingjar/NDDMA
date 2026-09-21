/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_UB_PASS_SLICE_INFO_UB_PASS_SLICE_INFO_BASE_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_UB_PASS_SLICE_INFO_UB_PASS_SLICE_INFO_BASE_H_

#include <vector>
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/node.h"

namespace fe {
const std::vector<int64_t> kNc1hwc0NCutAxis = {0};
const std::vector<int64_t> kNc1hwc0HCutAxis = {2};
const std::vector<int64_t> kNc1hwc0WCutAxis = {3};
const std::vector<int64_t> kNc1hwc0CoutCutAxis = {1};

class UbPassSliceInfoBase {
 public:
  virtual ~UbPassSliceInfoBase() = default;
  virtual Status ModifySliceInfoByPattern(const ge::NodePtr &fusion_node, const vector<ge::NodePtr> &fusion_nodes,
                                          OpCalcInfo &op_calc_info, size_t &input_size, const bool &is_head_fusion);
  virtual Status ModifySliceInfoByPattern(const ge::NodePtr &fusion_node);
  Status SetOpSliceInfoForSingleOp(const ge::NodePtr& node, OpCalcInfo& op_slice_info) const;
  Status GetOpSliceInfoForSingleOp(const ge::NodePtr& node, OpCalcInfo& op_slice_info) const;
};
using UbPassSliceInfoBasePtr = std::shared_ptr<UbPassSliceInfoBase>;
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_UB_PASS_SLICE_INFO_UB_PASS_SLICE_INFO_BASE_H_
