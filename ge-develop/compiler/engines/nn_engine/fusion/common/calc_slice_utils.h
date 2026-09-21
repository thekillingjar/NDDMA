/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_COMMON_CALC_SLICE_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_COMMON_CALC_SLICE_UTILS_H_

#include <vector>
#include "graph/node.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
using BufferFusionPassBasePtr = std::unique_ptr<BufferFusionPassBase>;
class CalcSliceUtils {
public:
  static void CalcSliceInfo(const BufferFusionPassBasePtr &buffer_fusion_pass_base_ptr,
                            std::vector<ge::NodePtr> &fusion_nodes);
private:
  static bool Stratege1(const BufferFusionPassBasePtr &buffer_fusion_pass_base_ptr,
                        std::vector<ge::NodePtr> &sorted_fusion_nodes, OpCalcInfo &op_calc_info);
  static bool Stratege2(std::vector<ge::NodePtr> &sorted_fusion_nodes, OpCalcInfo &op_calc_info);
  static void SetOpSliceInfoForFusionOp(const std::vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_calc_info);
  static bool UpdateOpSliceInfoForSpecificOp(const ge::NodePtr &fusion_node, const std::string &op_pattern);
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_COMMON_CALC_SLICE_UTILS_H_
