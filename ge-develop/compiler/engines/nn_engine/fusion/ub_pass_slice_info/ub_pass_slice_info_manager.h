/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_UB_PASS_SLICE_INFO_UB_PASS_SLICE_INFO_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_UB_PASS_SLICE_INFO_UB_PASS_SLICE_INFO_MANAGER_H_

#include "ub_pass_slice_info/ub_pass_slice_info_base.h"

namespace fe {
enum class UbMatchedType {
  UBMATCHTYPE_AIPP = 0,
  UBMATCHTYPE_CONV2D,
  UBMATCHTYPE_READ_WRITE_SELECT,
  UBMATCHTYPE_STRIDED_READ_WRITE_SELECT,
  UBMATCHTYPE_DEQUANT,
  UBMATCHTYPE_REQUANT,
  UBMATCHTYPE_ELTWISE,
  UBMATCHTYPE_POOLING,
  UBMATCHTYPE_DEQUANTS16,
  UBMATCHTYPE_REQUANTS16,
  MATCHED_RESERVED
};

const std::map<std::string, UbMatchedType> kOpPatternUbMatchedTypeMap = {
        {"aipp", UbMatchedType::UBMATCHTYPE_AIPP},
        {"write_select", UbMatchedType::UBMATCHTYPE_READ_WRITE_SELECT},
        {"read_select", UbMatchedType::UBMATCHTYPE_READ_WRITE_SELECT},
        {"strided_write", UbMatchedType::UBMATCHTYPE_STRIDED_READ_WRITE_SELECT},
        {"strided_read", UbMatchedType::UBMATCHTYPE_STRIDED_READ_WRITE_SELECT},
        {"dequant", UbMatchedType::UBMATCHTYPE_DEQUANT},
        {"requant", UbMatchedType::UBMATCHTYPE_REQUANT},
        {"ElemWise", UbMatchedType::UBMATCHTYPE_ELTWISE},
        {"Broadcast", UbMatchedType::UBMATCHTYPE_ELTWISE},
        {"Pool2d", UbMatchedType::UBMATCHTYPE_POOLING},
        {"MaxPool", UbMatchedType::UBMATCHTYPE_POOLING},
        {"dequant_s16", UbMatchedType::UBMATCHTYPE_DEQUANTS16},
        {"requant_s16", UbMatchedType::UBMATCHTYPE_REQUANTS16}
};

class UbPassSliceInfoManager {
 public:
  static Status SetSliceInfoForFusionNodes(vector<ge::NodePtr> &fusion_nodes);

  static UbPassSliceInfoBasePtr SwitchSliceInfoPtrByPattern(const UbMatchedType &ub_matched_pattern,
                                                            const ge::NodePtr &fusion_node,
                                                            size_t &input_size);

  static bool CheckOpPatternSupport(const string &op_pattern);

  static bool CheckOpPatternSliceInfoUpdate(const string &op_pattern);

  static Status CalcSliceInfoForFusionOp(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info);
private:
  static bool IsHeadFusion(const ge::NodePtr &fusion_node, const vector<ge::NodePtr> &fusion_nodes);
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_UB_PASS_SLICE_INFO_UB_PASS_SLICE_INFO_MANAGER_H_
