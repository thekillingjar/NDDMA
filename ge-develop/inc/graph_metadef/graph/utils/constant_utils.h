/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_UTILS_CONSTANT_UTILS_H_
#define COMMON_GRAPH_UTILS_CONSTANT_UTILS_H_
#include "graph/node.h"
#include "graph/op_desc.h"

namespace ge {
class ConstantUtils {
 public:
  // check is constant
  static bool IsConstant(const NodePtr &node);
  static bool IsConstant(const OpDescPtr &op_desc);
  static bool IsPotentialConst(const OpDescPtr &op_desc);
  static bool IsRealConst(const OpDescPtr &op_desc);
  // get/set  weight
  static bool GetWeight(const OpDescPtr &op_desc, const uint32_t index, ConstGeTensorPtr &weight);
  static bool MutableWeight(const OpDescPtr &op_desc, const uint32_t index, GeTensorPtr &weight);
  static bool SetWeight(const OpDescPtr &op_desc, const uint32_t index, const GeTensorPtr weight);
  static bool MarkPotentialConst(const OpDescPtr &op_desc, const std::vector<int> indices,
                                 const std::vector<GeTensorPtr> weights);
  static bool UnMarkPotentialConst(const OpDescPtr &op_desc);
  // for fileconstant
  static bool GetWeightFromFile(const OpDescPtr &op_desc, ConstGeTensorPtr &weight);
 private:
  static bool GetPotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                 std::vector<ConstGeTensorPtr> &weights);
  static bool MutablePotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                     std::vector<GeTensorPtr> &weights);
};
}

#endif // COMMON_GRAPH_UTILS_CONSTANT_UTILS_H_
