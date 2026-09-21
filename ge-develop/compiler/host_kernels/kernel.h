/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_INC_KERNEL_H_
#define GE_INC_KERNEL_H_

#include <vector>

#include "framework/common/op/ge_op_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/op_desc.h"

namespace ge {
/// @ingroup domi_omg
/// @brief Kernel interface
class Kernel {
 public:

  /// Constant calculation interface, the result is appended to output
  /// @param [in] op_desc_ptr Operator related parameters
  /// @param [in] input Constant to be calculated
  /// @param [inout] output Save calculation results
  virtual Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr>& input,
                         std::vector<ge::GeTensorPtr>& v_output) {
    (void)op_desc_ptr;
    (void)input;
    (void)v_output;
    return UNSUPPORTED;
  }

  virtual Status Compute(const NodePtr& node, std::vector<GeTensorPtr>& v_output) const {
    (void)node;
    (void)v_output;
    return UNSUPPORTED;
  }

  virtual Status Compute(const NodePtr& node_ptr) const {
    (void)node_ptr;
    return UNSUPPORTED;
  }

  /// Destructor
  virtual ~Kernel() {}
};
}  // namespace ge
#endif  // GE_INC_KERNEL_H_
