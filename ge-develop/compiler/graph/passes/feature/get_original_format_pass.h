/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_GET_ORIGINAL_FORMAT_PASS_H_
#define GE_GRAPH_PASSES_GET_ORIGINAL_FORMAT_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {
/// Set the original format of operator which is sensitive to format in order to change the real format.
/// The original format parameters only used in online model generate phase.
class GetOriginalFormatPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);

 private:
  /// Whether format transpose or not
  /// @author
  bool IsFormatTranspose(const ge::OpDescPtr op_ptr, int32_t ori_format) const;

  /// Set the original format of operator
  /// @author
  Status SetOriginalFormat(const ge::ComputeGraphPtr &graph) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_GET_ORIGINAL_FORMAT_PASS_H_
