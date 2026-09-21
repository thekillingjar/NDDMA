/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_NO_USE_RESHAPE_REMOVE_PASS_H_
#define GE_GRAPH_PASSES_NO_USE_RESHAPE_REMOVE_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
class NoUseReshapeRemovePass : public BaseNodePass {
 public:
  /// Entry of the NoUseReshapeRemovePass optimizer
  /// To satisfy fusion rule of FE, remove reshape op which input & output format is same
  /// @param [in] node: Input Node
  /// @return SUCCESS: Dont find need to delete node
  /// @return NOT_CHANGED: find need to delete node
  /// @return OTHERS:  Execution failed
  /// @author
  Status Run(ge::NodePtr &node) override;

 private:
  Status TryRemoveConstShapeInput(NodePtr &reshape_node);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_NO_USE_RESHAPE_REMOVE_PASS_H_
