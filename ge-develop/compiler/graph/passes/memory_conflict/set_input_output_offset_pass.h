/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SET_INPUT_OUTPUT_OFFSET_PASS_H_
#define GE_GRAPH_PASSES_SET_INPUT_OUTPUT_OFFSET_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {
class SetInputOutputOffsetPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  Status SetInputOffset(const NodePtr &node, const std::vector<int32_t> &connect_input) const;
  Status SetOutputOffset(const NodePtr &node, const std::vector<int32_t> &connect_output) const;
  Status SetInputOffsetForFusion(const std::vector<int64_t> &memory_type, const ge::NodePtr &node) const;
  Status SetInputOffsetForHcom(const NodePtr &node, const std::vector<int32_t> &connect_input) const;
  Status SetOutputOffsetForConcat(const NodePtr &node) const;
  Status SetOutputOffsetForHcom(const NodePtr &node, const std::vector<int32_t> &connect_output) const;

  bool CheckBufferFusion(const NodePtr &node) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SET_INPUT_OUTPUT_OFFSET_PASS_H_
