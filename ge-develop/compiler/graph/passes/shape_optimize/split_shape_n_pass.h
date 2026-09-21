/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SPLIT_SHAPEN_PASS_H_
#define GE_GRAPH_PASSES_SPLIT_SHAPEN_PASS_H_

#include <vector>
#include "graph/passes/base_pass.h"

namespace ge {
class SplitShapeNPass : public BaseNodePass {
 public:
  Status Run(NodePtr& node) override;
  
 private:
  Status SplitShapeN(NodePtr &node);
  Status RelinkAnchors(const ComputeGraphPtr &graph, const NodePtr &node, const std::string &desc_name);
  void Clear();

  std::vector<int32_t> known_index_;
  std::vector<int32_t> unknown_index_;
  std::vector<ConstGeTensorDescPtr> known_input_desc_;
  std::vector<ConstGeTensorDescPtr> known_output_desc_;
  std::vector<ConstGeTensorDescPtr> unknown_input_desc_;
  std::vector<ConstGeTensorDescPtr> unknown_output_desc_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SPLIT_SHAPEN_PASS_H_
