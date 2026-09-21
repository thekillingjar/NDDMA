/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_ITERATOR_FUSION_PASS_H_
#define GE_GRAPH_PASSES_ITERATOR_FUSION_PASS_H_

#include "common/graph_pass.h"
#include "register/register_fmk_types.h"

namespace ge {
class IteratorFusionPass : public GraphPass {
 public:
  explicit IteratorFusionPass(domi::FrameworkType type) : fmk_type_(type) {}

  ~IteratorFusionPass() override {};

  Status Run(ge::ComputeGraphPtr graph) final;

 private:
  domi::FrameworkType fmk_type_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_ITERATOR_FUSION_PASS_H_
