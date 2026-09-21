/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "iterator_fusion_pass.h"

#include <memory>

#include "framework/omg/parser/parser_types.h"
#include "common/util.h"
#include "parser_graph_optimizer.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
Status IteratorFusionPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  domi::FrameworkType fmk_type = static_cast<domi::FrameworkType>(fmk_type_);
  std::unique_ptr<ParserGraphOptimizer> graph_optimizer(new (std::nothrow) ParserGraphOptimizer(graph, fmk_type));
  if (graph_optimizer == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New ParserGraphOptimizer failed");
    return FAILED;
  }
  return graph_optimizer->FusionFmkop();
}
}  // namespace ge
