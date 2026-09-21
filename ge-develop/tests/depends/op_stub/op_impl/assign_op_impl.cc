/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "common/checker.h"
using namespace gert;
namespace ops {
ge::graphStatus InferShapeForAssign(InferShapeContext *context) {
  GELOGD("InferShapeForAssign");
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  *output_shape = input_shape_1;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Assign).InferShape(InferShapeForAssign);

}  // namespace ops
