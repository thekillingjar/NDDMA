/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
graphStatus InferShape4ApplyAdamD(gert::InferSymbolShapeContext *context) {
  auto var_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(var_shape);
  auto m_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(m_shape);
  auto v_shape = context->GetInputSymbolShape(2);
  GE_UNSUPPORTED_IF_NULL(v_shape);

  auto varout_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(varout_shape);
  auto mout_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(mout_shape);
  auto vout_shape = context->GetOutputSymbolShape(2);
  GE_ASSERT_NOTNULL(vout_shape);

  *varout_shape = *var_shape;
  *mout_shape = *m_shape;
  *vout_shape = *v_shape;
  return ge::GRAPH_SUCCESS;
}


IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ApplyAdamD).InferSymbolShape(InferShape4ApplyAdamD);
}
} // namespace ge