/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor.h"

#include "exe_graph/runtime/infer_shape_context.h"
#include "sequence_op_impl.h"

using namespace ge;
namespace gert {
void SetOpInfo(const ge::OpDescPtr &op_desc, Format format, DataType dt, std::initializer_list<int64_t> shape) {
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(format);
    op_desc->MutableInputDesc(i)->SetOriginFormat(format);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetDataType(dt);
    op_desc->MutableInputDesc(i)->SetOriginDataType(dt);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(format);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(format);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetDataType(dt);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(dt);
  }

  return;
}

ge::graphStatus InferShapeSplitToSequence(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SplitToSequence).InferShape(InferShapeSplitToSequence);

ge::graphStatus InferShapeSequenceConstruct(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SequenceConstruct).InferShape(InferShapeSequenceConstruct);

ge::graphStatus InferShapeSequenceErase(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SequenceErase).InferShape(InferShapeSequenceErase);

ge::graphStatus InferShapeSequenceInsert(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SequenceInsert).InferShape(InferShapeSequenceInsert);

ge::graphStatus InferShapeSequenceAtCompute(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SequenceAt).InferShape(InferShapeSequenceAtCompute);

ge::graphStatus InferShapeSequenceEmpty(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SequenceEmpty).InferShape(InferShapeSequenceEmpty);

ge::graphStatus InferShapeSequenceLength(InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(SequenceLength).InferShape(InferShapeSequenceLength);

}  // namespace gert
