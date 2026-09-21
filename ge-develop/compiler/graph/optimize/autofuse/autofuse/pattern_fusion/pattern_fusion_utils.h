/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_PATTERN_FUSION_UTILS_H
#define AUTOFUSE_PATTERN_FUSION_UTILS_H

#include "graph/node.h"
#include "graph/ge_tensor.h"
#include "lowering/lowerings.h"

namespace ge {
namespace pattern_fusion {
inline bool SameShape(const GeShape &shape1, const GeShape &shape2) {
  if (shape1.GetDimNum() != shape2.GetDimNum()) {
    return false;
  }
  const auto &dims1 = shape1.GetDims();
  const auto &dims2 = shape2.GetDims();
  for (size_t i = 0; i < dims1.size(); ++i) {
    if (dims1[i] != dims2[i]) {
      return false;
    }
  }
  return true;
}

// 注意：broadcast类型也会被标记成PointWise，所以这里增加输入输出shape一致的校验, 更符合后端对Elementwise的理解
// 注意：Transpose/TransposeD的FuseType是pointwise，很坑。。。
inline bool IsElementwise(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  // 排除会改变数据布局的操作
  const auto &node_type = node->GetType();
  if (node_type == "Transpose" || node_type == "TransposeD") {
    return false;
  }
  LoweringManager::Lowering(node);
  const auto node_out_anchor = node->GetOutDataAnchor(0);
  const auto node_kernel_box = loop::GetKernelBox(node_out_anchor);
  if (FuseTypeToString(node_kernel_box.Type()) != "pointwise") {
    return false;
  }
  // 检查输入输出shape是否相同，排除broadcast等会改变shape的操作
  const auto &input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape();
  const auto &output_shape = node->GetOpDesc()->GetOutputDesc(0).GetShape();
  return SameShape(input_shape, output_shape);
}

inline ge::SymbolicDescAttr *GetNodeMutableOutputAttr(const NodePtr &node) {
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_output_desc = node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(node_output_desc);
  auto node_output_attr = node_output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
  return node_output_attr;
}

inline ge::SymbolicDescAttr *GetNodeMutableInputAttr(const NodePtr &node) {
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_input_desc = node_op_desc->MutableInputDesc(0);
  GE_ASSERT_NOTNULL(node_input_desc);
  auto node_input_attr = node_input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
  return node_input_attr;
}

inline const gert::SymbolShape &GetNodeSymbolShape(const NodePtr &node) {
  static const gert::SymbolShape kEmptySymbolShape;
  auto node_output_attr = GetNodeMutableOutputAttr(node);
  if (node_output_attr == nullptr) {
    return kEmptySymbolShape;
  }
  return node_output_attr->symbolic_tensor.GetOriginSymbolShape();
}

// 设置节点的shape和符号化shape
inline Status SetNodeShape(const NodePtr &node, const GeShape &input_shape, const GeShape &output_shape,
                           const gert::SymbolShape &symbol_shape) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  // 设置所有输入的shape
  auto input_descs = op_desc->GetAllInputsDescPtr();
  for (const auto &input_desc : input_descs) {
    input_desc->SetShape(input_shape);
    input_desc->SetOriginShape(input_shape);
    auto node_input_attr = input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node_input_attr);
    node_input_attr->symbolic_tensor.MutableOriginSymbolShape() = symbol_shape;
  }

  // 设置所有输出的shape
  auto output_descs = op_desc->GetAllOutputsDescPtr();
  for (const auto &output_desc : output_descs) {
    output_desc->SetShape(output_shape);
    output_desc->SetOriginShape(output_shape);
    auto node_output_attr = output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node_output_attr);
    node_output_attr->symbolic_tensor.MutableOriginSymbolShape() = symbol_shape;
  }

  return SUCCESS;
}
}  // namespace pattern_fusion
}  // namespace ge

#endif  // AUTOFUSE_PATTERN_FUSION_UTILS_H
