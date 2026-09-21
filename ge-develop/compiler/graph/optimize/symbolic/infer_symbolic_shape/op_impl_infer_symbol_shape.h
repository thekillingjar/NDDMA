/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_IMPL_INFER_SYMBOL_SHAPE_H
#define OP_IMPL_INFER_SYMBOL_SHAPE_H

#include "graph/any_value.h"
#include "register/op_impl_kernel_registry.h"

namespace gert {
class OpImplInferSymbolShape {
 public:
  OpImplInferSymbolShape() = default;
  explicit OpImplInferSymbolShape(const ge::char_t *op_type_param) : op_type(op_type_param) {}
  OpImplInferSymbolShape &InferSymbolShape(OpImplKernelRegistry::InferSymbolShapeKernelFunc func);
  OpImplRegisterV2::OpType op_type;
};

class OpImplInferSymbolShapeRegistry {
 public:
  static OpImplInferSymbolShapeRegistry &GetInstance();
  const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImpl(const ge::char_t *op_type) const;

 private:
  friend class OpImplInferSymbolShape;
  OpImplInferSymbolShapeRegistry() = default;
  ~OpImplInferSymbolShapeRegistry() = default;
  OpImplInferSymbolShapeRegistry(const OpImplInferSymbolShapeRegistry &single) = delete;
  OpImplInferSymbolShapeRegistry &operator=(const OpImplInferSymbolShapeRegistry &single) = delete;

  OpImplKernelRegistry::OpImplFunctionsV2 *CreateOrGetOpImpl(const ge::char_t *op_type);
  std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctionsV2> types_to_impl_;
};

#define IMPL_OP_INFER_SYMBOL_SHAPE_INNER_COUNTER(op_type, name, counter) \
  static gert::OpImplInferSymbolShape VAR_UNUSED name##counter = gert::OpImplInferSymbolShape(#op_type)
#define IMPL_OP_INFER_SYMBOL_SHAPE_INNER_COUNTER_NUMBER(op_type, name, counter) \
  IMPL_OP_INFER_SYMBOL_SHAPE_INNER_COUNTER(op_type, name, counter)
#define IMPL_OP_INFER_SYMBOL_SHAPE_INNER(op_type) \
  IMPL_OP_INFER_SYMBOL_SHAPE_INNER_COUNTER_NUMBER(op_type, op_impl_register_v2_##op_type, __COUNTER__)
}  // namespace gert
#endif  // OP_IMPL_INFER_SYMBOL_SHAPE_H
