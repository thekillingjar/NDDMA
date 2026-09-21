/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AUTO GEN PLEASE DO NOT MODIFY IT
#include "common/op_def/shape_n_operator.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/omg/parser/parser_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY ShapeNOperator::ShapeNOperator() : ParserOperator("ShapeN") {}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator::~ShapeNOperator() {}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::N(int64_t n) {
  Attr(SHAPEN_ATTR_N, n);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY int64_t ShapeNOperator::GetN() const { return GetIntAttr(SHAPEN_ATTR_N); }

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::InType(ge::DataType t) {
  Attr(SHAPEN_ATTR_IN_TYPE, static_cast<int64_t>(t));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY ge::DataType ShapeNOperator::GetInType() const {
  return static_cast<ge::DataType>(GetIntAttr(SHAPEN_ATTR_IN_TYPE));
}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::OutType(ge::DataType t) {
  Attr(SHAPEN_ATTR_OUT_TYPE, static_cast<int64_t>(t));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY ge::DataType ShapeNOperator::GetOutType() const {
  return static_cast<ge::DataType>(GetIntAttr(SHAPEN_ATTR_OUT_TYPE));
}
}  // namespace ge
