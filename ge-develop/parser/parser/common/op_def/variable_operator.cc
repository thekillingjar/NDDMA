/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_def/variable_operator.h"

#include "graph/debug/ge_attr_define.h"

namespace ge {
VariableOperator::VariableOperator() : ParserOperator(ge::parser::VARIABLE) {}

VariableOperator::~VariableOperator() {}

VariableOperator &VariableOperator::Container(const std::string &container) {
  Attr(VAR_ATTR_CONTAINER, container);
  return *this;
}

VariableOperator &VariableOperator::SharedName(const std::string &sharedname) {
  Attr(VAR_ATTR_SHARED_NAME, sharedname);
  return *this;
}

VariableOperator &VariableOperator::Placement(const std::string &placement) {
  Attr(ATTR_VARIABLE_PLACEMENT, placement);
  return *this;
}

VariableOperator &VariableOperator::MemType(const uint32_t &mem_type) {
  Attr(ATTR_OUTPUT_MEMORY_TYPE, mem_type);
  return *this;
}

VariableOperator &VariableOperator::SrcType(const int64_t &dtype) {
  Attr(VAR_ATTR_DTYPE, dtype);
  return *this;
}

VariableOperator &VariableOperator::VarShape(const std::vector<int64_t> &shape_value) {
  Attr(VAR_ATTR_SHAPE, shape_value);
  return *this;
}

int64_t VariableOperator::GetVarSrcType() const { return GetIntAttr(VAR_ATTR_DTYPE); }
}  //  namespace ge
