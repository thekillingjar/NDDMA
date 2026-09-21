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
#ifndef DOMI_OP_VARIABLE_H_
#define DOMI_OP_VARIABLE_H_
#include <vector>
#include "parser/common/op_def/operator.h"
#include "framework/omg/parser/parser_types.h"

namespace ge {
class VariableOperator : public ParserOperator {
 public:
  VariableOperator();
  ~VariableOperator() override;

  VariableOperator &Container(const std::string &container);

  VariableOperator &SharedName(const std::string &sharedname);

  VariableOperator &Placement(const std::string &placement);

  VariableOperator &MemType(const uint32_t &mem_type);

  VariableOperator &SrcType(const int64_t &dtype);

  VariableOperator &VarShape(const std::vector<int64_t> &shape_value);

  int64_t GetVarSrcType() const;
};
}  // namespace ge

#endif  // DOMI_OP_VAR_H_ AUTO GEN PLEASE DO NOT MODIFY IT
