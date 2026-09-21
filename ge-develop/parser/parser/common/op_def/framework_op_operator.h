/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DOMI_OP_FRAMEWORKOP_OP_OPERATOR_H_
#define DOMI_OP_FRAMEWORKOP_OP_OPERATOR_H_
#include "graph/debug/ge_attr_define.h"
#include "parser/common/op_def/operator.h"

namespace ge {
class FrameworkOpOperator : public ParserOperator {
 public:
  FrameworkOpOperator();

  ~FrameworkOpOperator() override;

  FrameworkOpOperator &OriginalType(const std::string &type);

  FrameworkOpOperator &NodeDefPkg(const std::string &nodedef_pkg);

  FrameworkOpOperator &Frameworktype(int64_t framework_type);

  FrameworkOpOperator &TfOpDef(const std::string &opdef_string);

  FrameworkOpOperator &Index(int64_t index);

  FrameworkOpOperator &FuncDefPkg(const std::string &func_string);

  int64_t GetFrameworkType() const;

  std::string GetNodeDefPkg() const;
};
}  // namespace ge

#endif  // DOMI_OP_FRAMEWORKOP_OP_OPERATOR_H_
