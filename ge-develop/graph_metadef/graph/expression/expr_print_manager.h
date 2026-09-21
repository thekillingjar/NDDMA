/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_EXPRESSION_EXPR_PRINT_MANAGER_H_
#define GRAPH_EXPRESSION_EXPR_PRINT_MANAGER_H_
#include <map>
#include <string>
#include <vector>

#include "expression_impl.h"

namespace ge {
using OpPrinter = std::string (*)(const std::vector<SymEngineExprPtr> &args, StrType type);

class ExprManager {
 public:
  static ExprManager &GetInstance() {
    static ExprManager instance;
    return instance;
  }

  void RegisterDefaultOpPrinter(const OperationType &opType, const OpPrinter &operation) {
    defaultOpPrinters_[opType] = operation;
  }

  OpPrinter GetPrinter(const OperationType &type)
  {
    return GetPrinterHelper(type, defaultOpPrinters_);
  }

 private:
  ExprManager() = default;
  ~ExprManager() = default;
  ExprManager(const ExprManager&) = delete;
  ExprManager &operator=(const ExprManager&) = delete;
  OpPrinter GetPrinterHelper(const OperationType op, const std::map<OperationType, OpPrinter> &printerMap) const {
    const auto iter = printerMap.find(op);
    if (iter == printerMap.end()) {
      return nullptr;
    }
    return iter->second;
  }

  std::map<OperationType, OpPrinter> defaultOpPrinters_;
};

class ExprManagerRegister
{
 public:
  ExprManagerRegister(const OperationType op, const OpPrinter &printer) {
    ExprManager::GetInstance().RegisterDefaultOpPrinter(op, printer);
  }
  ~ExprManagerRegister() = default;
};
} // namespace ge

#define REGISTER_EXPR_DEFAULT_PRINTER(opType, funcName) \
  ExprManagerRegister register_##opType_default_##funcName(opType, funcName)
#endif  // GRAPH_EXPRESSION_EXPR_PRINT_MANAGER_H_