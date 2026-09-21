/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TBE_OP_FUNC_MANAGER_H_
#define TBE_OP_FUNC_MANAGER_H_

#include <string>
#include <map>
#include "tensor_engine/tbe_op_info.h"

using namespace std;
using te::TbeOpInfo;
using te::CheckSupportedResult;
using CheckSupportFunc = function<bool(const TbeOpInfo &, CheckSupportedResult &, std::string &)>;
using SelectFormatFunc = function<bool(const TbeOpInfo &, string &)>;

class TbeOpFuncManager {
 public:
  static TbeOpFuncManager& Instance();

  void AddCheckSupportFunction(const std::string &op_type, const CheckSupportFunc &func);
  void AddSelectFormatFunction(const std::string &op_type, const SelectFormatFunc &func);

  bool GetCheckSupportFunction(const std::string &op_type, CheckSupportFunc &func);
  bool GetSelectFormatFunction(const std::string &op_type, SelectFormatFunc &func);

 private:
  map<string, CheckSupportFunc> check_support_func_map_;
  map<string, SelectFormatFunc> select_format_func_map_;
};

class TbeOpSelectFormatFuncRegister {
 public:
  TbeOpSelectFormatFuncRegister(const std::string &op_type, const SelectFormatFunc &func) {
    TbeOpFuncManager::Instance().AddSelectFormatFunction(op_type, func);
  }
};

class TbeOpCheckSupportFuncRegister {
public:
  TbeOpCheckSupportFuncRegister(const std::string &op_type, const CheckSupportFunc &func) {
    TbeOpFuncManager::Instance().AddCheckSupportFunction(op_type, func);
  }
};

#define IMPL_CHECK_SUPPORT_FUNC(op_type) \
  bool CheckSupport##op_type(const TbeOpInfo& op_info, CheckSupportedResult &is_support, std::string &reason)

#define IMPL_SELECT_FORMAT_FUNC(op_type) \
  static bool SelectFormat##op_type(const TbeOpInfo &op_info, string &format_str)

#define __CHECK_SUPPORT_FUNC_REG_IMPL__(op_type, func, count) \
  static const TbeOpCheckSupportFuncRegister FuncCheckSupport##op_type##count(#op_type, func);

#define REG_CHECK_SUPPORT_FUNC(op_type) __CHECK_SUPPORT_FUNC_REG_IMPL__(op_type, CheckSupport##op_type, __COUNTER__)

#define __SELECT_FORMAT_FUNC_REG_IMPL__(op_type, func, count) \
  static const TbeOpSelectFormatFuncRegister FuncSelectFormat##op_type##count(#op_type, func);

#define REG_SELECT_FORMAT_FUNC(op_type) __SELECT_FORMAT_FUNC_REG_IMPL__(op_type, SelectFormat##op_type, __COUNTER__)

#endif  // TBE_OP_FUNC_MANAGER_H_
