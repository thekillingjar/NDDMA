/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_ARG_LIST_MANAGER_H_
#define EXPR_GEN_ARG_LIST_MANAGER_H_

#include <unordered_map>
#include <vector>
#include "parser/tuning_space.h"
#include "base/base_types.h"

namespace att {
constexpr char kArgsNameTmpBuffer[] = "tmp_tbuf_size";
constexpr char kArgsNameBuiltInTmpBuffer[] = "BuiltinTmpBuffer";

inline std::string GetTmpBufferName(int32_t index) {
  return std::string("tmp_tbuf_") + std::to_string(index) + "_size";
}

class ArgListManager {
public:
  static ArgListManager &GetInstance() {
    static ArgListManager arg_list_mgr;
    return arg_list_mgr;
  }

  // 按照目标名保存表达式
  ge::Status SetArgExpr(const std::string &name, const Expr &expr) {
    if (arg_list_map_.find(name) == arg_list_map_.end()) {
      arg_list_map_[name] = expr;
    }
    return ge::SUCCESS;
  }

  // 添加表达式
  ge::Status AddArgExpr(const std::string &name, const Expr &size) {
    return SetArgExpr(name, size);
  }

  // 根据name查找表达式
  Expr GetArgExpr(const std::string &name) {
    if (arg_list_map_.find(name) != arg_list_map_.end()) {
      return arg_list_map_[name];
    }
    Expr res;
    return res;
  }
  
  Expr GetArgExpr(const std::vector<std::string> &names) {
    Expr cur_var = CreateExpr(1U);
    for (auto &name : names) {
      if (arg_list_map_.find(name) == arg_list_map_.end()) {
        Expr res;
        return res;
      }
      cur_var = ge::sym::Mul(arg_list_map_[name], cur_var);
    }
    return cur_var;
  }

  // 获取所有表达式
  void GetArgList(std::vector<Expr> &arg_list) {
    for (const auto &arg : arg_list_map_) {
      if(arg.second.IsValid()) {
        arg_list.emplace_back(arg.second);
      }
    }
  }

  ge::Status GetArgList(const std::vector<std::string> &names, std::vector<Expr> &arg_list) {
    arg_list.clear();
    for (const auto &name : names) {
      if (arg_list_map_.find(name) == arg_list_map_.end()) {
        GELOGE(ge::FAILED, "Variable [%s] has no expr.", name.c_str());
        return ge::FAILED;
      }
      arg_list.emplace_back(arg_list_map_[name]);
    }
    return ge::SUCCESS;
  }

  ExprExprMap GetVariableExprMap() const {
    ExprExprMap res;
    for (const auto &tensor : replace_container_) {
      res[tensor.variable] = tensor.expr;
    }
    return res;
  }

  std::map<Expr, std::string, ExprCmp> GetVariableNameMap() const {
    std::map<Expr, std::string, ExprCmp> res;
    for (const auto &tensor : replace_container_) {
      res[tensor.variable] = tensor.name;
    }
    return res;
  }

  // 设置轴、buffer、queue、tensor size表达式
  ge::Status LoadArgList(const TuningSpacePtr &tuning_space);
  
  Expr SetTensorInfo(const std::string &name, const Expr &expr) {
    size_t idx = replace_container_.size();
    std::string tensor_name = "tensor_" + std::to_string(idx);
    Expr tensor_expr = CreateExpr(tensor_name.c_str());
    TensorInfo tensor_info(name, tensor_expr, expr);
    replace_container_.emplace_back(tensor_info);
    GELOGD("Rename tensor [%s] to [%s], value is [%s]", name.c_str(), tensor_name.c_str(), Str(expr).c_str());
    return tensor_expr;
  }

private:
  // 设置tensor大小的表达式
  ge::Status SetTensorSizeExpr(const std::vector<TensorPtr> &allocated_tensors);

private:
  ArgListManager() = default;
  ~ArgListManager() = default;
  std::unordered_map<std::string, Expr> arg_list_map_;
  std::vector<TensorInfo> replace_container_;
};
}

#endif