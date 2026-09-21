/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GENERATOR_PREPROCESS_VAR_INFO_H_
#define ATT_GENERATOR_PREPROCESS_VAR_INFO_H_
#include <string>
#include <vector>
#include <cstdint>

#include "base/base_types.h"
#include "nlohmann/json.hpp"
#include "util/base_types_printer.h"

namespace att {
struct Replacement
{
  Expr orig_expr;
  Expr new_replaced_expr;
};

struct VarInfo {
  VarInfo() = default;
  ~VarInfo() = default;
  Expr align = ge::Symbol(1);;  // 变量替换前的符号align值由图上确定，替换后的符号align值为1
  uint32_t prompt_align{1u}; // 性能最好的align
  uint32_t data_type_size{4U}; // 类型占用的内存大小
  bool is_concat_outer_dim; // 是否是concat node的concat dim外轴
  bool is_concat_inner_dim; // 是否是concat node的concat dim尾轴
  std::pair<int64_t, int64_t> value_range{-1, -1}; // 变量的最大值和最小值
  std::vector<HardwareDef> scopes;
  Replacement replacement;  // 用来表达替换前后的映射
  std::vector<Expr> cut_leq_cons;  // 符号相关的切分不等式约束
  std::vector<std::pair<Expr, Expr>> cut_eq_cons;  // 符号相关的等式约束
  bool is_input_var{false};
  bool is_const_var{false};
  bool do_search{false};
  bool is_node_innerest_dim_size{false};
  Expr init_value;
  Expr min_value;
  Expr max_value;
  std::vector<Expr> from_axis_size;
  uint32_t const_value{0u};
  std::vector<Expr> orig_axis_size;
  std::vector<std::string> orig_axis_name;
};
using ExprInfoMap = std::map<Expr, VarInfo, ExprCmp>;
using ExprExprMap = std::map<Expr, Expr, ExprCmp>;
std::string MakeJson(const ExprInfoMap& expr_info_map);
void to_json(nlohmann::json &j, const std::vector<HardwareDef> &scopes);
void to_json(nlohmann::json &j, const Replacement &exprs);
void to_json(nlohmann::json& j, const VarInfo& p);
}  // namespace

#endif  // ATT_GENERATOR_PREPROCESS_VAR_INFO_H_