/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "generator/preprocess/var_info.h"
namespace att {
void to_json(nlohmann::json &j, const std::vector<HardwareDef> &scopes) {
  std::vector<std::string> scope_vec;
  for (const auto &p : scopes) {
    scope_vec.push_back(BaseTypeUtils::DumpHardware(p));
  }
  j = nlohmann::json{
    {scope_vec},
  };
}

void to_json(nlohmann::json &j, const Replacement &exprs) {
  j = nlohmann::json{
    {"ori_expr", exprs.orig_expr},
    {"new_epxr", exprs.new_replaced_expr},
  };
}

void to_json(nlohmann::json& j, const VarInfo& p) {
  j = nlohmann::json {
    {"align", p.align},
    {"scopes", p.scopes},
    {"replacement", p.replacement},
    {"cut_leq_cons", p.cut_leq_cons},
    {"cut_eq_cons", p.cut_eq_cons},
    {"is_input_var", p.is_input_var},
    {"is_const_var", p.is_const_var},
    {"do_search", p.do_search},
    {"is_node_innerest_dim_size", p.is_node_innerest_dim_size},
    {"init_value", p.init_value},
    {"max_value", p.max_value},
    {"min_value", p.min_value},
    {"parent_size", p.from_axis_size},
    {"const_value", p.const_value},
    {"orig_axis_size", p.orig_axis_size},
  };
}

std::string MakeJson(const ExprInfoMap& expr_info_map) {
  nlohmann::json j;
  for (const auto &p : expr_info_map) {
    j[Str(p.first)] = p.second;
  }
  return j.dump();
}
}  // namespace att