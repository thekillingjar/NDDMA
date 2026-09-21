/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_UTILS_BASE_TYPES_PRINTER_H_
#define ATT_UTILS_BASE_TYPES_PRINTER_H_
#include <map>
#include <set>
#include <string>
#include "nlohmann/json.hpp"
#include "base/base_types.h"
#include "generator/preprocess/ast_optimizer.h"

namespace att {
const std::string AddAnotationBlock(std::string strs, std::string indent = "");
const std::string AddAnotationLine(std::string strs, std::string indent = "");
const std::string GenWorkspaceRelatedVars(const std::map<int64_t, Expr> &workspace_size_map, const ExprExprMap &container_expr);
const std::string GenRelatedVars(const std::vector<Expr> &funcs, const ExprExprMap &container_expr, const std::map<Expr, std::vector<Expr>, ExprCmp> &args);
const std::string GenBufRelatedVars(const Expr &func, const ExprExprMap &container_expr);
const std::map<HardwareDef, std::string> kHardwareNameMap = {
  {HardwareDef::GM, "hbm_size"},
  {HardwareDef::L1, "l1_size"},
  {HardwareDef::L2, "l2_size"},
  {HardwareDef::L0A, "l0a_size"},
  {HardwareDef::L0B, "l0b_size"},
  {HardwareDef::L0C, "l0c_size"},
  {HardwareDef::UB, "ub_size"},
  {HardwareDef::BTBUF, "btbuf_size"},
  {HardwareDef::CORENUM, "block_dim"},
  {HardwareDef::HARDWAREERR, "Hardware_unknow"}
};

const std::map<std::string, std::string> kCoreMemsizeMap = {
  {"ub_size", "UB"},
  {"hbm_size", "HBM"},
  {"l1_size", "L1"},
  {"l2_size", "L2"},
  {"l0a_size", "L0_A"},
  {"l0b_size", "L0_B"},
  {"l0c_size", "L0_C"}
};

const std::map<PipeType, std::string> kPipetypeNameMap = {
  {PipeType::AIC_MTE1, "AIC_MTE1"},
  {PipeType::AIC_MTE2, "AIC_MTE2"},
  {PipeType::AIC_FIXPIPE, "AIC_FIXPIPE"},
  {PipeType::AIC_MAC, "AIC_MAC"},
  {PipeType::AIV_MTE2, "AIV_MTE2"},
  {PipeType::AIV_MTE3, "AIV_MTE3"},
  {PipeType::AIV_VEC, "AIV_VEC"},
  {PipeType::AICORE_MTE1, "AICORE_MTE1"},
  {PipeType::AICORE_MTE2, "AICORE_MTE2"},
  {PipeType::AICORE_MTE3, "AICORE_MTE3"},
  {PipeType::AICORE_CUBE, "AICORE_CUBE"},
  {PipeType::AICORE_VEC, "AICORE_VEC"},
  {PipeType::PIPE_NONE, "PIPE_NONE"}
};

class BaseTypeUtils {
 public:
  static std::string DumpHardware(const HardwareDef hardware);
  static std::string DtypeToStr(ge::DataType dtype);
}; 
}  // namespace att

namespace ge {
void to_json(nlohmann::json &j, const Expression &arg);
}
#endif // ATT_UTILS_BASE_TYPES_PRINTER_H_

