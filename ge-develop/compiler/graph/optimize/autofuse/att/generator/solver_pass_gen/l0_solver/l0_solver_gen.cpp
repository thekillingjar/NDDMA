/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "l0_solver_gen.h"
#include "ge_common/ge_api_error_codes.h"
#include "ge_common/debug/log.h"

namespace att {
bool L0TileSolverGen::CheckIsInnerMost(const Expr &arg) {
  for (const auto &innermost_arg : innermost_args_) {
    if (innermost_arg == arg) {
      return true;
    }
  }
  return false;
}

std::string L0TileSolverGen::GenClassDef() {
  std::stringstream ss;
  ss << AddAnotationLine("根据tiling case id创建对应的L0TileSolver子类\n");
  ss << "class " << tiling_case_id_ << "L0TileSolver : public L0TileSolver {\n";
  ss << "  public:\n";
  ss << AddAnotationLine("构造函数，接受L0TileInput类型的参数 input\n", "    ");
  ss << "    explicit " << tiling_case_id_ << "L0TileSolver(L0TileInput &input) : L0TileSolver(input) {};\n";
  ss << AddAnotationLine("成员函数声明\n", "    ");
  for (auto buffer_use : buffer_use_map_) {
    auto hardware = buffer_use.first;
    ss << "    void Set" << BaseTypeUtils::DumpHardware(hardware) << "(const uint32_t &value) { "
       << BaseTypeUtils::DumpHardware(hardware) << "_ = value; }\n";
  }
  ss << "    bool CheckBufferUseValid() override;\n";
  ss << AddAnotationLine("定义private类型的成员变量\n", "  ");
  ss << "  private:\n";
  for (auto buffer_use : buffer_use_map_) {
    auto hardware = buffer_use.first;
    ss << "    uint32_t " << BaseTypeUtils::DumpHardware(hardware) << "_;\n";
  }
  for (auto const_var : const_vars_) {
    if (!IsValid(const_var.first)) {
      continue;
    }
    ss << "    uint32_t " << const_var.first << "{" << const_var.second << "};\n";
  }
  ss << "};\n";
  return ss.str();
}

std::string L0TileSolverGen::GenSolverClassImpl() {
  std::stringstream ss;
  std::string strs;
  ss << GenClassDef();
  strs = "";
  strs += "  对创建的L0TileSolver子类的成员函数CheckBufferUseValid进行定义，判断缓存占用是否合法\n";
  strs += "  i从0开始到待求解l0相关变量的数量，依次递增循环\n";
  strs += "  @return 如果代求解l0变量的数量为空或i>=待求解l0相关变量的数量，返回false。如果在DEBUG域内输出提示信息\n";
  strs += "  定义uint32_t类型变量：第i个待求解l0变量的字符串名称，值为第i个待求解l0相关变量的值\n";
  ss << AddAnotationBlock(strs, "  ");

  ss << "bool " << tiling_case_id_ << "L0TileSolver::CheckBufferUseValid() {\n";
  for (size_t i = 0u; i < l0_args_.size(); i++) {
    ss << "  if ((input_.l0_vars == nullptr) || (" << i << " >= " << l0_args_.size() << ")) {\n";
    ss << "      OP_LOGW(OP_NAME, \"l0_vars is nullptr or overflow\");\n";
    ss << "      return false;\n"
       << "   }\n";
    ss << "  uint32_t " << l0_args_[i] << " = input_.l0_vars[" << std::to_string(i) << "].value;\n";
  }

  strs = "";
  strs += "  遍历buffer_use_map_中的所有缓存占用表达式，将其转换为字符串类型，赋值给对应的设备类型\n";
  strs += "  将每种设备类型缓存占用计算值和其设定值对比，如果缓存占用计算值大于设定值，返回false\n";
  strs += "  @return 遍历所有设备类型后，如果所有设备类型缓存占用计算值都小于设定值，返回true\n";
  ss << AddAnotationBlock(strs, "  ");

  for (auto buffer_use : buffer_use_map_) {
    auto hardware = buffer_use.first;
    auto expr = buffer_use.second;
    if ((hardware != HardwareDef::L0A) && (hardware != HardwareDef::L0B) && (hardware != HardwareDef::L0C)) {
      continue;
    }
    ss << "  uint32_t " << BaseTypeUtils::DumpHardware(hardware) << " =  " << expr << ";\n";
    ss << "  if (" << BaseTypeUtils::DumpHardware(hardware) << " > " << BaseTypeUtils::DumpHardware(hardware)
       << "_) {\n";
    ss << "    return false;\n"
       << "   };\n";
  }
  ss << "  return true;\n"
     << "}\n";
  return ss.str();
}

ge::Status L0TileSolverGen::GetLargestAlign(const Expr &arg, Expr &max_align) {
  auto iter = father_args_map_.find(arg);
  if (iter != father_args_map_.end()) {
    Expr father_arg = iter->second;
    auto arg_align_iter = arg_align_map_.find(father_arg);
    if (arg_align_iter == arg_align_map_.end()) {
      GELOGE(ge::FAILED, "arg align map does not find arg [%s]", father_arg.Str().get());
      max_align = ge::Symbol(0U);
      return ge::FAILED;
    }
    Expr align = arg_align_iter->second;
    max_align = ge::sym::Max(max_align, align);
    GetLargestAlign(father_arg, max_align);
  }
  return ge::SUCCESS;
}

bool L0TileSolverGen::IsMulticoreArg(const Expr &arg) {
  if (!IsValid(arg)) {
    return false;
  }
  for (auto mc_arg : mc_args_) {
    if (mc_arg == arg) {
      return true;
    }
  }
  return false;
}

bool L0TileSolverGen::IsBindMulticore(const Expr &arg) {
  if (IsMulticoreArg(arg)) {
    return true;
  }
  auto iter = father_args_map_.find(arg);
  if (iter == father_args_map_.end()) {
    return false;
  }
  return IsBindMulticore(iter->second);
}

std::string L0TileSolverGen::GenSolverFuncInvoke() {
  std::string strs = "";
  strs += "    if (!ExecuteL0Solver(tiling_data)) {\n";
  strs += "        return false;\n";
  strs += "    }\n";
  return strs;
}

std::string L0TileSolverGen::GenSolverInvokeDoc() const {
  std::string strs = "";
  strs += "定义L0求解器调用函数\n";
  strs += "@param tiling_data 输入参数为tiling_data\n";
  strs += "定义求解器接受的输入为L0TileInput类型结构体变量，并将结构体的成员变量赋值为对应值\n";
  strs += "定义basem_size为L0Var类型的结构体变量，并将相关成员变量赋值为对应值\n";
  strs += "定义basen_size为L0Var类型的结构体变量，并将相关成员变量赋值为对应值\n";
  strs +=
      "定义solver为case0L0TileSolver类型的结构体变量，并将此结构体变量的成员变量L1_, L2_, L0A_, L0B_, L0C_, "
      "CORENUM_赋值为tiling_data中的设定值\n";
  strs +=
      "@return "
      "执行L0TileSolver类的主要流程solver.run()，如果solver.run()的返回值不为true,则l0求解器执行失败，函数返回false\n";
  strs += "用output指针指向solver的输出\n";
  strs += "i从0开始到L0相关变量个数依次递增\n";
  strs += "@return 如果output为空指针或者i>=L0相关的变量个数，则函数返回false，否则将basem_size设置为output的第i个值\n";
  strs += "@return 若上述没有触发返回false的场景，则函数返回true\n";
  return AddAnotationBlock(strs);
}

std::string L0TileSolverGen::GenInitTilingData() {
  std::stringstream ss;
  ss << "  bool ExecuteL0Solver(" + type_name_ + "& tiling_data) {\n";
  ss << "    L0TileInput l0_input;\n";
  ss << "    l0_input.l0_vars = new(std::nothrow) L0Var[" << std::to_string(l0_args_.size()) << "];\n";
  ss << "    l0_input.size = " << std::to_string(l0_args_.size()) << ";\n";
  ss << "    l0_input.core_num = corenum_;\n";
  for (size_t i = 0u; i < l0_args_.size(); i++) {
    auto l0_arg = l0_args_[i];
    ss << "    L0Var " << l0_arg << ";\n";
    if (arg_max_value_map_.find(l0_arg) == arg_max_value_map_.end()) {
      GELOGE(ge::FAILED, "Ori arg map does not find l0 arg [%s]", l0_arg.Str().get());
      return kSolverGenError;
    }
    ss << "    " << l0_arg << ".max_value = tiling_data.get_" << arg_max_value_map_[l0_arg] << "();\n";
    std::string true_or_false = IsBindMulticore(l0_arg) ? "true" : "false";
    ss << "    " << l0_arg << ".bind_multicore = " << true_or_false << ";\n";
    ss << "    " << l0_arg << ".align = " << Str(arg_align_map_[l0_arg]) << ";\n";
    if (arg_align_map_.find(l0_arg) == arg_align_map_.end()) {
      GELOGE(ge::FAILED, "Arg align map does not find l0 arg [%s]", l0_arg.Str().get());
      return kSolverGenError;
    }
    Expr max_align = arg_align_map_[l0_arg];
    if (max_align.IsConstExpr() && (ge::SymbolicUtils::StaticCheckEq(max_align, ge::Symbol(0)) == ge::TriBool::kTrue)) {
      GELOGE(ge::FAILED, "l0 arg [%s] align is 0", l0_arg.Str().get());
      return kSolverGenError;
    }
    GetLargestAlign(l0_arg, max_align);
    if (max_align.IsConstExpr() && (ge::SymbolicUtils::StaticCheckEq(max_align, ge::Symbol(0)) == ge::TriBool::kTrue)) {
      GELOGE(ge::FAILED, "Get largest align failed");
      return kSolverGenError;
    }
    if (CheckIsInnerMost(l0_arg)) {
      ss << "    " << l0_arg << ".is_innermost = true;\n";
      ss << "    " << l0_arg << ".value = 256;\n";
    } else if (max_align.IsConstExpr() &&
               (ge::SymbolicUtils::StaticCheckGt(max_align, arg_align_map_[l0_arg]) == ge::TriBool::kTrue)) {
      ss << "    " << l0_arg << ".value = 64;\n";
    } else {
      ss << "    " << l0_arg << ".value = 128;\n";
    }
    ss << "    " << l0_arg << ".prompt_align = " << Str(max_align) << ";\n";
    ss << "    l0_input.l0_vars[" << std::to_string(i) << "] = " << l0_arg << ";\n";
  }
  return ss.str();
}

std::string L0TileSolverGen::GenSolverFuncImpl() {
  std::stringstream ss;
  ss << GenSolverInvokeDoc();
  ss << GenInitTilingData();
  ss << "    " << tiling_case_id_ << "L0TileSolver solver(l0_input);\n";
  for (auto buffer_use : buffer_use_map_) {
    auto hardware = buffer_use.first;
    auto expr = buffer_use.second;
    ss << "    solver.Set" << BaseTypeUtils::DumpHardware(hardware) << "(tiling_data.get_"
       << BaseTypeUtils::DumpHardware(hardware) << "());\n";
  }
  ss << "    if (!solver.Run()) {\n";
  ss << "      OP_LOGW(OP_NAME, \"l0 solver run failed\");\n";
  ss << "      return false;\n";
  ss << "    }\n";
  ss << "    uint32_t *output = solver.GetOutput();\n";
  for (size_t i = 0u; i < l0_args_.size(); i++) {
    auto l0_arg = l0_args_[i];
    if (i >= l0_args_.size()) {
      ss << "    OP_LOGW(OP_NAME, \"output overflows\");\n";
      ss << "    return false;\n";
    } else {
      ss << "    if (output == nullptr) {\n";
      ss << "      OP_LOGW(OP_NAME, \"output is nullptr\");\n";
      ss << "      return false;\n";
      ss << "    }\n";
    }
    ss << "    tiling_data.set_" << l0_arg << "(output[" << std::to_string(i) << "]);\n";
  }
  ss << "    return true;\n";
  ss << "  }\n";
  return ss.str();
}
}  // namespace att
