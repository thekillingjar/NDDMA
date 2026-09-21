/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "l2_solver_gen.h"

namespace att
{
  bool L2TileSolverGen::IsRepeatArgs(const Expr &arg) const
  {
    if (!IsValid(arg)) {
      return false;
    }
    if (l2_use_.ContainVar(arg))
    {
      return true;
    }
    return false;
  }

  bool L2TileSolverGen::IsClashPossible() const
  {
    std::vector<Expr> input_and_const_args;
    input_and_const_args.insert(input_and_const_args.end(), input_args_.begin(), input_args_.end());
    for (auto &arg : const_vars_) {
      input_and_const_args.emplace_back(arg.first);
    }
    for (size_t i = 0u; i < input_and_const_args.size(); i++)
    {
      bool has_repeat_arg = true;
      for (auto &l2_arg : l2_args_)
      {
        if (!IsValid(l2_arg)) {
          return false;
        }
        if (input_and_const_args[i] == arg_max_value_map_.at(l2_arg))
        {
          has_repeat_arg = false;
        }
      }
      if (has_repeat_arg)
      {
        if (IsRepeatArgs(input_and_const_args[i]))
        {
          return true;
        }
      }
    }
    return false;
  }
  
  std::string L2TileSolverGen::GetL2RelInputArg() {
    std::string l2_rel_input_arg = "";
    for (const auto &arg : input_args_)
    {
      if (IsRepeatArgs(arg)) {
        l2_rel_input_arg = Str(arg);
      }
    }
    return l2_rel_input_arg;
  }

  std::string L2TileSolverGen::GenClassDef() {
    std::stringstream ss;
    ss << AddAnotationLine("根据tiling case id创建对应的L2TileSolver子类\n");
    ss << "class " << tiling_case_id_ << "L2TileSolver : public L2TileSolver {\n";
    ss << "  public:\n";
    auto l2_rel_input_arg = GetL2RelInputArg();
    ss << AddAnotationLine("构造函数，接受L2TileInput类型的参数 input\n", "    ");  
    ss << "    explicit " << tiling_case_id_ << "L2TileSolver(L2TileInput &input, " + type_name_ + " &tiling_data) : L2TileSolver(input) {\n";
    if (l2_rel_input_arg != "") {
      ss << "      " << l2_rel_input_arg << " =  tiling_data.get_" << l2_rel_input_arg << "();\n";
    }
    ss << "    };\n";
    ss << AddAnotationLine("成员函数声明\n", "    ");
    ss << "    uint64_t GetL2Use() override;\n";
    ss << "    bool IsClash(const uint32_t idx) override;\n";
    ss << "    uint32_t GetInitVal(uint32_t L2Size) override;\n";
    for (auto const_var : const_vars_) {
      ss << "    uint32_t " << const_var.first << "{" << const_var.second << "};\n";
    }
    if (l2_rel_input_arg != "") {
      ss << "    uint64_t " << l2_rel_input_arg << ";\n";
    }
    ss << "};\n";
    std::string strs = "";
    strs += "对创建的L2TileSolver子类的成员函数GetL2Use进行定义，计算L2的可用内存大小\n";
    strs += "用input_存储输入数据，并将输入数据对应赋值给tilem_size, tilen_size, k_size\n";
    strs += "使用公式计算L2的可用内存大小l2_size\n";
    strs += "@return 返回L2可用内存的大小l2_size\n";
    ss << AddAnotationBlock(strs);
    ss << "uint64_t " << tiling_case_id_ << "L2TileSolver::GetL2Use() {\n";
    for (size_t i = 0u; i < l2_args_.size(); i++) {
      ss << "  uint64_t " << l2_args_[i] << " = input_.l2_vars[" << std::to_string(i) << "].value;\n";
    }
    ss << "  uint64_t l2_size = " << l2_use_ << ";\n";
    ss << "  return l2_size;\n";
    ss << "}\n";
    ss << "uint32_t " << tiling_case_id_ << "L2TileSolver::GetInitVal(uint32_t L2Size) {\n";
    if (l2_rel_input_arg != "") {
      ss << "  uint64_t kSquare = " << l2_rel_input_arg << " * " << l2_rel_input_arg << ";\n"; 
      ss << "  return (uint32_t)((std::sqrt(16 * kSquare + 8 * L2Size) - 4 * " << l2_rel_input_arg << ") / 4);\n";
    } else {
      ss << "  uint32_t max_val = 0u;\n";
      ss << "  for (uint32_t i=0u; i < input_.size; i++) {\n";
      ss << "    auto &var = input_.l2_vars[i];\n";
      ss << "    max_val = var.max_value > max_val ? var.max_value : max_val;\n";
      ss << "  }\n";
      ss << " return max_val;\n";
    }
    ss << "}\n";
    return ss.str();
  }

  std::string L2TileSolverGen::GenSolverClassImpl()
  {
    std::stringstream ss;
    std::string strs;
    ss << GenClassDef();
    strs = "";
    strs += "对创建的L2TileSolver子类的成员函数IsClash进行定义，用于检测是否存在读冲突\n";
    strs += "@param idx 输入参数为idx，用于获取当前tile的编号\n";
    strs += "@return 如果满足条件blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0,则存在读冲突，返回true\n";
    strs += "用公式计算尾块个数blocknum_tail\n";
    strs += "@return 如果满足条件blocknum_tail % (used_corenum_ / 2) == 0,则存在读冲突，返回true\n";
    strs += "@return 如果以上两个条件均不满足，说明不存在读冲突，返回false\n";
    ss << AddAnotationBlock(strs);

    ss << "bool " << tiling_case_id_ << "L2TileSolver::IsClash(const uint32_t idx) {\n";
    if (!IsClashPossible())
    {
      ss << "  return false;\n";
    }
    else
    {
      ss << "  if ((used_corenum_ <= 1) || (used_corenum_ % 2 !=0)) {\n";
      ss << "    return false;\n  }\n";
      ss << "  if (blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0) {\n";
      ss << "    return true;\n";
      ss << "  }\n";
      ss << "  auto blocknum_tail = total_blocknum_[idx] - (tilenum_[idx] "
            "- 1) * blocknum_per_tile_[idx];\n";
      ss << "  if (blocknum_tail % (used_corenum_ / 2) == 0) {\n";
      ss << "    return true;\n";
      ss << "  }\n";
      ss << "  return false;\n";
    }
    ss << "}\n";
    return ss.str();
  }

  Expr L2TileSolverGen::GetRelateL0Arg(const Expr &l2_arg)
  {
    Expr res;
    auto iter = arg_max_value_map_.find(l2_arg);
    GE_ASSERT_TRUE(iter != arg_max_value_map_.end(), "Ori arg map does not find l2 arg[%s]", l2_arg.Str().get());
    Expr ori_l2_arg = iter->second;
    for (auto l0_arg : l0_args_) {
      GE_ASSERT_TRUE(arg_max_value_map_.find(l0_arg) != arg_max_value_map_.end(),
                     "Ori arg map does not find l0 arg[%s]", l0_arg.Str().get());
      if (arg_max_value_map_[l0_arg] == ori_l2_arg) {
        return l0_arg;
      }
    }
    return res;
  }

  std::string L2TileSolverGen::GenSolverFuncInvoke()
  {
    std::string strs = "";
    strs += "    if (!ExecuteL2Solver(tiling_data)) {\n";
    strs += "        return false;\n";
    strs += "    }\n";
    return strs;
  }

  std::string L2TileSolverGen::GenSolverInvokeDoc() const {
    std::string strs = "";
    strs += "定义L2求解器调用函数\n";
    strs += "@param tiling_data 输入参数为tiling_data\n";
    strs += "定义求解器的输入为L2TileInput类型结构体变量，并将结构体的成员变量赋值为对应值\n";
    strs += "定义tilem_size为L2Var类型的结构体变量，并将相关成员变量赋值为对应值\n";
    strs += "定义tilen_size为L2Var类型的结构体变量，并将相关成员变量赋值为对应值\n";
    strs += "定义l2_solver为L2TileSolver类型的结构体变量\n";
    strs += "计算l2的可用内存大小以及是否存在读冲突\n";
    strs += "@return 执行L2TileSolver类的主要流程l2_solver.run()，如果l2_solver.run()的返回值不为true,则l2求解器执行失败，函数返回false\n";
    strs += "用output指针指向l2_solver的输出\n";
    strs += "i从0开始到L2相关变量个数依次递增\n";
    strs += "@return 如果output为空指针或者i>=L2相关的变量个数，则函数返回false，否则将tilem_size设置为output的第i个值\n";
    strs += "@return 若上述没有触发返回false的场景，则函数返回true\n";
    return AddAnotationBlock(strs);
  }

  std::string L2TileSolverGen::GenSolverFuncImpl() {
    std::stringstream ss;
    ss << GenSolverInvokeDoc();
    ss << "  bool ExecuteL2Solver(" + type_name_ + "& tiling_data) {\n";
    ss << "    L2TileInput l2_input;\n";
    ss << "    l2_input.l2_vars = new(std::nothrow) L2Var[" << std::to_string(l2_args_.size()) << "];\n";
    ss << "    l2_input.size = " << std::to_string(l2_args_.size()) << ";\n";
    ss << "    l2_input.core_num = corenum_;\n";
    ss << "    l2_input.l2_size = EMPIRIC_L2_SIZE;\n";
    for (size_t i = 0u; i < l2_args_.size(); i++) {
      auto l2_arg = l2_args_[i];
      ss << "    L2Var " << l2_arg << ";\n";
      ss << "    " << l2_arg << ".max_value = tiling_data.get_" << arg_max_value_map_[l2_arg] << "();\n";
      if (arg_align_map_.find(l2_arg) == arg_align_map_.end()) {
        GELOGE(ge::FAILED, "Arg align map does not find l2 arg[%s]", l2_arg.Str().get());
        return kSolverGenError;
      }
      ss << "    " << l2_arg << ".align = " << Str(arg_align_map_[l2_arg]) << ";\n";
      auto relate_l0_arg = GetRelateL0Arg(l2_arg);
      GE_ASSERT_TRUE(IsValid(relate_l0_arg), "Get relate l0 arg failed");
      ss << "    " << l2_arg << ".base_val = tiling_data.get_" << relate_l0_arg << "();\n";
      ss << "    l2_input.l2_vars[" << std::to_string(i) << "] = " << l2_arg << ";\n";
    }
    ss << "    " << tiling_case_id_ << "L2TileSolver l2_solver(l2_input, tiling_data);\n";
    ss << "    if (!l2_solver.Run()) {\n";
    ss << "      OP_LOGW(OP_NAME, \"l2 solver run failed\");\n";
    ss << "      return false;\n";
    ss << "    }\n";
    ss << "    uint32_t *output = l2_solver.GetL2Tile();\n";
    for (size_t i = 0u; i < l2_args_.size(); i++) {
      auto l2_arg = l2_args_[i];
      ss << "    if ((output == nullptr) || (" << i << " >= " << l2_args_.size() << ")) {\n";
      ss << "      OP_LOGW(OP_NAME, \"l2_vars is nullptr or overflow\");\n";
      ss << "      return false;\n";
      ss << "    }\n";
      ss << "    tiling_data.set_" << l2_arg << "(output[" << std::to_string(i) << "]);\n";
    }
    ss << "    return true;\n";
    ss << "  }\n";
    return ss.str();
  }
} // namespace att
