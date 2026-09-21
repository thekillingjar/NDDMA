/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_L2_SOLVER_H_
#define ATT_L2_SOLVER_H_
#include <sstream>
#include <string>

namespace att {
inline std::string GenCeilDivision() {
  std::string strs;
  strs += "// L2的占用经验值大小\n";
  strs += "const uint32_t EMPIRIC_L2_SIZE = 128 * 1024 * 1024u;\n";
  strs += "inline uint32_t CeilDivision(uint32_t a, uint32_t b) {\n";
  strs += "  if (b == 0) {\n";
  strs += "    return 0;\n";
  strs += "  }\n";
  strs += "  return uint32_t((a + b - 1) / b);\n";
  strs += "}\n";
  strs += "\n";
  return strs;
}

inline std::string GenL2Var() {
  std::string strs;
  strs += "// 每个L2变量的信息\n";
  strs += "struct L2Var {\n";
  strs += "  // 最大值，初始化为原始输入的大小\n";
  strs += "  uint32_t max_value{0};\n";
  strs += "  // 对气值\n";
  strs += "  uint32_t align{0};\n";
  strs += "  // 对应的L0基本块的大小，举例，TileL2M对应的基本块basem\n";
  strs += "  uint32_t base_val{0};\n";
  strs += "  // 当前变量的值\n";
  strs += "  uint32_t value{0};\n";
  strs += "};\n";
  strs += "\n";
  return strs;
}

inline std::string L2TileInput() {
  std::string strs;
  strs += "// 求解器的输入\n";
  strs += "struct L2TileInput {\n";
  strs += "  // L2变量的集合\n";
  strs += "  L2Var *l2_vars{nullptr};\n";
  strs += "  // L2变量的个数\n";
  strs += "  uint32_t size{0};\n";
  strs += "  // 核数\n";
  strs += "  uint32_t core_num{0};\n";
  strs += "  // l2的大小，默认为经验值的大小\n";
  strs += "  uint32_t l2_size{0};\n";
  strs += "};\n";
  strs += "\n";
  return strs;
}

inline std::string GenL2TileSolverAnnotation() {
  std::string strs;
  strs += "// L2求解器的适用范围如下\n";
  strs += "// 举例如下，一个TileL2M * TileL2N的结果矩阵，每个小方格表示一个basem * basen的基本块\n";
  strs += "// TileL2M 和 TileL2N大小的结果矩阵存储在L2中，以基本块为粒度分多核，如下图所示，假设有4个核，每个核计算两个基本块\n";
  strs += "//                             \n";
  strs += "//                         tileL2N      basen\n";
  strs += "//             .-------.-------.-------.-------.\n";
  strs += "//             | core0 | core0 | core1 | core1 | ->basem\n";
  strs += "// tileL2M <-  '-------'-------'-------'-------'\n";
  strs += "//             | core2 | core2 | core3 | core3 |\n";
  strs += "//             '-------'-------'-------'-------'\n";
  return strs;
}

inline std::string GenDeconstructL2TileSolver() {
  std::string strs;
  strs += "  // 析构函数，用于清理堆上分配的内存\n";
  strs += "  ~L2TileSolver() {\n";
  strs += "    // 如果 blocknum_per_tile_ 指针不为空，则释放其指向的内存\n";
  strs += "    if (blocknum_per_tile_!= nullptr) {\n";
  strs += "      delete[] blocknum_per_tile_;\n";
  strs += "    }\n";
  strs += "    // 如果 size_per_tile_ 指针不为空，则释放其指向的内存\n";
  strs += "    if (size_per_tile_!= nullptr) {\n";
  strs += "      delete[] size_per_tile_;\n";
  strs += "    }\n";
  strs += "    // 如果 tilenum_ 指针不为空，则释放其指向的内存\n";
  strs += "    if (tilenum_!= nullptr) {\n";
  strs += "      delete[] tilenum_;\n";
  strs += "    }\n";
  strs += "    // 如果 total_blocknum_ 指针不为空，则释放其指向的内存\n";
  strs += "    if (total_blocknum_!= nullptr) {\n";
  strs += "      delete[] total_blocknum_;\n";
  strs += "    }\n";
  strs += "    if (input_.l2_vars != nullptr) {\n";
  strs += "      delete[] input_.l2_vars;\n";
  strs += "    }\n";
  strs += "  }\n";
  return strs;
}

inline std::string GenL2TileSolver() {
  std::string strs;
  strs += GenL2TileSolverAnnotation();
  strs += "class L2TileSolver {\n";
  strs += "public:\n";
  strs += "  // 构造函数，接受 L2TileInput 类型的参数 input\n";
  strs += "  explicit L2TileSolver(L2TileInput input) : input_(input) {};\n";
  strs += "  // 无参构造函数\n";
  strs += "  L2TileSolver() {}\n";
  strs += GenDeconstructL2TileSolver();
  strs += "  // Run() 成员函数，返回布尔值，可能用于指示某个操作的成功或失败\n";
  strs += "  bool Run();\n";
  strs += "  // GetL2Tile() 成员函数，返回 uint32_t 类型的指针\n";
  strs += "  uint32_t *GetL2Tile() { return size_per_tile_; }\n";
  strs += "\n";
  strs += "protected:\n";
  strs += "  // 纯虚函数，需要在子类中实现，用于获取 L2 的使用情况\n";
  strs += "  virtual uint64_t GetL2Use() = 0;\n";
  strs += "  virtual uint32_t GetInitVal(uint32_t L2Size) = 0;\n";
  strs += "  // 纯虚函数，需要在子类中实现，用于判断索引 idx 处是否存在冲突\n";
  strs += "  virtual bool IsClash(uint32_t idx) = 0;\n";
  strs += "  // L2TileInput 类型的成员变量，用于存储输入数据\n";
  strs += "  L2TileInput input_;\n";
  strs += "  // 默认初始化值为 1 的 used_corenum_ 成员变量\n";
  strs += "  uint32_t used_corenum_{1};\n";
  strs += "  // 指向 uint32_t 类型的指针 blocknum_per_tile_，初始化为空指针，用于表达每个方向的输入包含多少个基本块\n";
  strs += "  uint32_t *blocknum_per_tile_{nullptr};\n";
  strs += "  // 指向 uint32_t 类型的指针 size_per_tile_，初始化为空指针，用于表达每个方向的输入的大小\n";
  strs += "  uint32_t *size_per_tile_{nullptr};\n";
  strs += "  // 指向 uint32_t 类型的指针 tilenum_，初始化为空指针，用于表达每个方向的输入的L2块的个数\n";
  strs += "  uint32_t *tilenum_{nullptr};\n";
  strs += "  // 指向 uint32_t 类型的指针 total_blocknum_，初始化为空指针，用于表达每个方向输入基本块的总数\n";
  strs += "  uint32_t *total_blocknum_{nullptr};\n";
  strs += "\n";
  strs += "private:\n";
  strs += "  // 私有的 CheckInput() 成员函数，返回布尔值，用于检查输入数据的有效性\n";
  strs += "  bool CheckInput();\n";
  strs += "  // 私有的 InitInput() 成员函数，用于初始化输入数据\n";
  strs += "  void InitInput();\n";
  strs += "  // 私有的 CheckSolvable() 成员函数，返回布尔值，用于检查问题是否可解\n";
  strs += "  bool CheckSolvable();\n";
  strs += "  void HandleClash(uint32_t loop_id, uint32_t *ori_val, uint32_t *best_val, uint64_t &max_l2_use);\n";
  strs += "};\n";
  strs += "\n";
  return strs;
}

inline std::string GenL2CheckInput() {
  std::string strs;
  strs += "/**\n";
  strs += " * 检查输入参数是否有效。\n";
  strs += " *\n";
  strs += " * 该函数用于检查 L2TileSolver 类的输入参数是否有效。\n";
  strs += " * 首先检查 l2_vars 指针是否为空。如果为空，将记录一条错误消息并返回 false，表示输入无效。\n";
  strs += " * 然后检查 size、core_num 和 l2_size 参数是否都不为零。如果其中任何一个为零，将记录一条错误消息并返回 false，表示输入无效。\n";
  strs += " * 最后遍历 l2_vars 指针数组中的所有 L2Var 结构。对于每个结构，检查 align、base_val 和 max_value 成员是否都不为零。如果其中任何一个为零，将记录一条错误消息并返回 false，表示输入无效。\n";
  strs += " * 如果所有检查都通过，该函数将返回 true，表示输入有效。\n";
  strs += " *\n";
  strs += " * @Return bool, 表示输入参数是否有效\n";
  strs += " */\n";
  strs += "bool L2TileSolver::CheckInput() {\n";
  strs += "  if (input_.l2_vars == nullptr) {\n";
  strs += "    OP_LOGW(OP_NAME, \"Input l2var is null\");\n";
  strs += "    return false;\n";
  strs += "  }\n";
  strs += "  if (input_.size == 0 || input_.core_num == 0 || input_.l2_size == 0) {\n";
  strs += "    OP_LOGW(OP_NAME, \"Exist input 0, please check size, core_num and l2_size\");\n";
  strs += "    return false;\n";
  strs += "  }\n";
  strs += "  for (uint32_t i = 0; i < input_.size; i++) {\n";
  strs += "    auto var = input_.l2_vars[i];\n";
  strs += "    if (var.align == 0 || var.base_val == 0 || var.max_value == 0) {\n";
  strs += "      OP_LOGW(OP_NAME, \"Input [%u] exists 0\", i);\n";
  strs += "      return false;\n";
  strs += "    }\n";
  strs += "  }\n";
  strs += "  return true;\n";
  strs += "}\n";
  strs += "\n";
  return strs;
}

inline std::string GenL2CheckSolvable() {
  std::string strs;
  strs += "/**\n";
  strs += " * 检查问题是否可以解决。\n";
  strs += " *\n";
  strs += " * 这个函数检查当前的问题是否可以根据输入参数的设置来解决。\n";
  strs += " * 首先，它初始化每个变量的值为 align 参数的值。\n";
  strs += " * 然后，它调用 GetL2Use() 函数来获得所需的 L2 使用量。\n";
  strs += " * 如果所需的 L2 使用量超过了可用的缓存大小（input_.l2_size），它将记录一条警告消息并返回 false，表示没有解决方案。\n";
  strs += " * 如果所需的 L2 使用量小于或等于可用的缓存大小，函数将返回 true，表示问题可以被解决。\n";
  strs += " *\n";
  strs += " * @Return true 如果问题可以解决，false 否则。\n";
  strs += " */\n";
  strs += "bool L2TileSolver::CheckSolvable() {\n";
  strs += "  for (uint32_t i = 0; i < input_.size; i++) {\n";
  strs += "    auto &var = input_.l2_vars[i];\n";
  strs += "    var.value = var.align;\n";
  strs += "  }\n";
  strs += "  if (GetL2Use() > input_.l2_size) {\n";
  strs += "    OP_LOGW(OP_NAME, \"No solution, l2 size is too small\");\n";
  strs += "    return false;\n";
  strs += "  }\n";
  strs += "  return true;\n";
  strs += "}\n";
  strs += "\n";
  return strs;
}

inline std::string GetL2InitInput() {
  std::string strs;
  strs += "/**\n";
  strs += " * 初始化输入数据。\n";
  strs += " *\n";
  strs += " * 这个函数负责初始化 L2TileSolver 对象的输入数据。\n";
  strs += " * 它首先为输入数据结构中的每个变量计算最大值，将其向上取整为对齐值的最近倍数。这确保了每个变量的最大值是其对齐值的倍数。\n";
  strs += " * 然后，它找出所有变量中的最大最大值，并将每个变量的值初始化为这个最大最大值。\n";
  strs += " *\n";
  strs += " * 初始化过程对于准备输入数据以进行后续处理步骤（如优化或分析）至关重要。\n";
  strs += " * 确保最大值是对齐值的倍数，可以简化数据的处理，并可能是依赖于这种属性的算法或操作所必需的。\n";
  strs += " */\n";
  strs += "void L2TileSolver::InitInput() {\n";
  strs += "  uint32_t init_value = 0;\n";
  strs += "  for (uint32_t i = 0; i < input_.size; i++) {\n";
  strs += "    auto &var = input_.l2_vars[i];\n";
  strs += "    var.max_value = CeilDivision(var.max_value, var.align) * var.align;\n";
  strs += "  }\n";
  strs += "}\n";
  strs += "\n";
  return strs;
}

inline std::string GetHandleClash() {
  std::string strs;
  strs += "void L2TileSolver::HandleClash(uint32_t loop_id, uint32_t *ori_val, uint32_t *best_val, uint64_t &max_l2_use) {\n";
  strs += "  auto max_blocknum = ori_val[loop_id];\n";
  strs += "  auto &var = input_.l2_vars[loop_id];\n";
  strs += "  for (uint32_t i = max_blocknum; i >= 1u; i--) {\n";
  strs += "    blocknum_per_tile_[loop_id] = i;\n";
  strs += "    size_per_tile_[loop_id] = blocknum_per_tile_[loop_id] * var.base_val;\n";
  strs += "    tilenum_[loop_id] = CeilDivision(var.max_value, size_per_tile_[loop_id]);\n";
  strs += "    var.value = size_per_tile_[loop_id];\n";
  strs += "    if (loop_id == input_.size-1) {\n";
  strs += "      uint32_t tmp_corenum = 1;\n";
  strs += "      for (uint32_t j = 0; j < input_.size; j++) {\n";
  strs += "        tmp_corenum *= blocknum_per_tile_[j];\n";
  strs += "      }\n";
  strs += "      used_corenum_ = std::min(input_.core_num, tmp_corenum);\n";
  strs += "      bool solved=true;\n";
  strs += "      for (uint32_t k = 0; k < input_.size; k++) {\n";
  strs += "        if (IsClash(k)) {\n";
  strs += "          solved=false;\n";
  strs += "        }\n";
  strs += "      }\n";
  strs += "      if (solved) {\n";
  strs += "        uint64_t l2_use = GetL2Use();\n";
  strs += "        if (l2_use > max_l2_use) {\n";
  strs += "          for (uint32_t l = 0; l < input_.size; l++) {\n";
  strs += "            best_val[l] = blocknum_per_tile_[l];\n";
  strs += "          }\n";
  strs += "          max_l2_use = l2_use;\n";
  strs += "        }\n";
  strs += "        return;\n";
  strs += "      }\n";
  strs += "    } else {\n";
  strs += "      HandleClash(loop_id+1, ori_val, best_val, max_l2_use);\n";
  strs += "    }\n";
  strs += "  }\n";
  strs += "}\n";
  strs += "\n";
  return strs;
}

inline std::string GetL2RunAnnotation() {
  std::string strs;
  strs += "/**\n";
  strs += " * 运行 L2TileSolver 算法来解决 L2 缓存分块问题。\n";
  strs += " *\n";
  strs += " * 这个函数是 L2TileSolver 算法的核心。它尝试根据输入参数和约束条件找到 L2 缓存的最优分块方案。\n";
  strs += " * 函数首先通过 CheckInput() 函数检查输入参数的有效性。如果输入无效，它将记录一条错误消息并返回 false。\n";
  strs += " * 然后通过 CheckSolvable() 函数检查问题是否有解。如果没有解，它将记录一条错误消息并返回 false。\n";
  strs += " * 如果输入有效并且问题有解，函数将通过 InitInput() 函数初始化输入数据。\n";
  strs += " *\n";
  strs += " * 算法的核心是一个循环，在这个循环中，它不断地调整每个变量的值，以找到一个合适的分块方案。\n";
  strs += " * 循环结束的条件是总内存使用量小于或等于 L2 缓存的大小。\n";
  strs += " * 循环结束后，它将计算每个变量的每个分块的大小、总块数、分块数和占用的核心数。\n";
  strs += " * 然后，它检查是否存在读冲突。如果存在读冲突，它将调整每个分块的大小，直到不再检测到冲突。\n";
  strs += " * 最后，它检查每个变量的最大值是否可以放在一个分块中。\n";
  strs += " * 如果不能，它将相应地调整分块数和每个分块的大小。\n";
  strs += " *\n";
  strs += " * 如果找到合适的分块方案，函数返回 true，否则返回 false。\n";
  strs += " *\n";
  strs += " * @Return 如果成功则返回 true，否则返回 false\n";
  strs += " */\n";
  return strs;
}

inline std::string GetL2Run() {
  std::string strs;
  strs += GetL2RunAnnotation();
  strs += "bool L2TileSolver::Run() {\n";
  strs += "  if (!CheckInput()) {\n";
  strs += "    OP_LOGW(OP_NAME, \"Check input failed\");\n";
  strs += "    return false;\n";
  strs += "  }\n";
  strs += "  if (!CheckSolvable()) {\n";
  strs += "    OP_LOGW(OP_NAME, \"Check Solvable failed\");\n";
  strs += "    return false;\n";
  strs += "  }\n";
  strs += "  InitInput();\n";
  strs += "  uint32_t core_num = input_.core_num;\n";
  strs += "  uint32_t l2_size = input_.l2_size;\n";
  strs += "  blocknum_per_tile_ = new(std::nothrow) uint32_t[input_.size];\n";
  strs += "  size_per_tile_ = new(std::nothrow) uint32_t[input_.size];\n";
  strs += "  tilenum_ = new(std::nothrow) uint32_t[input_.size];\n";
  strs += "  total_blocknum_ = new(std::nothrow) uint32_t[input_.size];\n";
  strs += "  for (uint32_t i=0u; i < input_.size; i++) {\n";
  strs += "    auto &var = input_.l2_vars[i];\n";
  strs += "    auto val = GetInitVal(input_.l2_size);\n";
  strs += "    var.value = std::min(var.max_value, CeilDivision(val, var.align) * var.align);\n";
  strs += "  }\n";
  strs += "  // 遍历直到满足L2占用停止\n";
  strs += "  while (GetL2Use() > l2_size) {\n";
  strs += "    for (uint32_t i = 0; i < input_.size; i++) {\n";
  strs += "      auto &var = input_.l2_vars[i];\n";
  strs += "      var.value = (var.align < var.value) ? (var.value - var.align) : var.align;\n";
  strs += "    }\n";
  strs += "  }\n";
  strs += "  for (uint32_t i = 0; i < input_.size; i++) {\n";
  strs += "    auto &var = input_.l2_vars[i];\n";
  strs += "    blocknum_per_tile_[i] = CeilDivision(var.value, var.base_val);\n";
  strs += "    size_per_tile_[i] = blocknum_per_tile_[i] * var.base_val;\n";
  strs += "    tilenum_[i] = CeilDivision(var.max_value, size_per_tile_[i]);\n";
  strs += "    total_blocknum_[i] = CeilDivision(var.max_value, var.base_val);\n";
  strs += "  }\n";
  strs += "  for (uint32_t i = 0; i < input_.size; i++) {\n";
  strs += "    auto &var = input_.l2_vars[i];\n";
  strs += "    if (var.max_value <= size_per_tile_[i]) {\n";
  strs += "      tilenum_[i] = 1;\n";
  strs += "      blocknum_per_tile_[i] = CeilDivision(var.max_value, var.base_val);\n";
  strs += "      size_per_tile_[i] = blocknum_per_tile_[i] * var.base_val;\n";
  strs += "    }\n";
  strs += "  }\n";
  strs += "  return true;\n";
  strs += "}\n";
  return strs;
}

inline std::string GetL2SolverHead() {
  std::string strs;
  strs += GenCeilDivision();
  strs += GenL2Var();
  strs += L2TileInput();
  strs += GenL2TileSolver();
  return strs;
}

inline std::string GetL2SolverFunc() {
  std::string strs;
  strs += GenL2CheckInput();
  strs += GenL2CheckSolvable();
  strs += GetL2InitInput();
  strs += GetHandleClash();
  strs += GetL2Run();
  return strs;
}

inline const std::string L2_SOLVER_CODE_HEAD = GetL2SolverHead();
inline const std::string L2_SOLVER_CODE_FUNC = GetL2SolverFunc();
} // namespace att
#endif