/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#ifdef DEBUG
#define ATT_LOG(log)
do {
  std::cout << "[ERROR]" << log << std::endl;
} while (0)
#else
#define ATT_LOG(log)
#endif

namespace att {
// L2的占用经验值大小
const uint32_t EMPIRIC_L2_SIZE = 128 * 1024 * 1024u;
uint32_t CeilDivision(uint32_t a, uint32_t b) {
  if (b == 0) {
    return 0;
  }
  return uint32_t((a + b - 1) / b);
}

// 每个L2变量的信息
struct L2Var {
  // 最大值，初始化为原始输入的大小
  uint32_t max_value{0};
  // 对气值
  uint32_t align{0};
  // 对应的L0基本块的大小，举例，TileL2M对应的基本块basem
  uint32_t base_val{0};
  // 当前变量的值
  uint32_t value{0};
};

// 求解器的输入
struct L2TileInput {
  // L2变量的集合
  L2Var *l2_vars{nullptr};
  // L2变量的个数
  uint32_t size{0};
  // 核数
  uint32_t core_num{0};
  // l2的大小，默认为经验值的大小
  uint32_t l2_size{0};
};

// L2求解器的适用范围如下
// 举例如下，一个TileL2M * TileL2N的结果矩阵，每个小方格表示一个basem * basen的基本块
// TileL2M 和 TileL2N大小的结果矩阵存储在L2中，以基本块为粒度分多核，如下图所示，假设有4个核，每个核计算两个基本块
//                             
//                         tileL2N      basen
//             .-------.-------.-------.-------.
//             | core0 | core0 | core1 | core1 | ->basem
// tileL2M <-  '-------'-------'-------'-------'
//             | core2 | core2 | core3 | core3 |
//             '-------'-------'-------'-------'
class L2TileSolver {
public:
  // 构造函数，接受 L2TileInput 类型的参数 input
  explicit L2TileSolver(L2TileInput input) : input_(input) {};
  // 无参构造函数
  L2TileSolver() {}
  // 析构函数，用于清理堆上分配的内存
  ~L2TileSolver() {
    // 如果 blocknum_per_tile_ 指针不为空，则释放其指向的内存
    if (blocknum_per_tile_!= nullptr) {
      delete[] blocknum_per_tile_;
    }
    // 如果 size_per_tile_ 指针不为空，则释放其指向的内存
    if (size_per_tile_!= nullptr) {
      delete[] size_per_tile_;
    }
    // 如果 tilenum_ 指针不为空，则释放其指向的内存
    if (tilenum_!= nullptr) {
      delete[] tilenum_;
    }
    // 如果 total_blocknum_ 指针不为空，则释放其指向的内存
    if (total_blocknum_!= nullptr) {
      delete[] total_blocknum_;
    }
  }
  // Run() 成员函数，返回布尔值，可能用于指示某个操作的成功或失败
  bool Run();
  // GetL2Tile() 成员函数，返回 uint32_t 类型的指针
  uint32_t *GetL2Tile() { return size_per_tile_; }

protected:
  // 纯虚函数，需要在子类中实现，用于获取 L2 的使用情况
  virtual uint64_t GetL2Use() = 0;
  // 纯虚函数，需要在子类中实现，用于判断索引 idx 处是否存在冲突
  virtual bool IsClash(uint32_t idx) = 0;
  // L2TileInput 类型的成员变量，用于存储输入数据
  L2TileInput input_;
  // 默认初始化值为 1 的 used_corenum_ 成员变量
  uint32_t used_corenum_{1};
  // 指向 uint32_t 类型的指针 blocknum_per_tile_，初始化为空指针，用于表达每个方向的输入包含多少个基本块
  uint32_t *blocknum_per_tile_{nullptr};
  // 指向 uint32_t 类型的指针 size_per_tile_，初始化为空指针，用于表达每个方向的输入的大小
  uint32_t *size_per_tile_{nullptr};
  // 指向 uint32_t 类型的指针 tilenum_，初始化为空指针，用于表达每个方向的输入的L2块的个数
  uint32_t *tilenum_{nullptr};
  // 指向 uint32_t 类型的指针 total_blocknum_，初始化为空指针，用于表达每个方向输入基本块的总数
  uint32_t *total_blocknum_{nullptr};

private:
  // 私有的 CheckInput() 成员函数，返回布尔值，用于检查输入数据的有效性
  bool CheckInput();
  // 私有的 InitInput() 成员函数，用于初始化输入数据
  void InitInput();
  // 私有的 CheckSolvable() 成员函数，返回布尔值，用于检查问题是否可解
  bool CheckSolvable();
  void HandleClash(uint32_t loop_id, uint32_t *ori_val, uint32_t *best_val, uint64_t &max_l2_use);
};

/**
 * 检查输入参数是否有效。
 *
 * 该函数用于检查 L2TileSolver 类的输入参数是否有效。
 * 首先检查 l2_vars 指针是否为空。如果为空，将记录一条错误消息并返回 false，表示输入无效。
 * 然后检查 size、core_num 和 l2_size 参数是否都不为零。如果其中任何一个为零，将记录一条错误消息并返回 false，表示输入无效。
 * 最后遍历 l2_vars 指针数组中的所有 L2Var 结构。对于每个结构，检查 align、base_val 和 max_value 成员是否都不为零。如果其中任何一个为零，将记录一条错误消息并返回 false，表示输入无效。
 * 如果所有检查都通过，该函数将返回 true，表示输入有效。
 *
 * @Return bool, 表示输入参数是否有效
 */
bool L2TileSolver::CheckInput() {
  if (input_.l2_vars == nullptr) {
    ATT_LOG("Input l2var is null");
    return false;
  }
  if (input_.size == 0 || input_.core_num == 0 || input_.l2_size == 0) {
    ATT_LOG("Exist input 0, please check size, core_num and l2_size");
    return false;
  }
  for (uint32_t i = 0; i < input_.size; i++) {
    auto var = input_.l2_vars[i];
    if (var.align == 0 || var.base_val == 0 || var.max_value == 0) {
      ATT_LOG("Input [" + std::to_string(i) + "] exists 0");
      return false;
    }
  }
  return true;
}

/**
 * 检查问题是否可以解决。
 *
 * 这个函数检查当前的问题是否可以根据输入参数的设置来解决。
 * 首先，它初始化每个变量的值为 align 参数的值。
 * 然后，它调用 GetL2Use() 函数来获得所需的 L2 使用量。
 * 如果所需的 L2 使用量超过了可用的缓存大小（input_.l2_size），它将记录一条警告消息并返回 false，表示没有解决方案。
 * 如果所需的 L2 使用量小于或等于可用的缓存大小，函数将返回 true，表示问题可以被解决。
 *
 * @Return true 如果问题可以解决，false 否则。
 */
bool L2TileSolver::CheckSolvable() {
  for (uint32_t i = 0; i < input_.size; i++) {
    auto &var = input_.l2_vars[i];
    var.value = var.align;
  }
  if (GetL2Use() > input_.l2_size) {
    ATT_LOG("No solution, l2 size is too small");
    return false;
  }
  return true;
}

/**
 * 初始化输入数据。
 *
 * 这个函数负责初始化 L2TileSolver 对象的输入数据。
 * 它首先为输入数据结构中的每个变量计算最大值，将其向上取整为对齐值的最近倍数。这确保了每个变量的最大值是其对齐值的倍数。
 * 然后，它找出所有变量中的最大最大值，并将每个变量的值初始化为这个最大最大值。
 *
 * 初始化过程对于准备输入数据以进行后续处理步骤（如优化或分析）至关重要。
 * 确保最大值是对齐值的倍数，可以简化数据的处理，并可能是依赖于这种属性的算法或操作所必需的。
 */
void L2TileSolver::InitInput() {
  uint32_t init_value = 0;
  for (uint32_t i = 0; i < input_.size; i++) {
    auto &var = input_.l2_vars[i];
    var.max_value = CeilDivision(var.max_value, var.align) * var.align;
    init_value = var.max_value > init_value ? var.max_value : init_value;
  }
  for (uint32_t i = 0; i < input_.size; i++) {
    auto &var = input_.l2_vars[i];
    var.value = init_value;
  }
}

void L2TileSolver::HandleClash(uint32_t loop_id, uint32_t *ori_val, uint32_t *best_val, uint64_t &max_l2_use) {
  auto max_blocknum = ori_val[loop_id];
  auto &var = input_.l2_vars[loop_id];
  for (uint32_t i = max_blocknum; i >= 1u; i--) {
    blocknum_per_tile_[loop_id] = i;
    size_per_tile_[loop_id] = blocknum_per_tile_[loop_id] * var.base_val;
    tilenum_[loop_id] = CeilDivision(var.max_value, size_per_tile_[loop_id]);
    var.value = size_per_tile_[loop_id];
    if (loop_id == input_.size-1) {
      uint32_t tmp_corenum = 1;
      for (uint32_t j = 0; j < input_.size; j++) {
        tmp_corenum *= blocknum_per_tile_[j];
      }
      used_corenum_ = std::min(input_.core_num, tmp_corenum);
      bool solved=true;
      for (uint32_t k = 0; k < input_.size; k++) {
        if (IsClash(k)) {
          solved=false;
        }
      }
      if (solved) {
        uint64_t l2_use = GetL2Use();
        if (l2_use > max_l2_use) {
          for (uint32_t l = 0; l < input_.size; l++) {
            best_val[l] = blocknum_per_tile_[l];
          }
          max_l2_use = l2_use;
        }
        return;
      }
    } else {
      HandleClash(loop_id+1, ori_val, best_val, max_l2_use);
    }
  }
}

/**
 * 运行 L2TileSolver 算法来解决 L2 缓存分块问题。
 *
 * 这个函数是 L2TileSolver 算法的核心。它尝试根据输入参数和约束条件找到 L2 缓存的最优分块方案。
 * 函数首先通过 CheckInput() 函数检查输入参数的有效性。如果输入无效，它将记录一条错误消息并返回 false。
 * 然后通过 CheckSolvable() 函数检查问题是否有解。如果没有解，它将记录一条错误消息并返回 false。
 * 如果输入有效并且问题有解，函数将通过 InitInput() 函数初始化输入数据。
 *
 * 算法的核心是一个循环，在这个循环中，它不断地调整每个变量的值，以找到一个合适的分块方案。
 * 循环结束的条件是总内存使用量小于或等于 L2 缓存的大小。
 * 循环结束后，它将计算每个变量的每个分块的大小、总块数、分块数和占用的核心数。
 * 然后，它检查是否存在读冲突。如果存在读冲突，它将调整每个分块的大小，直到不再检测到冲突。
 * 最后，它检查每个变量的最大值是否可以放在一个分块中。
 * 如果不能，它将相应地调整分块数和每个分块的大小。
 *
 * 如果找到合适的分块方案，函数返回 true，否则返回 false。
 *
 * @Return 如果成功则返回 true，否则返回 false
 */
bool L2TileSolver::Run() {
  if (!CheckInput()) {
    ATT_LOG("Check input failed");
    return false;
  }
  if (!CheckSolvable()) {
    ATT_LOG("Check Solvable failed");
    return false;
  }
  InitInput();
  uint32_t core_num = input_.core_num;
  uint32_t l2_size = input_.l2_size;
  blocknum_per_tile_ = new(std::nothrow) uint32_t[input_.size];
  size_per_tile_ = new(std::nothrow) uint32_t[input_.size];
  tilenum_ = new(std::nothrow) uint32_t[input_.size];
  total_blocknum_ = new(std::nothrow) uint32_t[input_.size];
  
  // 遍历直到满足L2占用停止
  while (GetL2Use() > l2_size) {
    for (uint32_t i = 0; i < input_.size; i++) {
      auto &var = input_.l2_vars[i];
      var.value = (var.align < var.value) ? (var.value - var.align) : var.align;
    }
  }

  uint32_t *best_val = new uint32_t[input_.size];
  uint32_t *ori_val = new uint32_t[input_.size];
  for (uint32_t i = 0; i < input_.size; i++) {
    auto &var = input_.l2_vars[i];
    blocknum_per_tile_[i] = CeilDivision(var.value, var.base_val);
    size_per_tile_[i] = blocknum_per_tile_[i] * var.base_val;
    tilenum_[i] = CeilDivision(var.max_value, size_per_tile_[i]);
    total_blocknum_[i] = CeilDivision(var.max_value, var.base_val);
    best_val[i] = blocknum_per_tile_[i];
    ori_val[i] = blocknum_per_tile_[i];
  }

  uint64_t max_l2_use = 0u;
  HandleClash(0, ori_val, best_val, max_l2_use);

  for (uint32_t i = 0; i < input_.size; i++) {
    auto &var = input_.l2_vars[i];
    blocknum_per_tile_[i] = best_val[i];
    size_per_tile_[i] = blocknum_per_tile_[i] * var.base_val;
  }

  delete[] best_val;
  delete[] ori_val;

  for (uint32_t i = 0; i < input_.size; i++) {
    auto &var = input_.l2_vars[i];
    if (var.max_value <= size_per_tile_[i]) {
      tilenum_[i] = 1;
      blocknum_per_tile_[i] = CeilDivision(var.max_value, var.base_val);
      size_per_tile_[i] = blocknum_per_tile_[i] * var.base_val;
    }
  }
  return true;
}
}  // namespace att