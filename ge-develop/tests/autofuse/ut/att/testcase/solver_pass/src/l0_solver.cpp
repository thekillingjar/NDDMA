/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <memory.h>
#ifdef DEBUG
#define ATT_LOG(log)
do {
  std::cout << "[ERROR]" << log << std::endl;
} while (0)
#else
#define ATT_LOG(log)
#endif
namespace att {
// L0Var的备选值的个数
static const uint32_t candidate_size = 7u;
// L0Var的备选值
static const uint32_t candidate_value[] = {16u,  32u,  64u,  128u,
                                           256u, 512u, 1024u};
// 表达L0的求解值至少要满足核数的比例，可手动修改
static const double CORE_NUM_RATIO = 0.6f;
// 表达L0的求解值pad之后的值不允许超过原始值大小的倍数，可手动修改
static const uint32_t UPPER_BOUND_RATIO = 2u;
// 表达最大L0Var的个数
static const uint32_t MAX_L0_VAR_NUM = 3u;

// L0相关变量的数据结构
struct L0Var {
  // 最大值，初始化为输入原始轴的大小
  uint32_t max_value{0u};
  // 是否绑多核
  bool bind_multicore{false};
  bool is_innermost{false};
  // 对齐值
  uint32_t align{0u};
  // 提示当前L0Var的最佳对齐值，通常源于父轴的对齐值，
  // 举例，stepm是basem的父轴，stepm的对齐要求是256和basem，那么basem的prompt_align就是256，
  // 同时约束L0Var的取值必须是256对齐或者是256的因子，因为这样父轴stepm才能既满足256也满足basem对齐
  uint32_t prompt_align{0u};
  // L0变量的索引
  uint32_t idx;
  // L0变量的值
  uint32_t value{0u};
};

// 求解器接收的输入
struct L0TileInput {
  // 待求解L0变量的集合
  L0Var *l0_vars{nullptr};
  // 待求解L0变量的数量
  uint32_t size;
  // 核数
  uint32_t core_num;
};

/**
 * 比较两个 L0Var 类型变量的大小
 *
 * 这个函数用来比较两个 L0Var 类型变量的大小。它遵循特定的比较逻辑：
 * 1. 如果 a 变量绑定到多核且 b 变量没有绑定多核，则 a 被认为大于 b，函数返回
 * true。
 * 2. 如果 a 变量没有绑定多核且 b 变量绑定多核，则 a 被认为小于 b，函数返回
 * false。
 * 3. 如果 a 和 b 变量都绑定多核或都未绑定多核，则比较它们的 prompt_align
 * 属性，prompt_align 属性值大的变量被认为更大。
 *
 * @param a 第一个要比较的 L0Var 变量。
 * @param b 第二个要比较的 L0Var 变量。
 * @return 如果 a 大于 b，返回 true；如果 a 小于 b，返回 false。
 */
static bool L0VarCmp(L0Var a, L0Var b) {
  if (a.bind_multicore && !b.bind_multicore) {
    return true;
  }
  if (!a.bind_multicore && b.bind_multicore) {
    return false;
  }
  if (a.is_innermost && !b.is_innermost) {
    return true;
  }
  if (!a.is_innermost && b.is_innermost) {
    return false;
  }
  return a.prompt_align > b.prompt_align;
}
/**
 * L0 求解器类
 */
class L0TileSolver {
public:
  /**
   * 构造函数
   *
   * @param input 一个 L0TileInput 结构体，包含了 L0 变量相关的信息
   *
   * 这个构造函数初始化了 L0TileSolver 对象
   */
  explicit L0TileSolver(L0TileInput input) : input_(input) {}
  L0TileSolver() {};
  /**
   * 析构函数
   *
   * 当 L0TileSolver 对象被销毁时，析构函数被调用
   * 用来释放使用 new 运算符动态分配的内存，确保没有内存泄漏
   */
  ~L0TileSolver() {
    if (sortedvars_ != nullptr) {
      delete[] sortedvars_;
    }
    if (output_ != nullptr) {
      delete[] output_;
    }
  }
  /**
   * 运行求解器
   *
   * @return 如果求解成功，返回 true；否则返回 false
   *
   * 这个方法是算法的入口点，调用它会启动求解过程
   * 成功与否取决于 CheckBufferUseValid() 方法的返回值
   */
  bool Run();
  /**
   * 获取优化结果
   *
   * @return 指向求解结果数据的指针
   *
   * 如果有求解结果，这个方法将返回一个指向结果数据的指针
   * 结果数据的内存使用完后，应该使用 delete[] 释放内存
   */
  uint32_t *GetOutput() { return output_; }

protected:
  /**
   * 检查是否满足buffer约束
   *
   * @return 如果满足，返回 true；否则返回 false
   *
   * 这个纯虚函数要求派生类提供实现，以确保缓冲区使用是有效的
   * 在 L0TileSolver 类中，它是一个抽象方法，需要在子类中实现
   */
  virtual bool CheckBufferUseValid() = 0;
  L0TileInput input_;
  uint32_t *output_{nullptr};

private:
  /**
   * 检查输入数据的完整性和正确性
   *
   * @return 如果输入数据有效，返回 true；否则返回 false
   *
   * 这个私有方法检查输入数据的格式和逻辑，确保它们适用求解算法
   */
  bool CheckInput();

  /**
   * 使用输入数据初始化算法所需的内部数据结构
   *
   * 这个方法根据输入的 L0TileInput 结构体中的数据，初始化算法所需的内部数据结构
   * 确保 sortedvars_ 和 output_ 成员变量被正确初始化
   */
  void InitInput();

  /**
   * 检查算法运行的结果，确保它们符合预期
   *
   * @return 如果输出数据有效，返回 true；否则返回 false
   *
   * 这个方法检查运行算法后得到的结果，确保它们在逻辑上是合理的
   */
  bool CheckOutput();

  /**
   * 更新算法执行过程中的对齐设置
   *
   * 这个方法根据算法执行过程中的数据更新对齐提示值，确保结果按照预期的方式对齐
   */
  void UpdateAlign();

  /**
   * 为指定索引的 L0 变量获取最佳对齐值
   *
   * @param i 想要的变量索引值
   * @return 最佳对齐值
   *
   * 这个方法计算并返回给定索引值的 L0 变量的最佳对齐值，
   * 确保变量以最恰当的方式对齐，从而提高效率或者减少资源浪费
   */
  uint32_t GetBestAlign(uint32_t i) const;

  /**
   * 为 L0 变量找到最优值进行迭代运行
   *
   * @param loop_id 当前循环索引，表示正在处理的 L0 变量的位置
   * @param best_var_value 一个指针，指向用于存储每个 L0
   * 变量迄今为止找到的最佳值的数组
   *
   * 这个函数使用递归方法来遍历 L0
   * 变量的所有可能值。对于每个值，它检查是否满足约束条件，如小于上界且满足对齐要求。如果满足这些条件，它将继续下一个循环或者递归调用自身来处理下一个
   * L0 变量。如果是最后一个 L0
   * 变量，它将检查当前组合是否满足优化条件，如核心数量和数据处理量。如果满足条件，它将当前组合存储为最优解。
   *
   * 请注意，这个函数没有返回值，而是将最优解存储在传入的 best_var_value
   * 数组中。
   */
  void IterativeRun(uint32_t loop_id, uint32_t *best_var_value);

  /**
   * 根据 L0 变量信息和总核心数计算可以分配的最大核心数
   *
   * @param l0_vars 一个指向 L0Var 结构体数组的指针
   * @param core_num 可用于分配的总核心数
   * @return 可以分配的最大核心数
   *
   * 这个函数计算在给定 L0 变量信息和总核心数的情况下，可以分配的最大核心数。
   * 它遍历输入的 L0Var
   * 结构体数组，对于每个变量，根据其是否绑定多核心以及最大、当前和提示对齐值计算所需的块数。
   * 通过将所有变量的块数相乘，得到总块数。
   * 最大核心数是总块数和总核心数中的最小值，以确保核心数不会超过可用资源。
   *
   * 返回值表示可以分配给 L0
   * 变量的最大核心数，这对于在多核心系统中进行资源分配是有用的。
   */
  int32_t MaxCoreNum(const L0Var *l0_vars, const uint32_t &core_num);

  /**
   * 计算所有 L0 变量值的乘积，作为mac计算量的度量
   *
   * @return mac计算量
   *
   * 这个函数计算所有 L0 变量值的乘积，结果是一个数字。
   * 这个数字可以作为数据处理量的度量，例如在评估算法性能时。
   * 通过不断更新 usage 变量，乘法操作确保了所有 L0 变量的影响都被计入。
   * 最终 usage 变量中的值就是所有 L0 变量值的乘积，代表了整体的数据处理量。
   * 返回值可以帮助了解算法处理的数据量，从而对算法的效率和扩展性有更直观的认识。
   */
  uint32_t GetMacUse() const;
  /**
   * 用于排序的 L0Var 对象数组
   */
  L0Var *sortedvars_{nullptr};
  /**
   * 最大核心数
   */
  int64_t max_corenum_{-1};
  /**
   * 最大 MAC 使用量
   */
  int32_t max_macuse_{-1};
};

/**
 * 获取给定索引的 L0 变量的最佳对齐值
 *
 * @param i 想要的变量索引值
 * @return 最佳对齐值
 *
 * 这个方法为给定索引的 L0
 * 变量计算最佳对齐值。它考虑到变量的最大、当前和提示对齐值，以确保数据存储和访问的效率。
 * 根据变量的原始值（ori_value），它首先确定最小和最大对齐值的范围。然后，通过在这个范围内以二的幂次方递增，它找到最大的满足条件的值。
 * 如果没有找到这样的值，它将返回最小对齐值。如果在范围内找到了一个值，它将返回这个值的二分之一，作为最佳对齐值。
 * 这个最佳对齐值可以用于确保数据以最有效的方式存储
 */
uint32_t L0TileSolver::GetBestAlign(uint32_t i) const {
  uint32_t ori_value = input_.l0_vars[i].max_value;
  uint32_t min_align = input_.l0_vars[i].align;
  uint32_t max_align = input_.l0_vars[i].prompt_align;
  uint32_t ori_align = min_align;
  uint32_t max_value = std::min(ori_value, max_align);
  while (ori_align <= max_value) {
    ori_align = ori_align << 1;
  }
  if (ori_align == min_align) {
    return min_align;
  }
  return std::max(1u, ori_align >> 1);
}

/**
 * 根据给定的 L0 变量信息计算可以分配的最大核心数
 *
 * @param l0_vars 指向 L0Var 结构数组的指针
 * @param core_num 总核心数
 * @return 可以分配的最大核心数
 *
 * 这个函数遍历 L0Var 结构数组，根据每个变量的 bind_multicore 属性以及
 * max_value、value 和 prompt_align
 * 的值来计算每个变量所需的块数。对于绑定多核心的变量，块数计算方式为：（max_value
 * + max（value，prompt_align）-1）/
 * max（value，prompt_align）。对于未绑定多核心的变量，块数为 1。
 * 总块数通过将所有变量的块数相乘得到。之后，通过比较总块数和
 * core_num，返回两者中的最小值，作为可以分配的最大核心数。如果总块数超过了
 * core_num，那么系统的核心数将成为瓶颈，因此需要将 core_num
 * 设置为最大核心数。如果总块数小于等于
 * core_num，那么总块数就是可以分配的最大核心数。
 */
int32_t L0TileSolver::MaxCoreNum(const L0Var *l0_vars,
                                 const uint32_t &core_num) {
  uint32_t total_block_size = 1u;
  for (uint32_t i = 0u; i < input_.size; i++) {
    auto var = l0_vars[i];
    uint32_t block_num =
        var.bind_multicore
            ? ((var.max_value + std::max(var.value, var.prompt_align) - 1)) /
                  std::max(var.value, var.prompt_align)
            : 1;
    total_block_size *= block_num;
  }
  int64_t max_core_num =
      total_block_size > core_num ? core_num : total_block_size;
  return max_core_num;
}

/**
 * 计算所有 L0 变量值的乘积，作为数据处理量的度量
 *
 * @return 数据处理量
 *
 * 这个函数遍历 L0TileInput 结构体中的所有 L0Var 对象，计算它们的 value
 * 属性的乘积。这个乘积代表了所有 L0
 * 变量值的联合效应，或者说数据处理量的一个度量。 通过不断更新 usage
 * 变量，乘法操作确保了所有 L0 变量的贡献都被包含在内。最终 usage
 * 变量中的值就是所有 L0 变量值的乘积。
 * 返回值可以帮助评估算法在处理给定输入数据时的效率，以及比较不同算法或优化策略的数据处理量。
 */
uint32_t L0TileSolver::GetMacUse() const {
  uint32_t usage = 1u;
  for (uint32_t j = 0; j < input_.size; j++) {
    usage *= input_.l0_vars[j].value;
  }
  return usage;
}

/**
 * 为 L0 变量找到最优值进行迭代运行
 *
 * @param loop_id 当前循环索引，表示正在处理的 L0 变量的位置
 * @param best_var_value 一个指针，指向用于存储每个 L0 变量迄今为止找到的最佳值的数组
 *
 * 这个函数使用递归方法来遍历 L0 变量的所有可能值。对于每个值，它检查是否满足约束条件，如小于上界且满足对齐要求。
 * 如果满足这些条件，它将继续下一个循环或者递归调用自身来处理下一个L0 变量。
 * 如果是最后一个 L0 变量，它将检查当前组合是否满足优化条件，如核心数量和数据处理量。如果满足条件，它将当前组合存储为最优解。
 *
 * 请注意，这个函数没有返回值，而是将最优解存储在传入的 best_var_value 数组中。
 */
void L0TileSolver::IterativeRun(uint32_t loop_id, uint32_t *best_var_value) {
  for (uint32_t i = 0u; i < candidate_size; i++) {
    uint32_t candi_value = candidate_value[i];
    const auto &l0_tile = sortedvars_[loop_id];
    // L0Var的上限
    uint32_t upper_bound = l0_tile.max_value * UPPER_BOUND_RATIO;
    if (candi_value >= upper_bound) {
      continue;
    }
    // 必须满足prompt_align对齐或者是prompt_align的因子
    if ((candi_value % l0_tile.prompt_align != 0) &&
        (l0_tile.prompt_align % candi_value != 0)) {
      continue;
    }
    auto idx = l0_tile.idx;
    input_.l0_vars[idx].value = candi_value;
    // 终止条件为遍历到最后一个变量
    if (loop_id == input_.size - 1) {
      if (!CheckBufferUseValid()) {
        break;
      }
      int32_t usage = GetMacUse();
      int32_t core_num = MaxCoreNum(input_.l0_vars, input_.core_num);
      // 最大核数如果满足核数*系数（默认0.6），则比较mac利用率即可，否则需要比较核数的使用和mac利用率
      if (((core_num >= max_corenum_) ||
           (core_num >=
            static_cast<int32_t>(input_.core_num * CORE_NUM_RATIO))) &&
          (usage >= max_macuse_)) {
        max_corenum_ = core_num;
        max_macuse_ = usage;
        for (uint32_t k = 0u; k < input_.size; k++) {
          best_var_value[k] = input_.l0_vars[k].value;
        }
      }
    } else {
      IterativeRun(loop_id + 1, best_var_value);
    }
  }
}

/**
 * 更新 L0Var 对象的对齐值
 *
 * 这个函数用于更新 L0Var 对象的 prompt_align
 * 值，以确保它们在内存中按照最优方式对齐。它遍历 input_ 对象中的 l0_vars
 * 数组，为每个 L0Var 对象计算并设置最佳的对齐值。
 *
 * @param 无
 * @return 无
 */
void L0TileSolver::UpdateAlign() {
  for (uint32_t i = 0u; i < input_.size; i++) {
    uint32_t best_align = GetBestAlign(i);
    input_.l0_vars[i].prompt_align = best_align;
  }
}

/**
 * 检查输入数据的有效性
 *
 * 这个函数用来检查 L0TileSolver 类的输入数据是否有效。它验证以下几个方面：
 * - 基础变量指针（l0_vars）是否为空。
 * - 输入数据的大小（size）是否为0，表示没有 L0 参数需要求解。
 * - 输入数据的大小（size）是否超过最大支持的参数数量（MAX_L0_VAR_NUM）。
 * - 核心数量（core_num）是否为0。
 * - 对于输入数据中的每个 L0Var 对象（通过索引 i 访问），它检查几个属性：
 *   - max_value、align 和 prompt_align 是否都不等于0。
 *   - align 是否不大于 prompt_align。
 *
 * 如果以上任何一个条件不满足，函数将通过 ATT_LOG 宏记录一条错误消息，并返回
 * false，表示输入无效。如果所有条件都满足，函数返回 true，表示输入有效。
 *
 * @return 如果输入数据有效，则返回 true；否则返回 false。
 */
bool L0TileSolver::CheckInput() {
  if (input_.l0_vars == nullptr) {
    ATT_LOG("Input basevar is null");
    return false;
  }
  if (input_.size == 0u) {
    ATT_LOG("Size is 0, no l0 arg to be solved");
    return false;
  }
  if (input_.size > MAX_L0_VAR_NUM) {
    ATT_LOG("L0 solver does not support more than 3 input args");
    return false;
  }
  if (input_.core_num == 0) {
    ATT_LOG("Corenum is 0");
    return false;
  }
  for (uint32_t i = 0u; i < input_.size; i++) {
    auto var = input_.l0_vars[i];
    if ((var.max_value == 0) || (var.align == 0) || (var.prompt_align == 0)) {
      ATT_LOG("Input [" + std::to_string(i) + "] exists 0");
      return false;
    }
    if (var.align > var.prompt_align) {
      ATT_LOG("Input [" + std::to_string(i) +
              "] align is larger than prompt align");
      return false;
    }
  }
  return true;
}

/**
 * 初始化 L0Var 对象数组
 *
 * 这个函数用于初始化 L0TileSolver 类的 input_ 对象中的 l0_vars 数组。它遍历
 * l0_vars 数组中的每一个元素，对于每个元素，执行以下操作：
 * 1. 通过访问索引 i 对应的 L0Var 对象的引用 var，重置其 max_value
 * 属性。具体重置方式是，先将 max_value 增加 align 属性值减 1，再除以 align
 * 属性值，最后乘以 align 属性值。这样做的目的可能是为了确保 max_value 是 align
 * 的整数倍。
 * 2. 将当前循环的索引值 i 设置为 var 的 idx 属性。这可能是为了标记每个 L0Var
 * 对象在数组中的位置，以便后续处理。
 *
 * @param 无
 * @return 无
 */
void L0TileSolver::InitInput() {
  for (uint32_t i = 0u; i < input_.size; i++) {
    auto &var = input_.l0_vars[i];
    var.max_value = (var.max_value + var.align - 1) / var.align * var.align;
    var.idx = i;
  }
}

/**
 * 检查输出数据的有效性
 *
 * 这个函数用于检查 L0TileSolver 类的输出数据是否有效。它首先检查 output_
 * 指针是否为空。如果 output_ 指针为空，通过 ATT_LOG 宏记录一条错误消息，并返回
 * false，表示输出无效。
 *
 * 接着，函数遍历 output_ 数组中的每个元素。对于每个元素，它检查其值是否为
 * 0。如果发现任何一个元素的值为 0，函数会通过 ATT_LOG
 * 宏记录相应的错误消息，并返回 false，表示输出数据中存在无效的元素。
 *
 * 如果输出数据有效，即 output_ 指针不为空且 output_ 数组中没有 0
 * 值元素，函数返回 true。
 *
 * @return 如果输出数据有效，则返回 true；否则返回 false。
 */
bool L0TileSolver::CheckOutput() {
  if (output_ == nullptr) {
    ATT_LOG("Output is null");
    return false;
  }
  for (uint32_t i = 0u; i < input_.size; i++) {
    if (output_[i] == 0u) {
      ATT_LOG("Output [" + std::to_string(i) + "] is 0");
      return false;
    }
  }
  return true;
}

/**
 * 执行 L0TileSolver 类的主要流程
 * @return 如果所有操作成功并且输出有效，则返回 true；否则返回 false
 */
bool L0TileSolver::Run() {
  // 检查输入数据的有效性
  if (!CheckInput()) {
    // 如果输入检查失败，则记录一条错误日志，并返回 false
    ATT_LOG("Check input failed");
    return false;
  }

  // 初始化输入数据
  InitInput();

  // 更新 L0Var 对象的对齐值
  UpdateAlign();

  // 为排序后的变量申请内存，并初始化为 0
  sortedvars_ = new (std::nothrow) L0Var[input_.size];
  output_ = new (std::nothrow) uint32_t[input_.size]();

  // 将输入数据复制到新的内存中
  std::copy(input_.l0_vars, input_.l0_vars + input_.size, sortedvars_);

  // 根据比较函数对变量进行排序
  std::sort(sortedvars_, sortedvars_ + input_.size, L0VarCmp);

  // 调用 IterativeRun 函数，传递参数 0 和 output_ 数组的指针
  IterativeRun(0u, output_);

  // 检查输出数据的有效性
  if (!CheckOutput()) {
    // 如果输出检查失败，则记录一条错误日志，并返回 false
    ATT_LOG("Check output failed");
    return false;
  }

  // 如果所有操作都成功，返回 true
  return true;
}
}  // namespace att