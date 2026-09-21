/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <memory.h>
#include <iostream>
#include <algorithm>
#include "op_log.h"
#include "tiling_data.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "OpTest"

namespace optiling {
using namespace std;
inline bool IsEqual(double a, double b)
{
    const double epsilon = 0.001;
    double abs = (a > b) ? (a - b) : (b - a);
    return abs < epsilon;
}
template<typename T>
inline T ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (IsEqual(value, a)) ? value : (value + 1);
}

class TilingCaseImpl {
 public:
  virtual ~TilingCaseImpl() = default;
  bool GetTiling(MMTilingData &tiling_data) {
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    ExtraTilingData(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual void GetTilingData(MMTilingData &tiling_data, MMTilingData &to_tiling) = 0;
  virtual void SetTilingData(MMTilingData &from_tiling, MMTilingData &tiling_data) = 0;
  virtual double GetPerf(MMTilingData &tiling_data) { return 0.0; }
 protected:
  virtual bool DoTiling(MMTilingData &tiling_data) = 0;
  virtual void DoApiTiling(MMTilingData &tiling_data) = 0;
  virtual void ExtraTilingData(MMTilingData &tiling_data) = 0;
  virtual void TilingSummary(MMTilingData &tiling_data) = 0;
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;

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
 * 如果以上任何一个条件不满足，函数将通过 OP_LOG 宏记录一条错误消息，并返回
 * false，表示输入无效。如果所有条件都满足，函数返回 true，表示输入有效。
 *
 * @return 如果输入数据有效，则返回 true；否则返回 false。
 */
bool L0TileSolver::CheckInput() {
  if (input_.l0_vars == nullptr) {
    OP_LOGW(OP_NAME, "Input basevar is null");
    return false;
  }
  if (input_.size == 0u) {
    OP_LOGW(OP_NAME, "Size is 0, no l0 arg to be solved");
    return false;
  }
  if (input_.size > MAX_L0_VAR_NUM) {
    OP_LOGW(OP_NAME, "L0 solver does not support more than 3 input args");
    return false;
  }
  if (input_.core_num == 0) {
    OP_LOGW(OP_NAME, "Corenum is 0");
    return false;
  }
  for (uint32_t i = 0u; i < input_.size; i++) {
    auto var = input_.l0_vars[i];
    if ((var.max_value == 0) || (var.align == 0) || (var.prompt_align == 0)) {
      OP_LOGW(OP_NAME, "Input [%u] exists 0", i);
      return false;
    }
    if (var.align > var.prompt_align) {
      OP_LOGW(OP_NAME, "Input [%u] align is larger than prompt align", i);
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
 * 指针是否为空。如果 output_ 指针为空，通过 OP_LOG 宏记录一条错误消息，并返回
 * false，表示输出无效。
 *
 * 接着，函数遍历 output_ 数组中的每个元素。对于每个元素，它检查其值是否为
 * 0。如果发现任何一个元素的值为 0，函数会通过 OP_LOG
 * 宏记录相应的错误消息，并返回 false，表示输出数据中存在无效的元素。
 *
 * 如果输出数据有效，即 output_ 指针不为空且 output_ 数组中没有 0
 * 值元素，函数返回 true。
 *
 * @return 如果输出数据有效，则返回 true；否则返回 false。
 */
bool L0TileSolver::CheckOutput() {
  if (output_ == nullptr) {
    OP_LOGW(OP_NAME, "Output is null");
    return false;
  }
  for (uint32_t i = 0u; i < input_.size; i++) {
    if (output_[i] == 0u) {
      OP_LOGW(OP_NAME, "Output [%u] is 0", i);
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
    OP_LOGW(OP_NAME, "Check input failed");
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
    OP_LOGW(OP_NAME, "Check output failed");
    return false;
  }

  // 如果所有操作都成功，返回 true
  return true;
}
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
    OP_LOGW(OP_NAME, "Input l2var is null");
    return false;
  }
  if (input_.size == 0 || input_.core_num == 0 || input_.l2_size == 0) {
    OP_LOGW(OP_NAME, "Exist input 0, please check size, core_num and l2_size");
    return false;
  }
  for (uint32_t i = 0; i < input_.size; i++) {
    auto var = input_.l2_vars[i];
    if (var.align == 0 || var.base_val == 0 || var.max_value == 0) {
      OP_LOGW(OP_NAME, "Input [%u] exists 0", i);
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
    OP_LOGW(OP_NAME, "No solution, l2 size is too small");
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
    OP_LOGW(OP_NAME, "Check input failed");
    return false;
  }
  if (!CheckSolvable()) {
    OP_LOGW(OP_NAME, "Check Solvable failed");
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
/*
(可修改变量)用于控制通用求解器求解质量的超参数
cfg_top_num:保留目标函数最优的前top_num个解,用户可以打印这些解并从中选取较优项(默认值为5)
cfg_search_length:在可行域内执行局部搜索的搜索范围,当搜索范围内存在更优的解时会将该解视为候选
  搜索范围越大,越有可能获取更优的解,但求解耗时更长(默认值为1)
cfg_iterations:启发式求解算法的迭代轮次上限,算法最多执行iterations次,并在满足早停逻辑时提前退出
  在不满足早停逻辑的前提下,设置更大的iterations算法有机会取得更好的解,但求解耗时更长(默认值为500)
cfg_simple_ver:用户可以选择使用的求解器版本(高效率版/高性能版)
  高效率版采用二分搜索逻辑搜索更优解,变量求解顺序相对简单
  高性能版会检查搜索范围内所有的可行解,同时采用更精细的变量求解顺序
  高性能版的耗时相对更长,但是可能取到比高效率版更优的解(默认采用高效率版)
cfg_momentum_factor:更新变量信息时所采用的动量因子
  在选取变量时,变量的动量值为momentum * momentum_factor + update_value * (1 - momentum_factor)
  动量因子越大,求解器越可能反复选取同一个变量进行更新(默认值为0.9)
  当用户取大于1的数时取1,取小于0的数时取0
*/
static const uint64_t cfg_top_num = 5;
static const uint64_t cfg_search_length = 1;
static const uint64_t cfg_iterations = 100;
static const bool cfg_simple_ver = false;
static const double cfg_momentum_factor = 0.9;

/*
Locality:定域过程中待求解变量的优先级
  GLOBALVALID:更新该变量会使待求解变量走入可行域,即直接获取一个可行解
  LOCALVALID:更新该变量能满足该变量相关的约束
  CROSSREGION:更新该变量会跨越可行域,即由可行域的一侧到达另一侧
  INVALID:仅更新该变量无法获取可行域内的解,即定义域内不存在可行域
  ALTERNATIVE:(仅在高性能版本中生效)该变量的预期落点是曾搜索得到的解,尝试跨越可行域获取另一侧边界的解作为备选方案
  REJECT:该变量的落点为上轮迭代中的实际落点,即出现了反复震荡
*/
enum class Locality
{
    GLOBALVALID = 0,
    LOCALVALID = 1,
    CROSSREGION = 2,
    INVALID = 3,
    ALTERNATIVE = 4,
    REJECT = 5,
};

/*
TunePriority:微调过程中待求解变量的优先级
  HARMLESS:更新该变量会获得一个目标函数更优的可行解(即存在无损更新)
  DILATED:更新该变量会获得一个目标函数不变,距离缓存占用边界更近的可行解(即存在膨胀更新)
  NORMAL:沿着目标函数的优化方向进行更新会走出可行域
  OTHER:更新变量会走出可行域并获得一个更差的解
  TABU:该变量的落点为上轮迭代中的实际落点,即出现了反复震荡
  REFUSE:更新后会在可行域内获得一个更差的解
*/
enum class TunePriority
{
    HARMLESS = 0,
    DILATED = 1,
    NORMAL = 2,
    OTHER = 3,
    TABU = 4,
    REFUSE = 5,
};

/*
FuncInfo:函数信息
  LEQ:不等式约束所对应的罚函数
  BUFFER:缓存占用约束所对应的罚函数
*/
enum class FuncInfo
{
    LEQ = 0,
    BUFFER = 1,
};

/*
UpdateDirection:变量的更新方向
  POSITIVE:沿正方向更新
  NONE:不存在更新方向
  POSITIVE:沿负方向更新
*/
enum class UpdateDirection
{
    POSITIVE = 0,
    NONE = 1,
    NEGATIVE = 2,
};

/*
UpdateInfo:变量的更新信息
  idx:变量的索引值
  thres:沿着更新方向变量的更新阈值
  update_direction:变量的更新方向
  init_obj:更新前变量的目标函数值
  init_cons:更新前变量的缓存占用冗余
*/
struct UpdateInfo
{
    int32_t idx{0};
    uint64_t thres{0u};
    UpdateDirection update_direction{UpdateDirection::NONE};
    double init_obj{0};
    double init_cons{0};
    UpdateInfo(int32_t idx, uint64_t thres, UpdateDirection direction, double obj = 0, double cons = 0) : idx(idx), thres(thres), update_direction(direction), init_obj(obj), init_cons(cons) {}
};

/*
Node:用于记录待求解变量的数据结构,以{x0,x1}为例,假设当前指向x0
  value:x0的值
  next_val:x0的下一个值
  next_var:当前x0的value所对应的解中x1的第一个值
  next_node:指向下一个node对象的指针
*/
struct Node
{
    uint64_t value{0u};
    bool searched{false};
    Node *next_val{nullptr};
    Node *next_var{nullptr};
    Node *next_node{nullptr};
    explicit Node(uint64_t val) : value(val) {}
};

/*
VisitedNode:用于记录已搜索到的可行解
  depth:待求解变量的个数
  head:首个node节点(为值为0)
  tail:最后一个node节点
*/
class VisitedNode
{
public:
    explicit VisitedNode(int32_t var_num) : depth(var_num)
    {
        head = new(std::nothrow) Node(0);
        if (head == nullptr)
        {
            throw "Create head failed.";
        }
        tail = head;
    }
    ~VisitedNode()
    {
        Node *temp;
        Node *cur = head;
        while (cur != nullptr)
        {
            temp = cur;
            cur = cur->next_node;
            delete temp;
        }
    }
    Node *GetVarVal(uint64_t *vars);

private:
    uint64_t depth{0};
    Node *head{nullptr};
    Node *tail{nullptr};
};

/*
SolverInput:求解器所需的输入信息
  var_num:待求解的变量个数
  leq_num:不等式约束的个数
  upper_bound:每个待求解变量的上界(共var_num个元素)
  cur_vars:每个待求解变量的初始化值(共var_num个元素)
  update_last:用于标记需要最后切分的待求解变量,为true时对应位置的变量最后更新(共var_num个元素)
*/
struct SolverInput
{
    int32_t var_num{0};
    int32_t leq_num{0};
    uint64_t *upper_bound{nullptr};
    uint64_t *lower_bound{nullptr};
    uint64_t *cur_vars{nullptr};
    bool *update_last{nullptr};
};

struct SolverConfig
{
    uint64_t top_num{5u};
    uint64_t search_length{1u};
    uint64_t iterations{500u};
    bool simple_ver{false};
    double momentum_factor{0.9f};
};

/*
VarVal:用于输出至Result的中间信息
  var_num_:待求解变量的个数
  obj_:解的目标函数值
  cons_:解的缓存占用冗余值
  vars_:可行解的值
*/
class VarVal
{
public:
    VarVal(int32_t var_num, double obj, double cons, uint64_t *varval)
    {
        if (var_num == 0)
        {
            throw "var_num = 0.";
        }
        var_num_ = var_num;
        obj_ = obj;
        cons_ = cons;
        vars_ = new(std::nothrow) uint64_t[var_num];
        if (vars_ == nullptr)
        {
            throw "Create vars_ failed.";
        }
        for (int32_t i = 0; i < var_num; i++)
        {
            vars_[i] = varval[i];
        }
    }
    ~VarVal()
    {
        delete[] vars_;
    }
    void GetVarInfo(double &obj, double &cons) const;
    void GetVars(uint64_t *vars);

private:
    int32_t var_num_{0};
    double obj_{0};
    double cons_{0};
    uint64_t *vars_{nullptr};
};

/*
Result:最终输出的解信息
  top_n_:最多可以记录的可行解个数
  var_num_:待求解变量的个数
  solution_num_:输出的可行解个数(不会大于top_n)
  solution_:输出的可行解(占用空间的尺寸为top_n*var_num_,有效元素个数为solution_num_*var_num_)
    其中,第i组解可通过访问[(i-1)*var_num_, i*var_num_)范围内的元素获取
*/
class Result
{
public:
    Result(int32_t top_num, int32_t var_num)
    {
        if (top_num == 0)
        {
            throw "top_num = 0.";
        }
        solution_num_ = 0;
        top_n_ = top_num;
        var_num_ = var_num;
        solution_ = new(std::nothrow) VarVal *[top_num];
        if (solution_ == nullptr)
        {
            throw "Create solution_ failed.";
        }
    }
    ~Result()
    {
        for (uint32_t i = 0; i < solution_num_; i++)
        {
            delete solution_[i];
        }
        delete[] solution_;
    }
    bool AddVarVal(uint64_t *vars, double obj, double cons);
    bool GetResult(int32_t &solution_num, uint64_t *solution);

private:
    uint32_t top_n_{0};
    uint32_t var_num_{0};
    uint32_t solution_num_{0};
    VarVal **solution_{nullptr};
};

/*
VarInfo:求解过程中的中间参数
  var_num:待求解变量个数
  chosen_var_idx:本轮迭代过程中待更新的变量下标
  upper_bound:待求解变量的上界(var_num个)
  history_vars:上轮迭代过程启动前待求解变量的值(var_num个)
  rec_vars:执行本轮迭代时待求解变量的值(var_num个)
  cur_vars:待求解变量的当前值(var_num个)
  target_val:待求解变量在本轮迭代过程中的预期值(var_num个)
  update_last:用于标记待求解变量,指明该变量是否需要最后切分
*/
struct VarInfo
{
    int32_t var_num{0};
    int32_t chosen_var_idx{-1};
    uint64_t *upper_bound{nullptr};
    uint64_t *lower_bound{nullptr};
    uint64_t *history_vars{nullptr};
    uint64_t *rec_vars{nullptr};
    uint64_t *cur_vars{nullptr};
    uint64_t *target_val{nullptr};
    bool *update_last{nullptr};
    VarInfo(const SolverInput &input)
    {
        if (input.var_num == 0)
        {
            throw "input.var_num == 0";
        }
        var_num = input.var_num;
        upper_bound = new(std::nothrow) uint64_t[input.var_num];
        if (upper_bound == nullptr)
        {
            throw "Create upper_bound failed.";
        }
        lower_bound = new(std::nothrow) uint64_t[input.var_num];
        if (lower_bound == nullptr)
        {
            throw "Create lower_bound failed.";
        }
        history_vars = new(std::nothrow) uint64_t[input.var_num];
        if (history_vars == nullptr)
        {
            throw "Create history_vars failed.";
        }
        rec_vars = new(std::nothrow) uint64_t[input.var_num];
        if (rec_vars == nullptr)
        {
            throw "Create rec_vars failed.";
        }
        cur_vars = new(std::nothrow) uint64_t[input.var_num];
        if (cur_vars == nullptr)
        {
            throw "Create cur_vars failed.";
        }
        target_val = new(std::nothrow) uint64_t[input.var_num];
        if (target_val == nullptr)
        {
            throw "Create target_val failed.";
        }
        update_last = new(std::nothrow) bool[input.var_num];
        if (update_last == nullptr)
        {
            throw "Create update_last failed.";
        }
        for (int32_t i = 0; i < var_num; i++)
        {
            cur_vars[i] = input.cur_vars[i];
            upper_bound[i] = input.upper_bound[i];
            lower_bound[i] = input.lower_bound[i];
        }
    }
    ~VarInfo()
    {
        delete[] upper_bound;
        delete[] lower_bound;
        delete[] history_vars;
        delete[] rec_vars;
        delete[] cur_vars;
        delete[] target_val;
        delete[] update_last;
    }
};

/*
ConsInfo:不等式约束信息
  leq_num:不等式约束个数
  leqs:不等式约束的函数值
*/
struct ConsInfo
{
    int32_t leq_num{0};
    double *leqs{nullptr};
    ConsInfo(int32_t num_leq)
    {
        if (num_leq == 0)
        {
            throw "num_leq = 0.";
        }
        leq_num = num_leq;
        leqs = new(std::nothrow) double[leq_num];
        if (leqs == nullptr)
        {
            throw "Create leqs failed.";
        }
    }
    ~ConsInfo()
    {
        delete[] leqs;
    }
};

/*
Momentum:动量信息
  momentum:上轮迭代的动量值
  cur_value:本轮迭代的动量信息
  is_valid:用于判断是否为有效动量
*/
struct Momentum
{
    double *momentum{nullptr};
    double *cur_value{nullptr};
    bool *is_valid{nullptr};
    Momentum(int32_t var_num)
    {
        if (var_num == 0)
        {
            throw "var_num = 0.";
        }
        momentum = new(std::nothrow) double[var_num];
        if (momentum == nullptr)
        {
            throw "Create momentum failed.";
        }
        cur_value = new(std::nothrow) double[var_num];
        if (cur_value == nullptr)
        {
            throw "Create cur_value failed.";
        }
        is_valid = new(std::nothrow) bool[var_num];
        if (is_valid == nullptr)
        {
            throw "Create is_valid failed.";
        }
    }
    ~Momentum()
    {
        delete[] momentum;
        delete[] cur_value;
        delete[] is_valid;
    }
};

class GeneralSolver
{
public:
    explicit GeneralSolver(SolverConfig &config)
    {
        solver_config_ = config;
    }
    virtual ~GeneralSolver()
    {
        delete var_info_;
        delete cons_info_;
        delete momentum_info_;
        delete visited_node_;
        delete result_;
    }

    bool Init(const SolverInput &input);
    virtual bool Run(int32_t &solution_num, uint64_t *solutions);

    int32_t GetVarNum() const;

    double GetFuncVal(uint64_t *vars, double *weight, FuncInfo func_info);
    UpdateDirection GetDescent(uint64_t *vars, int32_t idx, FuncInfo func_info);

    virtual void DisplayVarVal(uint64_t *vars) = 0;
    virtual double GetObj(uint64_t *vars) = 0;
    virtual double GetSmoothObj(uint64_t *vars) = 0;
    virtual double GetBuffCost(uint64_t *vars) = 0;
    virtual double GetBuffDiff(uint64_t *vars, double *weight) = 0;
    virtual double GetLeqDiff(uint64_t *vars, double *weight) = 0;
    virtual bool CheckLocalValid(double *leqs, int32_t idx) = 0;
    virtual void UpdateLeqs(uint64_t *vars, int32_t idx, double *leqs) = 0;

    SolverConfig solver_config_;
private:
    bool SetSolverInput(const SolverInput &input);
    bool SearchVars(uint64_t *vars) const;
    bool UpdateCurVarVal(uint64_t value, int32_t idx);

    Locality GetLocality(int32_t idx, UpdateDirection update_direction);
    bool GetCoarseLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality);
    bool GetFineLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality);
    bool GetPeerLoc(const UpdateInfo *update_info, Locality &cur_locality);
    bool LocateLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality, Locality &best_locality);
    bool TryLocate(int32_t idx, double init_obj, Locality &best_locality);

    TunePriority GetTunePriority(int32_t idx, double rec_obj, double &cur_obj);
    bool SearchLoc(const UpdateInfo *update_info, uint64_t &step, double &cur_obj, TunePriority &cur_priority);
    bool GetHarmlessLoc(const UpdateInfo *update_info, uint64_t &step, double &cur_obj);
    bool GetDilatedLoc(const UpdateInfo *update_info, uint64_t &step);
    bool TuneLoc(const UpdateInfo *update_info, double cur_obj, uint64_t &step, TunePriority &cur_priority, TunePriority &best_priority);
    bool TryTune(int32_t idx, UpdateDirection update_direction, double init_obj, double init_cons, TunePriority &best_priority);

    bool CheckValid() const;
    void ResetMomentum();
    void UpdateMomentum(int32_t idx, double update_value, Locality cur_locality, Locality &best_locality);
    void UpdateMomentum(int32_t idx, double update_value, TunePriority cur_priority, TunePriority &best_priority);
    bool GetBestChoice();
    bool UpdateBestVar();

    void Initialize(int32_t iter);
    bool LocateRegion();
    bool FineTune();
    bool RecordBestVarVal();
    bool is_feasible_{false};
    bool has_feasible_{false};

    Result *result_{nullptr};
    VarInfo *var_info_{nullptr};
    ConsInfo *cons_info_{nullptr};
    Momentum *momentum_info_{nullptr};
    VisitedNode *visited_node_{nullptr};
};

inline int32_t GetValue(UpdateDirection update_direction)
{
    const int32_t positive = 1;
    const int32_t none = 0;
    const int32_t negative = -1;
    if (update_direction == UpdateDirection::POSITIVE) {
        return positive;
    } else if (update_direction == UpdateDirection::NEGATIVE) {
        return negative;
    }
    return none;
}

inline uint64_t Bound(uint64_t upper_bound, uint64_t lower_bound, uint64_t val, uint64_t step, UpdateDirection direction)
{
    if (direction == UpdateDirection::POSITIVE)
    {
        return (step + val > upper_bound) ? upper_bound : (step + val);
    }
    return (step > val) ? lower_bound : ((val - step < lower_bound) ? lower_bound : (val - step));
}

void VarVal::GetVarInfo(double &obj, double &cons) const
{
    obj = obj_;
    cons = cons_;
}

void VarVal::GetVars(uint64_t *vars)
{
    for (int32_t i = 0; i < var_num_; i++)
    {
        vars[i] = vars_[i];
    }
}

/*
函数名:GetVarVal
功能描述:在VisitedNode中检查vars是否曾被搜索,若未被搜索则会在VisitedNode中构建vars对象
输入参数:
  vars:待求解变量所对应的一组解
*/
Node *VisitedNode::GetVarVal(uint64_t *vars)
{
    Node *new_node;
    Node *cur_node = head;
    for (uint32_t i = 0; i < depth; i++)
    {
        if (!cur_node->next_var)
        {
            new_node = new(std::nothrow) Node(vars[i]);
            if (new_node == nullptr)
            {
                OP_LOGW(OP_NAME, "Create new_node failed.");
                return nullptr;
            }
            if (new_node != nullptr) {
                cur_node->next_var = new_node;
                tail->next_node = new_node;
                tail = tail->next_node;
            }
        }
        cur_node = cur_node->next_var;
        while (cur_node->next_val != nullptr)
        {
            if (cur_node->value == vars[i])
            {
                break;
            }
            cur_node = cur_node->next_val;
        }
        if (cur_node->value != vars[i])
        {
            new_node = new(std::nothrow) Node(vars[i]);
            if (new_node == nullptr)
            {
                OP_LOGW(OP_NAME, "Create new_node failed.");
                return nullptr;
            }
            if (new_node != nullptr) {
                cur_node->next_val = new_node;
                tail->next_node = new_node;
                tail = tail->next_node;
                cur_node = new_node;
            }
        }
    }
    return cur_node;
}

/*
函数名:AddVarVal
功能描述:将一组可行解vars传入Result
  若这组可行解的质量较差(目标函数值较大或距离约束边界较远),则舍弃
  若这组可行解可以被排进前top_n_,则保留该组可行解
  temp: 最大容量为top_n的备选可行解集
  先将solution_复制到temp中
  然后比较new_vars的目标值与temp中元素的目标值
  自小到大地将可行解填入solution_
输入参数:
  vars:一组可行解
  obj:该可行解所对应的目标函数值
  cons:可行解距约束边界的距离
*/
bool Result::AddVarVal(uint64_t *vars, double obj, double cons)
{
    uint64_t rec_num = solution_num_;
    if (rec_num > MAX_SOLUTION) {
        OP_LOGE(OP_NAME, "Too much solutions.");
        return false;
    }
    uint32_t cnt_num = 0;
    uint32_t temp_idx = 0;
    double cur_obj;
    double cur_cons;
    bool has_add = false;
    solution_num_ = Min(solution_num_ + 1, top_n_);
    VarVal *new_vars = new(std::nothrow) VarVal(var_num_, obj, cons, vars);
    if (new_vars == nullptr)
    {
        OP_LOGW(OP_NAME, "Create new_vars failed.");
        return false;
    }
    if (rec_num == 0)
    {
        solution_[0] = new_vars;
        return true;
    }
    VarVal **temp = new(std::nothrow) VarVal *[rec_num];
    if (temp == nullptr)
    {
        OP_LOGW(OP_NAME, "Create temp failed.");
        return false;
    }

    for (uint64_t i = 0; i < rec_num; i++)
    {
        temp[i] = solution_[i];
    }

    while ((cnt_num < solution_num_) && (temp_idx < rec_num))
    {
        temp[temp_idx]->GetVarInfo(cur_obj, cur_cons);
        if (!has_add && (obj < cur_obj || (IsEqual(obj, cur_obj) && cons < cur_cons)))
        {
            has_add = true;
            solution_[cnt_num++] = new_vars;
        }
        else
        {
            solution_[cnt_num++] = temp[temp_idx++];
        }
    }

    if ((!has_add) && (cnt_num < solution_num_))
    {
        solution_[cnt_num++] = new_vars;
        has_add = true;
    }

    if (!has_add) {
        delete new_vars;
    } else if (rec_num == solution_num_) {
        delete temp[temp_idx];
    }
    for (uint32_t i = 0; i < rec_num; i++)
    {
        temp[i] = nullptr;
    }
    delete[] temp;

    return cnt_num == solution_num_;
}

bool Result::GetResult(int32_t &solution_num, uint64_t *solution)
{
    for (uint32_t i = 0; i < solution_num_; i++)
    {
        solution_[i]->GetVars(solution + i * var_num_);
    }
    solution_num = solution_num_;
    return true;
}

double GeneralSolver::GetFuncVal(uint64_t *vars, double *weight, FuncInfo func_info)
{
    if (func_info == FuncInfo::BUFFER)
    {
        return GetBuffDiff(vars, weight);
    }
    else if (func_info == FuncInfo::LEQ)
    {
        return GetLeqDiff(vars, weight);
    }
    return 0;
}

/*
函数名:GetDescent
功能描述:获取“缓存占用函数/不等式约束的罚函数”的下降方向
输入参数:
  vars:当前待求解参数的下降方向
  idx:关于某参数下降方向中,某参数的下标
  func_info:用于指明计算下降方向的函数(FuncInfo::BUFFER/FuncInfo::LEQ)
*/
UpdateDirection GeneralSolver::GetDescent(uint64_t *vars, int32_t idx, FuncInfo func_info)
{
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return UpdateDirection::NONE;
    }
    double *weight = (double *)malloc(cons_info_->leq_num * sizeof(double));
    UpdateLeqs(vars, -1, weight);
    double cur_val = GetFuncVal(vars, weight, func_info);
    vars[idx] += 1;
    double next_val = GetFuncVal(vars, weight, func_info);
    vars[idx] -= 1;
    if (!IsEqual(cur_val, next_val))
    {
        return (cur_val > next_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
    }
    if (vars[idx] >= 1)
    {
        vars[idx] -= 1;
        double pre_val = GetFuncVal(vars, weight, func_info);
        vars[idx] += 1;
        if (!IsEqual(cur_val, pre_val))
        {
            return (pre_val > cur_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
        }
    }
    return UpdateDirection::NONE;
}

bool GeneralSolver::SetSolverInput(const SolverInput &input)
{
    if (input.var_num <= 0)
    {
        return false;
    }
    visited_node_ = new(std::nothrow) VisitedNode(input.var_num);
    if (visited_node_ == nullptr)
    {
        OP_LOGW(OP_NAME, "Create visited_node_ failed.");
        return false;
    }
    var_info_ = new(std::nothrow) VarInfo(input);
    cons_info_ = new(std::nothrow) ConsInfo(input.leq_num);
    momentum_info_ = new(std::nothrow) Momentum(input.var_num);
    if (var_info_ != nullptr && cons_info_ != nullptr && momentum_info_ != nullptr)
    {
        for (int32_t i = 0; i < var_info_->var_num; i++)
        {
            var_info_->update_last[i] = input.update_last[i];
        }
        return true;
    }
    return false;
}

/*
函数名:Init
功能描述:初始化通用求解器,导入待求解变量的先验信息,分配求解器所需的空间
*/
bool GeneralSolver::Init(const SolverInput &input)
{
    if (!SetSolverInput(input))
    {
        return false;
    }
    result_ = new(std::nothrow) Result(solver_config_.top_num, input.var_num);
    if (result_ == nullptr)
    {
        OP_LOGW(OP_NAME, "Create result_ failed.");
        return false;
    }
    return true;
}

/*
函数名:UpdateCurVarVal
功能描述:更新cur_var中某个待求解变量的值,并同步更新不等式约束的值
输入参数:
  value:待求解变量被更新成为的值
  idx:更新的待求解变量的下标
*/
bool GeneralSolver::UpdateCurVarVal(uint64_t value, int32_t idx)
{
    if (idx < 0 || idx >= var_info_->var_num) {
        return false;
    }
    var_info_->cur_vars[idx] = value;
    UpdateLeqs(var_info_->cur_vars, idx, cons_info_->leqs);
    return true;
}

/*
函数名:SearchVars
功能描述:用于判断某组解是否曾被搜索过
*/
bool GeneralSolver::SearchVars(uint64_t *vars) const
{
    Node *cur_node = visited_node_->GetVarVal(vars);
    if (cur_node != nullptr) {
        return cur_node->searched;
    }
    return false;
}

/*
函数名:CheckValid
功能描述:用于判断cur_var所对应的解是否为可行解
*/
bool GeneralSolver::CheckValid() const
{
    for (int32_t i = 0; i < cons_info_->leq_num; i++)
    {
        if (cons_info_->leqs[i] > 0)
        {
            return false;
        }
    }
    return true;
}

void GeneralSolver::ResetMomentum()
{
    for (int32_t i = 0; i < var_info_->var_num; i++)
    {
        momentum_info_->is_valid[i] = false;
    }
}

/*
函数名:Initialize
功能描述:用于在每一轮迭代开始执行前进行初始化操作
  在此过程中会重置var_info_中的部分参数
  并根据当前状态的cur_vars信息更新不等式约束值
输入参数:
  iter:迭代轮次
*/
void GeneralSolver::Initialize(int32_t iter)
{
    var_info_->chosen_var_idx = -1;
    UpdateLeqs(var_info_->cur_vars, -1, cons_info_->leqs);
    is_feasible_ = CheckValid();
    has_feasible_ = has_feasible_ || is_feasible_;
    for (int32_t i = 0; i < var_info_->var_num; i++)
    {
        var_info_->history_vars[i] = (iter == 1) ? (var_info_->cur_vars[i]) : (var_info_->rec_vars[i]);
        var_info_->rec_vars[i] = var_info_->cur_vars[i];
    }
}

/*
函数名:GetLocality
功能描述:用来检测定域操作过程中所选变量的优先级
输入参数:
  idx:变量的下标
  update_direction:变量在当前位置的下降方向
输出参数:
  Locality类型的优先级指标
*/
Locality GeneralSolver::GetLocality(int32_t idx, UpdateDirection update_direction)
{
    UpdateDirection cur_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);
    if (CheckValid())
    {
        return Locality::GLOBALVALID;
    }
    else if (CheckLocalValid(cons_info_->leqs, idx))
    {
        return Locality::LOCALVALID;
    }
    else if (GetValue(update_direction) * GetValue(cur_direction) < 0)
    {
        return (var_info_->cur_vars[idx] != var_info_->history_vars[idx]) ? Locality::CROSSREGION : Locality::REJECT;
    }
    return Locality::INVALID;
}

/*
函数名:GetCoarseLoc
功能描述:
  定域过程中的变量粗调,大致确定变量的落点信息
  该函数会沿不等式约束的下降方向进行二分搜索
  最终会输出一个位于约束边界/可行域边界的候选落点
输入参数:
  update_info:变量的更新信息,包括下标(idx),下降方向(update_direction)等指标
  step:变量的更新步长
  cur_locality:粗调过程中确定的定域优先级
*/
bool GeneralSolver::GetCoarseLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality)
{
    uint64_t update_value;

    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return false;
    }
    uint64_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    do
    {
        step = (step == 0) ? 1 : (step << 1);
        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step, update_direction);
        UpdateCurVarVal(update_value, idx);
        cur_locality = GetLocality(idx, update_direction);
        var_info_->cur_vars[idx] = var_info_->rec_vars[idx];
        if (cur_locality <= Locality::CROSSREGION)
        {
            step = ((cur_locality == Locality::CROSSREGION) && (step != 1)) ? (step >> 1) : step;
            break;
        }
    } while (step < thres);
    update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step, update_direction);
    UpdateCurVarVal(update_value, idx);
    return thres != 0;
}

/*
函数名:GetFineLoc
功能描述:
  定域过程中的变量精调,细致地确定变量的落点
  后验知识表明约束边界的解相对更好,因此尝试寻找位于边界的可行解
  该函数会在粗调所得的大致落点附近搜索,寻找不等式约束的边界点
*/
bool GeneralSolver::GetFineLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality)
{
    uint64_t update_value;
    Locality rec_locality;

    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return false;
    }
    UpdateDirection update_direction = update_info->update_direction;
    if (GetLocality(idx, update_direction) <= Locality::LOCALVALID)
    {
        while (step > 1)
        {
            step >>= 1;
            update_value = var_info_->cur_vars[idx] - GetValue(update_direction) * step;
            UpdateCurVarVal(update_value, idx);
            rec_locality = GetLocality(idx, update_direction);
            if (rec_locality > Locality::CROSSREGION) {
                update_value = var_info_->cur_vars[idx] + GetValue(update_direction) * step;
            } else {
                update_value = var_info_->cur_vars[idx];
            }
            UpdateCurVarVal(update_value, idx);
        }
        cur_locality = GetLocality(idx, update_direction);
    }
    return true;
}

/*
函数名:GetPeerLoc
功能描述:
  在定域过程中搜索某个解的对端解
  对端解:若当前解位于约束边界,则对端解位于可行域另一侧的约束边界
  当某个方向的可行解最优但曾被搜索过,该函数可以跨越可行域寻找另一个可行域边界上的解,跳出局部最优
*/
bool GeneralSolver::GetPeerLoc(const UpdateInfo *update_info, Locality &cur_locality)
{
    uint64_t left_value;
    uint64_t right_value;
    uint64_t mid_value;
    Locality rec_locality;
    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return false;
    }
    uint64_t rec_value = var_info_->cur_vars[idx];
    UpdateDirection update_direction = update_info->update_direction;
    UpdateCurVarVal((update_direction == UpdateDirection::NEGATIVE) ? var_info_->lower_bound[idx] : var_info_->upper_bound[idx], idx);
    rec_locality = GetLocality(idx, update_direction);
    if (rec_locality <= Locality::LOCALVALID)
    {
        var_info_->cur_vars[idx] = rec_value;
    }
    else
    {
        left_value = (update_direction == UpdateDirection::POSITIVE) ? (rec_value + 1) : 1;
        right_value = (update_direction == UpdateDirection::POSITIVE) ? (var_info_->upper_bound[idx]) : (rec_value - var_info_->lower_bound[idx]);
        while (left_value < right_value)
        {
            mid_value = (left_value + right_value) >> 1;
            UpdateCurVarVal(mid_value, idx);
            rec_locality = GetLocality(idx, update_direction);
            if (rec_locality > Locality::LOCALVALID)
            {
                left_value = mid_value + 1;
            }
            else
            {
                right_value = mid_value;
            }
        }
        var_info_->cur_vars[idx] = left_value;
        cur_locality = Locality::ALTERNATIVE;
    }
    return true;
}

/*
函数名:UpdateMomentum
功能描述:
  更新算法中的动量信息，以帮助算法更快地收敛到最优解
输入参数:
  idx:更新动量信息的变量索引。
  update_value:更新值。
  cur_locality:当前的LOCALITY信息
输出参数:
  best_locality:当前找到的最好的LOCALITY信息
*/
void GeneralSolver::UpdateMomentum(int32_t idx, double update_value, Locality cur_locality, Locality &best_locality)
{
    if (!SearchVars(var_info_->cur_vars))
    {
        if (cur_locality < best_locality)
        {
            ResetMomentum();
            best_locality = cur_locality;
        }
        if (cur_locality == best_locality)
        {
            var_info_->target_val[idx] = var_info_->cur_vars[idx];
            momentum_info_->is_valid[idx] = true;
            momentum_info_->cur_value[idx] = update_value;
        }
    }
}

/*
函数名:GetBestChoice
功能描述:
  根据动量信息选择最佳变量进行更新
  使用idx遍历所有变量,检查动量信息是否有效,并计算动量值
  选取动量值最佳的变量作为输出
输出参数:
  bool类型参数,用于标记是否找到了最佳变量
*/
bool GeneralSolver::GetBestChoice()
{
    bool better_choice;
    bool make_sense;
    double cur_value = 0.0;
    bool has_chosen = false;
    for (int32_t idx = 0; idx < var_info_->var_num; idx++)
    {
        if (momentum_info_->is_valid[idx])
        {
            momentum_info_->momentum[idx] *= solver_config_.momentum_factor;
            momentum_info_->momentum[idx] += momentum_info_->cur_value[idx] * (1 - solver_config_.momentum_factor);
            better_choice = !has_chosen || momentum_info_->momentum[idx] > cur_value;
            make_sense = var_info_->cur_vars[idx] != var_info_->target_val[idx];
            if (better_choice && make_sense)
            {
                var_info_->chosen_var_idx = idx;
                has_chosen = true;
                cur_value = momentum_info_->momentum[idx];
            }
        }
    }
    return var_info_->chosen_var_idx != -1;
}

/*
函数名:UpdateBestVar
功能描述:
  根据chosen_var_idx的值对变量进行更新
  并调整momentum_info_中其他变量的动量信息
*/
bool GeneralSolver::UpdateBestVar()
{
    for (int32_t idx = 0; idx < var_info_->var_num; idx++)
    {
        if (var_info_->chosen_var_idx == idx)
        {
            var_info_->cur_vars[idx] = var_info_->target_val[idx];
        }
        else
        {
            momentum_info_->momentum[idx] = 0;
        }
        momentum_info_->is_valid[idx] = false;
    }
    UpdateLeqs(var_info_->cur_vars, -1, cons_info_->leqs);
    return true;
}

/*
函数名:LocateLoc
功能描述:
  在需要精调变量落点的情况下寻找变量的落点
  该函数会根据cur_locality和best_locality确定是否需要精调
  若需要,则会调用GetFineLoc函数进行精调,并根据精调结果判断是否要取对端解
  最后根据预期落点更新动量信息
*/
bool GeneralSolver::LocateLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality, Locality &best_locality)
{
    int32_t idx = update_info->idx;
    double init_obj = update_info->init_obj;
    if (cur_locality <= best_locality)
    {
        GetFineLoc(update_info, step, cur_locality);
        if (!solver_config_.simple_ver && SearchVars(var_info_->cur_vars))
        {
            GetPeerLoc(update_info, cur_locality);
        }
        double update_value = init_obj - GetSmoothObj(var_info_->cur_vars);
        UpdateMomentum(idx, update_value, cur_locality, best_locality);
        return true;
    }
    return false;
}

/*
函数名:TryLocate
功能描述:
  尝试对特定变量进行定域操作
  若该更新该变量有希望走入可行域,则会使用GetCoarseLoc函数进行粗调
  根据粗调结果判断是否需要精调,若需要则调用LocateLoc函数进行精调
输入参数:
  idx:变量的索引
  init_idx:变量在当前位置的初始目标函数值
  best_locality:当前找到的最好的LOCALITY信息
*/
bool GeneralSolver::TryLocate(int32_t idx, double init_obj, Locality &best_locality)
{
    Locality cur_locality;
    uint64_t step = 0;
    UpdateDirection update_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);
    if (update_direction != UpdateDirection::NONE)
    {
        uint64_t neg_thres = var_info_->cur_vars[idx] - var_info_->lower_bound[idx];
        uint64_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];
        uint64_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;
        UpdateInfo *update_info = new(std::nothrow) UpdateInfo(idx, thres, update_direction, init_obj);
        if (update_info == nullptr)
        {
            OP_LOGW(OP_NAME, "Create update_info failed.");
            return false;
        }
        if (GetCoarseLoc(update_info, step, cur_locality))
        {
            if (!LocateLoc(update_info, step, cur_locality, best_locality))
            {
                delete update_info;
                UpdateCurVarVal(var_info_->rec_vars[idx], idx);
                return false;
            }
            UpdateCurVarVal(var_info_->rec_vars[idx], idx);
        }
        delete update_info;
    }
    return true;
}

/*
函数名:LocateRegion
功能描述:
  定域操作,用于实现可行域外的变量更新
  当变量位于可行域外时,由不等式约束驱动变量进行调整
  使用TryLocate函数确定变量的落点信息
  优先检测update_last为false的变量,在不存在可行的定域解时检测update_last为true的变量
  寻找目标函数更优的落点
*/
bool GeneralSolver::LocateRegion()
{
    OP_LOGD(OP_NAME, "Infeasible solution, start locating feasible region.");
    Locality best_locality = Locality::REJECT;
    double init_obj = GetSmoothObj(var_info_->cur_vars);
    for (int32_t idx = 0; idx < var_info_->var_num; idx++)
    {
        if (!var_info_->update_last[idx])
        {
            TryLocate(idx, init_obj, best_locality);
        }
    }
    if (has_feasible_ || best_locality == Locality::REJECT)
    {
        for (int32_t idx = 0; idx < var_info_->var_num; idx++)
        {
            if (var_info_->update_last[idx])
            {
                TryLocate(idx, init_obj, best_locality);
            }
        }
    }
    if (best_locality == Locality::REJECT || !GetBestChoice())
    {
        OP_LOGW(OP_NAME, "There is no nonredundant variables that can approximate the feasible region.");
        return false;
    }
    UpdateBestVar();
    OP_LOGD(OP_NAME, "Located feasible region successfully.");
    return true;
}

/*
函数名:GetTunePriority
功能描述:
  确定微调过程中某个待求解变量的优先级
输入参数:
  idx:待求解变量的下标
  rec_obj:本轮迭代前的初始目标函数值
输出参数:
  cur_obj:微调后变量的目标函数值
*/
TunePriority GeneralSolver::GetTunePriority(int32_t idx, double rec_obj, double &cur_obj)
{
    cur_obj = GetSmoothObj(var_info_->cur_vars);
    int64_t last_update = var_info_->rec_vars[idx] - var_info_->history_vars[idx];
    int64_t next_update = var_info_->cur_vars[idx] - var_info_->rec_vars[idx];
    if (last_update * next_update < 0)
    {
        return TunePriority::TABU;
    }
    else if (cur_obj <= rec_obj)
    {
        if (CheckLocalValid(cons_info_->leqs, idx))
        {
            return (cur_obj < rec_obj) ? TunePriority::HARMLESS : TunePriority::DILATED;
        }
        else
        {
            return (cur_obj < rec_obj) ? TunePriority::NORMAL : (solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER);
        }
    }
    return solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;
}

/*
函数名:SearchLoc
功能描述:
  沿着指定的更新方向进行探索,检查是否有机会取到更优的可行解
  该函数会探索至多solver_config_.search_length步,若存在更优的可行解则会进行标记
输入参数:
  update_info:变量的更新信息
输出参数:
  step:取得更优可行解时的步长
  cur_obj:微调后变量的目标函数值
  cur_priority:微调后变量的优先级
*/
bool GeneralSolver::SearchLoc(const UpdateInfo *update_info, uint64_t &step, double &cur_obj, TunePriority &cur_priority)
{
    TunePriority rec_priority;
    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return false;
    }
    uint64_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    double init_obj = update_info->init_obj;
    while (step < Min(thres, solver_config_.search_length))
    {
        step++;
        UpdateCurVarVal(var_info_->rec_vars[idx] + GetValue(update_direction) * step, idx);
        rec_priority = GetTunePriority(idx, init_obj, cur_obj);
        if (rec_priority <= cur_priority)
        {
            cur_priority = rec_priority;
            break;
        }
    }
    UpdateCurVarVal(var_info_->rec_vars[idx], idx);
    return rec_priority == cur_priority;
}

/*
函数名:GetHarmlessLoc
功能描述:
  当且仅当存在一个目标函数更优的可行解时称求解器能找到无损的局部最优解
  该函数尝试在搜索范围内检查所有的可行解,寻找最优的无损局部最优解
输入参数:
  update_info:变量的更新信息
输出参数:
  step:取得更优可行解时的步长
  cur_obj:微调后无损局部最优解的目标函数值
*/
bool GeneralSolver::GetHarmlessLoc(const UpdateInfo *update_info, uint64_t &step, double &cur_obj)
{
    double rec_obj;
    int32_t update_value;
    TunePriority rec_priority;
    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return false;
    }
    uint64_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    var_info_->cur_vars[idx] = var_info_->rec_vars[idx];
    while (step < thres)
    {
        step = solver_config_.simple_ver ? (step == 0 ? 1 : (step << 1)) : (step + 1);
        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step, update_direction);
        UpdateCurVarVal(update_value, idx);
        rec_priority = GetTunePriority(idx, cur_obj, rec_obj);
        if (rec_priority != TunePriority::HARMLESS)
        {
            step = solver_config_.simple_ver ? (step >> 1) : (step - 1);
            break;
        }
        cur_obj = rec_obj;
    }
    return true;
}

/*
函数名:GetDilatedLoc
功能描述:
  当且仅当存在一个目标函数不变但更接近可行域边界的可行解时称求解器能找到膨胀局部最优解
  该函数沿着缓存占用边界更新变量,寻找更新方向上最接近可行域边界的膨胀局部最优解
输入参数:
  update_info:变量的更新信息
输出参数:
  step:取得更优可行解时的步长
*/
bool GeneralSolver::GetDilatedLoc(const UpdateInfo *update_info, uint64_t &step)
{
    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx illegal.");
        return false;
    }
    uint64_t update_value;
    uint64_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    double cur_obj;
    double cur_cons;
    double init_obj = update_info->init_obj;
    double init_cons = update_info->init_cons;
    double pre_cons = init_cons;
    while (step < thres)
    {
        step = solver_config_.simple_ver ? (step == 0 ? 1 : (step << 1)) : (step + 1);
        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step, update_direction);
        UpdateCurVarVal(update_value, idx);
        cur_obj = GetSmoothObj(var_info_->cur_vars);
        cur_cons = GetBuffCost(var_info_->cur_vars);
        if (!CheckLocalValid(cons_info_->leqs, idx) || (!IsEqual(init_obj, cur_obj)) || (cur_cons > pre_cons))
        {
            step = solver_config_.simple_ver ? (step >> 1) : (step - 1);
            break;
        }
        pre_cons = cur_cons;
    }
    return true;
}

/*
函数名:UpdateMomentum
功能描述:
  是前一个UpdateMomentum的重载
  前一个UpdateMomentum函数用于更新定域过程中的动量信息
  本函数用于更新微调过程中的动量信息
*/
void GeneralSolver::UpdateMomentum(int32_t idx, double update_value, TunePriority cur_priority, TunePriority &best_priority)
{
    if (!SearchVars(var_info_->cur_vars))
    {
        if (cur_priority < best_priority)
        {
            ResetMomentum();
            best_priority = cur_priority;
        }
        if (cur_priority == best_priority)
        {
            if (update_value > momentum_info_->cur_value[idx] || !momentum_info_->is_valid[idx])
            {
                var_info_->target_val[idx] = var_info_->cur_vars[idx];
                momentum_info_->is_valid[idx] = true;
                momentum_info_->cur_value[idx] = update_value;
            }
        }
    }
}

/*
函数名:TuneLoc
功能描述:
  根据变量的更新信息对某个变量进行进一步的微调
  根据输入的微调优先级cur_priority选取微调策略对变量进行更新
  若优先级为HARMLESS,则会调用GetHarmlessLoc函数进行无损更新
  若优先级为DILATED,则会调用GetDilatedLoc函数进行膨胀更新
*/
bool GeneralSolver::TuneLoc(const UpdateInfo *update_info, double cur_obj, uint64_t &step, TunePriority &cur_priority, TunePriority &best_priority)
{
    if (cur_priority <= best_priority)
    {
        uint64_t update_value;
        int32_t idx = update_info->idx;
        if ((idx < 0) || (idx >= var_info_->var_num)) {
            OP_LOGE(OP_NAME, "idx illegal.");
            return false;
        }
        UpdateDirection update_direction = update_info->update_direction;
        double init_obj = update_info->init_obj;
        if (cur_priority == TunePriority::HARMLESS)
        {
            GetHarmlessLoc(update_info, step, cur_obj);
        }
        else if (cur_priority == TunePriority::DILATED)
        {
            UpdateDirection cur_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::BUFFER);
            if (GetValue(cur_direction) * GetValue(update_direction) >= 0)
            {
                GetDilatedLoc(update_info, step);
            }
            else
            {
                cur_priority = solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;
            }
        }
        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step, update_direction);
        UpdateCurVarVal(update_value, idx);
        UpdateMomentum(idx, (init_obj - cur_obj), cur_priority, best_priority);
        return true;
    }
    return false;
}

/*
函数名:TryTune
功能描述:
  对某个变量进行微调
  首先利用SearchLoc函数在领域内判断是否存在更优的可行解
  然后根据微调优先级cur_priority选取微调策略对变量进行更新
*/
bool GeneralSolver::TryTune(int32_t idx, UpdateDirection update_direction, double init_obj, double init_cons, TunePriority &best_priority)
{
    uint64_t step = 0;
    uint64_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];
    uint64_t neg_thres = var_info_->cur_vars[idx] - var_info_->lower_bound[idx];
    uint64_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;
    double cur_obj;
    TunePriority cur_priority = (thres > 0) ? best_priority : TunePriority::REFUSE;
    if (thres > 0)
    {
        UpdateInfo *update_info = new(std::nothrow) UpdateInfo(idx, thres, update_direction, init_obj, init_cons);
        if (update_info == nullptr)
        {
            OP_LOGW(OP_NAME, "Create update_info failed.");
            return false;
        }
        if (SearchLoc(update_info, step, cur_obj, cur_priority))
        {
            if (!TuneLoc(update_info, cur_obj, step, cur_priority, best_priority))
            {
                return false;
            }
            UpdateCurVarVal(var_info_->rec_vars[idx], idx);
        }
        delete update_info;
    }
    return cur_priority >= TunePriority::NORMAL;
}

/*
函数名:FineTune
功能描述:
  实现待求解变量的微调操作
  首先沿正方向对变量进行更新,若更新方向上存在更优的可行解则进行微调
  若正方向上不存在更优的可行解或采用高性能版本进行求解,则尝试沿负方向进行更新
*/
bool GeneralSolver::FineTune()
{
    OP_LOGD(OP_NAME, "Feasible solution, start tuning the tilling data.");
    double init_obj = GetSmoothObj(var_info_->cur_vars);
    double init_cons = GetBuffCost(var_info_->cur_vars);
    if (!RecordBestVarVal())
    {
        OP_LOGW(OP_NAME, "Failed to add a solution to the result.");
        return false;
    }
    TunePriority best_priority = TunePriority::TABU;
    for (int32_t idx = 0; idx < var_info_->var_num; idx++)
    {
        if (TryTune(idx, UpdateDirection::POSITIVE, init_obj, init_cons, best_priority) || !solver_config_.simple_ver)
        {
            TryTune(idx, UpdateDirection::NEGATIVE, init_obj, init_cons, best_priority);
        }
    }
    if (!GetBestChoice())
    {
        OP_LOGW(OP_NAME, "Unable to find a valuable update.");
        return false;
    }
    UpdateBestVar();
    OP_LOGD(OP_NAME, "Tuned the tiling data successfully.");
    return true;
}

bool GeneralSolver::RecordBestVarVal()
{
    if (is_feasible_)
    {
        double obj = GetObj(var_info_->cur_vars);
        double cons = GetBuffCost(var_info_->cur_vars);
        return result_->AddVarVal(var_info_->cur_vars, obj, cons);
    }
    return false;
}

/*
函数名:Run
功能描述:
  通用求解器求解函数
  算法会迭代solver_config_.iterations次
  在每轮迭代中根据当前的变量值选取定域或微调策略对变量进行更新
输出参数:
  solution_num:uint32_t类型的参数,用来输出实际得到的解的个数
  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,求解算法获取到的可行解放入该空间
*/
bool GeneralSolver::Run(int32_t &solution_num, uint64_t *solutions)
{
    Node* cur_node;
    uint64_t iter = 1;
    has_feasible_ = false;
    while (iter <= solver_config_.iterations)
    {
        Initialize(iter);
        OP_LOGD(OP_NAME, "iter : %lu", iter);
        DisplayVarVal(var_info_->cur_vars);
        if (!is_feasible_)
        {
            if (!LocateRegion())
            {
                OP_LOGW(OP_NAME, "The locating process cannot find more valuable updates, triggering an early stop.");
                break;
            }
        }
        else
        {
            if (SearchVars(var_info_->cur_vars))
            {
                OP_LOGW(OP_NAME, "Searched a feasible solution again, triggering an early stop.");
                break;
            }
            cur_node = visited_node_->GetVarVal(var_info_->cur_vars);
            if (cur_node == nullptr) {
                OP_LOGW(OP_NAME, "Failed to construct a new solution node, terminating the iteration.");
                break;
            }
            cur_node->searched = true;
            if (!FineTune())
            {
                break;
            }
        }
        iter++;
    }
    result_->GetResult(solution_num, solutions);
    return solution_num > 0;
}

int32_t GeneralSolver::GetVarNum() const
{
    return var_info_->var_num;
}

bool GetPlatformInfo(MMTilingData &tiling_data, gert::TilingContext *context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer platformInfoPtr is null.");
    return false;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  uint64_t l0a_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
  uint64_t l0b_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
  uint64_t l0c_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
  uint64_t l1_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
  uint64_t l2_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
  if ((aivNum == 0) || (aicNum == 0) || (l0a_size == 0) || (l0b_size == 0) || (l0c_size == 0) || (l1_size == 0) || (l2_size == 0)) {
    OP_LOGE(OP_NAME, "Get incorrect platform value.");
    return false;
  } 
  OP_LOGD(OP_NAME, "PlatformInfo is valid.");
  tiling_data.set_block_dim(ascendcPlatform.GetCoreNumAiv());
  OP_LOGD(OP_NAME, "Set block dim to %d.", tiling_data.get_block_dim());
  tiling_data.set_l0a_size(l0a_size);
  OP_LOGD(OP_NAME, "Set l0a_size to %d.", tiling_data.get_l0a_size());
  tiling_data.set_l0b_size(l0b_size);
  OP_LOGD(OP_NAME, "Set l0b_size to %d.", tiling_data.get_l0b_size());
  tiling_data.set_l0c_size(l0c_size);
  OP_LOGD(OP_NAME, "Set l0c_size to %d.", tiling_data.get_l0c_size());
  tiling_data.set_l1_size(l1_size);
  OP_LOGD(OP_NAME, "Set l1_size to %d.", tiling_data.get_l1_size());
  tiling_data.set_l2_size(l2_size);
  OP_LOGD(OP_NAME, "Set l2_size to %d.", tiling_data.get_l2_size());

  return true;
}

// 根据tiling case id创建对应的L0TileSolver子类
class case1L0TileSolver : public L0TileSolver {
  public:
    // 构造函数，接受L0TileInput类型的参数 input
    explicit case1L0TileSolver(L0TileInput &input) : L0TileSolver(input) {};
    // 成员函数声明
    void Setl1_size(const uint32_t &value) { l1_size_ = value; }
    void Setl2_size(const uint32_t &value) { l2_size_ = value; }
    void Setl0a_size(const uint32_t &value) { l0a_size_ = value; }
    void Setl0b_size(const uint32_t &value) { l0b_size_ = value; }
    void Setl0c_size(const uint32_t &value) { l0c_size_ = value; }
    bool CheckBufferUseValid() override;
  // 定义private类型的成员变量
  private:
    uint32_t l1_size_;
    uint32_t l2_size_;
    uint32_t l0a_size_;
    uint32_t l0b_size_;
    uint32_t l0c_size_;
};
  /*
  对创建的L0TileSolver子类的成员函数CheckBufferUseValid进行定义，判断缓存占用是否合法
  i从0开始到待求解l0相关变量的数量，依次递增循环
  @return 如果代求解l0变量的数量为空或i>=待求解l0相关变量的数量，返回false。如果在DEBUG域内输出提示信息
  定义uint32_t类型变量：第i个待求解l0变量的字符串名称，值为第i个待求解l0相关变量的值
  */
bool case1L0TileSolver::CheckBufferUseValid() {
  if ((input_.l0_vars == nullptr) || (0 >= 3)) {
  #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0_vars is nullptr or overflow");
  #endif
      return false;
   }
  uint32_t basem_size = input_.l0_vars[0].value;
  if ((input_.l0_vars == nullptr) || (1 >= 3)) {
  #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0_vars is nullptr or overflow");
  #endif
      return false;
   }
  uint32_t basen_size = input_.l0_vars[1].value;
  if ((input_.l0_vars == nullptr) || (2 >= 3)) {
  #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0_vars is nullptr or overflow");
  #endif
      return false;
   }
  uint32_t basek_size = input_.l0_vars[2].value;
  /*
  遍历buffer_use_map_中的所有缓存占用表达式，将其转换为字符串类型，赋值给对应的设备类型
  将每种设备类型缓存占用计算值和其设定值对比，如果缓存占用计算值大于设定值，返回false
  @return 遍历所有设备类型后，如果所有设备类型缓存占用计算值都小于设定值，返回true
  */
  uint32_t l0a_size =  (4 * basek_size * basem_size);
  if (l0a_size > l0a_size_) {
    return false;
   };
  uint32_t l0b_size =  (4 * basek_size * basen_size);
  if (l0b_size > l0b_size_) {
    return false;
   };
  uint32_t l0c_size =  (4 * basem_size * basen_size);
  if (l0c_size > l0c_size_) {
    return false;
   };
  return true;
}
// 根据tiling case id创建对应的L2TileSolver子类
class case1L2TileSolver : public L2TileSolver {
  public:
    // 构造函数，接受L2TileInput类型的参数 input
    explicit case1L2TileSolver(L2TileInput &input, MMTilingData &tiling_data) : L2TileSolver(input), tiling_data_(tiling_data) {};
    // 成员函数声明
    uint64_t GetL2Use() override;
    bool IsClash(const uint32_t idx) override;
    MMTilingData tiling_data_;
};
/*
对创建的L2TileSolver子类的成员函数GetL2Use进行定义，计算L2的可用内存大小
用input_存储输入数据，并将输入数据对应赋值给tilem_size, tilen_size, k_size
使用公式计算L2的可用内存大小l2_size
@return 返回L2可用内存的大小l2_size
*/
uint64_t case1L2TileSolver::GetL2Use() {
  uint64_t tilem_size = input_.l2_vars[0].value;
  uint64_t tilen_size = input_.l2_vars[1].value;
  uint64_t k_size = tiling_data_.get_k_size();
  uint64_t l2_size = (((tilem_size + tilen_size) * 2 * k_size) + (2 * tilem_size * tilen_size));
  return l2_size;
}
/*
对创建的L2TileSolver子类的成员函数IsClash进行定义，用于检测是否存在读冲突
@param idx 输入参数为idx，用于获取当前tile的编号
@return 如果满足条件blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0,则存在读冲突，返回true
用公式计算尾块个数blocknum_tail
@return 如果满足条件blocknum_tail % (used_corenum_ / 2) == 0,则存在读冲突，返回true
@return 如果以上两个条件均不满足，说明不存在读冲突，返回false
*/
bool case1L2TileSolver::IsClash(const uint32_t idx) {
  if ((used_corenum_ <= 1) || (used_corenum_ % 2 !=0)) {
    return false;
  }
  if (blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0) {
    return true;
  }
  auto blocknum_tail = total_blocknum_[idx] - (tilenum_[idx] - 1) * blocknum_per_tile_[idx];
  if (blocknum_tail % (used_corenum_ / 2) == 0) {
    return true;
  }
  return false;
}
/*
用户可以在派生类中重载Run函数,构造自定义的求解算法,即
  void bool Run(int32_t &solution_num, uint64_t *solutions) override;
其中:
  solution_num:int32_t类型的参数,用来输出实际得到的解的个数
  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,算法将可行解放入该空间
Run函数可以使用下述函数辅助求解:
  bool CheckValid()
    用于检测当前解是否为可行解
  bool UpdateCurVarVal(uint64_t value, int32_t idx)
    将下标为idx的待求解变量改为value,同时更新cons_info_->leqs中的值
  bool RecordBestVarVal()
    待求解变量的当前值所对应的目标函数寻优
Run函数可以使用下述参数辅助求解:
  cons_info_->leqs, double类型的数组, 用于记录不等式约束的函数值, 其下标含义如下:
    cons_info_->leqs[0] = ((4 * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + (4 * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) - l1_size)
    cons_info_->leqs[1] = ((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align) - k_size)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1(SolverConfig& config, MMTilingData& tiling_data) : GeneralSolver(config) {
            block_dim = tiling_data.get_block_dim();
            k_size = tiling_data.get_k_size();
            m_size = tiling_data.get_m_size();
            n_size = tiling_data.get_n_size();
            l1_size = tiling_data.get_l1_size();
            basem_size = tiling_data.get_basem_size();
            basen_size = tiling_data.get_basen_size();
            basek_size = tiling_data.get_basek_size();
            tilem_size = tiling_data.get_tilem_size();
            tilen_size = tiling_data.get_tilen_size();
            k_size = ((k_size + 256 - 1) / 256) * 256;
        }

        double GetObj(uint64_t* vars) override;
        double GetSmoothObj(uint64_t* vars) override;
        double GetBuffCost(uint64_t* vars) override;
        bool CheckLocalValid(double* leqs, int32_t idx) override;
        void DisplayVarVal(uint64_t* vars) override;
        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs) override;
        double GetBuffDiff(uint64_t* vars, double* weight) override;
        double GetLeqDiff(uint64_t* vars, double* weight) override;
        double Getl1_sizeCost(uint64_t* vars);
        double GetSmoothl1_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, MMTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, MMTilingData& tiling_data);
    private:
        const int64_t stepka_size_div_align_idx = 0;
        const int64_t stepkb_size_base_idx = 1;
        uint64_t block_dim;
        uint64_t k_size;
        uint64_t m_size;
        uint64_t n_size;
        uint64_t l1_size;
        uint64_t basem_size;
        uint64_t basen_size;
        uint64_t basek_size;
        uint64_t tilem_size;
        uint64_t tilen_size;
};
/*
函数名:Getl1_sizeCost(重要函数)
功能描述:
  根据待求解变量值l1_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1::Getl1_sizeCost(uint64_t* vars)
{
    double stepka_size_div_align = static_cast<double>(vars[stepka_size_div_align_idx]);
    double stepkb_size_base = static_cast<double>(vars[stepkb_size_base_idx]);
    return ((4 * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + (4 * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) - l1_size);
}

/*
函数名:GetSmoothl1_sizeCost(重要函数)
功能描述:
  根据待求解变量值l1_size的平滑化缓存占用信息
  与Getl1_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1::GetSmoothl1_sizeCost(uint64_t* vars)
{
    double stepka_size_div_align = static_cast<double>(vars[stepka_size_div_align_idx]);
    double stepkb_size_base = static_cast<double>(vars[stepkb_size_base_idx]);
    return ((4 * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + (4 * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) - l1_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1::GetObj(uint64_t* vars)
{
    double stepka_size_div_align = static_cast<double>(vars[stepka_size_div_align_idx]);
    double stepkb_size_base = static_cast<double>(vars[stepkb_size_base_idx]);
    double AIC_FIXPIPE = ((double)(1)/(double)(8) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (tilem_size * tilen_size));
    OP_LOGD(OP_NAME, "AIC_FIXPIPE = %f", AIC_FIXPIPE);
    double AIC_MAC = ((double)(1)/(double)(4096) * (k_size * k_size) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (basek_size * tilem_size * tilen_size));
    OP_LOGD(OP_NAME, "AIC_MAC = %f", AIC_MAC);
    double AIC_MTE2 = ((((((double)(1)/(double)(16) * Max(1, (256 / (basen_size))) * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + 210) * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * pow(2, (-1 * stepkb_size_base)) * stepka_size_div_align / (Max(16, basek_size))) + ((double)(1)/(double)(16) * Max(1, (256 / (Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align))) * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) + 210) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * k_size * m_size * n_size / (Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align * tilem_size * tilen_size));
    OP_LOGD(OP_NAME, "AIC_MTE2 = %f", AIC_MTE2);
    return Max(Max(AIC_FIXPIPE, AIC_MAC), AIC_MTE2);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1::GetSmoothObj(uint64_t* vars)
{
    double stepka_size_div_align = static_cast<double>(vars[stepka_size_div_align_idx]);
    double stepkb_size_base = static_cast<double>(vars[stepkb_size_base_idx]);
    double AIC_FIXPIPE = ((double)(1)/(double)(8) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (tilem_size * tilen_size));
    double AIC_MAC = ((double)(1)/(double)(4096) * (k_size * k_size) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (basek_size * tilem_size * tilen_size));
    double AIC_MTE2 = ((((((double)(1)/(double)(16) * Max(1, (256 / (basen_size))) * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + 210) * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * pow(2, (-1 * stepkb_size_base)) * stepka_size_div_align / (Max(16, basek_size))) + ((double)(1)/(double)(16) * Max(1, (256 / (Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align))) * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) + 210) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * k_size * m_size * n_size / (Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align * tilem_size * tilen_size));
    return Max(Max(AIC_FIXPIPE, AIC_MAC), AIC_MTE2);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1::GetBuffCost(uint64_t* vars)
{
    double l1_size_cost = Getl1_sizeCost(vars);
    return (Min(0, l1_size_cost) * Min(0, l1_size_cost));
}
/*
函数名:GetBuffDiff(重要函数)
功能描述:
  获取缓冲占用加权差分值,计算平滑缓冲占用的差分
  输出的计算公式为sigma_j(delta_{var_i}(g_j(var))) * g_j(var))
  其中g_j为第j个缓冲占用不等式,delta_{var_i}(g_j(var))为g_j(var)沿var_i方向更新一个单位后的变化值
  该函数用于确定变量沿缓冲占用增大的更新方向
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值
*/
double GeneralSolvercase1::GetBuffDiff(uint64_t* vars, double* weight)
{
    double l1_size_cost = GetSmoothl1_sizeCost(vars);
    l1_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return l1_size_cost;
}
/*
函数名:GetLeqDiff(重要函数)
功能描述:
  获取不等式约束的加权差分值,计算平滑的不等式函数的差分,权值为实际不等式函数值
  输出的计算公式为sigma_j(delta_{var_i}(f_j(var))) * f_j(var))
  其中f_j为第j个不等式约束式,delta_{var_i}(f_j(var))为f_j(var)沿var_i方向更新一个单位后的变化值
  该函数用于确定变量从可行域外侧沿不等式边界方向移动的更新方向
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值
*/
double GeneralSolvercase1::GetLeqDiff(uint64_t* vars, double* weight)
{
    double stepka_size_div_align = static_cast<double>(vars[stepka_size_div_align_idx]);
    double stepkb_size_base = static_cast<double>(vars[stepkb_size_base_idx]);
    double l1_size_cost = GetSmoothl1_sizeCost(vars);
    l1_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = ((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align) - k_size);
    leq1_cost *= weight[1] > 0 ? weight[1] : 0;
    return l1_size_cost + leq1_cost;
}
bool GeneralSolvercase1::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == stepka_size_div_align_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0;
    } else if (idx == stepkb_size_base_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0;
    }
    return true;
}

void GeneralSolvercase1::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double stepka_size_div_align = static_cast<double>(vars[stepka_size_div_align_idx]);
    double stepkb_size_base = static_cast<double>(vars[stepkb_size_base_idx]);
    if (idx == stepka_size_div_align_idx) {
        leqs[0] = ((4 * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + (4 * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) - l1_size);
        leqs[1] = ((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align) - k_size);
    } else if (idx == stepkb_size_base_idx) {
        leqs[0] = ((4 * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + (4 * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) - l1_size);
        leqs[1] = ((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align) - k_size);
    } else if (idx == -1) {
        leqs[0] = ((4 * Max(16, basek_size) * basen_size * pow(2, stepkb_size_base)) + (4 * Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * basem_size * stepka_size_div_align) - l1_size);
        leqs[1] = ((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align) - k_size);
    }
}

void GeneralSolvercase1::DisplayVarVal(uint64_t* vars)
{
    uint64_t stepka_size_div_align = vars[stepka_size_div_align_idx];
    uint64_t stepkb_size_base = vars[stepkb_size_base_idx];
    OP_LOGD(OP_NAME, "stepka_size = %lu", static_cast<uint64_t>((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align)));
    OP_LOGD(OP_NAME, "stepkb_size = %lu", static_cast<uint64_t>((Max(16, basek_size) * pow(2, stepkb_size_base))));
}

void GeneralSolvercase1::MapVarVal(uint64_t* vars, MMTilingData& tiling_data)
{
    uint64_t stepka_size_div_align = vars[stepka_size_div_align_idx];
    uint64_t stepkb_size_base = vars[stepkb_size_base_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1 is:");
    tiling_data.set_stepka_size(static_cast<uint64_t>((Max(256, (Max(16, basek_size) * pow(2, stepkb_size_base))) * stepka_size_div_align)));
    OP_LOGD(OP_NAME, "stepka_size = %u", tiling_data.get_stepka_size());
    tiling_data.set_stepkb_size(static_cast<uint64_t>((Max(16, basek_size) * pow(2, stepkb_size_base))));
    OP_LOGD(OP_NAME, "stepkb_size = %u", tiling_data.get_stepkb_size());
}

void GeneralSolvercase1::GetResult(int32_t solution_num, uint64_t* solution, MMTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "l1_size = %ld", static_cast<uint64_t>(Getl1_sizeCost(solution) + l1_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1Impl : public TilingCaseImpl {
 protected:
/*
定义L0求解器调用函数
@param tiling_data 输入参数为tiling_data
定义求解器接受的输入为L0TileInput类型结构体变量，并将结构体的成员变量赋值为对应值
定义basem_size为L0Var类型的结构体变量，并将相关成员变量赋值为对应值
定义basen_size为L0Var类型的结构体变量，并将相关成员变量赋值为对应值
定义solver为case0L0TileSolver类型的结构体变量，并将此结构体变量的成员变量L1_, L2_, L0A_, L0B_, L0C_, CORENUM_赋值为tiling_data中的设定值
@return 执行L0TileSolver类的主要流程solver.run()，如果solver.run()的返回值不为true,则l0求解器执行失败，函数返回false
用output指针指向solver的输出
i从0开始到L0相关变量个数依次递增
@return 如果output为空指针或者i>=L0相关的变量个数，则函数返回false，否则将basem_size设置为output的第i个值
@return 若上述没有触发返回false的场景，则函数返回true
*/
  bool ExecuteL0Solver(MMTilingData& tiling_data) {
    L0TileInput l0_input;
    l0_input.l0_vars = new(std::nothrow) L0Var[3];
    l0_input.size = 3;
    l0_input.core_num = tiling_data.get_block_dim();
    L0Var basem_size;
    basem_size.max_value = tiling_data.get_m_size();
    basem_size.bind_multicore = false;
  basem_size.align = 16;
    basem_size.prompt_align = 16;
    l0_input.l0_vars[0] = basem_size;
    L0Var basen_size;
    basen_size.max_value = tiling_data.get_n_size();
    basen_size.bind_multicore = false;
  basen_size.align = 16;
    basen_size.is_innermost = true;
    basen_size.prompt_align = 16;
    l0_input.l0_vars[1] = basen_size;
    L0Var basek_size;
    basek_size.max_value = tiling_data.get_k_size();
    basek_size.bind_multicore = false;
  basek_size.align = 16;
    basek_size.prompt_align = 256;
    l0_input.l0_vars[2] = basek_size;
    case1L0TileSolver solver(l0_input);
    solver.Setl1_size(tiling_data.get_l1_size());
    solver.Setl2_size(tiling_data.get_l2_size());
    solver.Setl0a_size(tiling_data.get_l0a_size());
    solver.Setl0b_size(tiling_data.get_l0b_size());
    solver.Setl0c_size(tiling_data.get_l0c_size());
    if (!solver.Run()) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0 solver run failed");
      #endif
      return false;
    }
    uint32_t *output = solver.GetOutput();
    if ((output == nullptr) || (0 >= 3)) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "output is nullptr or overflow");
      #endif
      return false;
    }
    tiling_data.set_basem_size(output[0]);
    if ((output == nullptr) || (1 >= 3)) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "output is nullptr or overflow");
      #endif
      return false;
    }
    tiling_data.set_basen_size(output[1]);
    if ((output == nullptr) || (2 >= 3)) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "output is nullptr or overflow");
      #endif
      return false;
    }
    tiling_data.set_basek_size(output[2]);
    return true;
  }
/*
定义L2求解器调用函数
@param tiling_data 输入参数为tiling_data
定义求解器的输入为L2TileInput类型结构体变量，并将结构体的成员变量赋值为对应值
定义tilem_size为L2Var类型的结构体变量，并将相关成员变量赋值为对应值
定义tilen_size为L2Var类型的结构体变量，并将相关成员变量赋值为对应值
定义l2_solver为L2TileSolver类型的结构体变量
计算l2的可用内存大小以及是否存在读冲突
@return 执行L2TileSolver类的主要流程l2_solver.run()，如果l2_solver.run()的返回值不为true,则l2求解器执行失败，函数返回false
用output指针指向l2_solver的输出
i从0开始到L2相关变量个数依次递增
@return 如果output为空指针或者i>=L2相关的变量个数，则函数返回false，否则将tilem_size设置为output的第i个值
@return 若上述没有触发返回false的场景，则函数返回true
*/
  bool ExecuteL2Solver(MMTilingData& tiling_data) {
    L2TileInput l2_input;
    l2_input.l2_vars = new(std::nothrow) L2Var[2];
    l2_input.size = 2;
    l2_input.core_num = tiling_data.get_block_dim();
    l2_input.l2_size = EMPIRIC_L2_SIZE;
    L2Var tilem_size;
    tilem_size.max_value = tiling_data.get_m_size();
    tilem_size.align = 16;
    tilem_size.base_val = tiling_data.get_basem_size();
    l2_input.l2_vars[0] = tilem_size;
    L2Var tilen_size;
    tilen_size.max_value = tiling_data.get_n_size();
    tilen_size.align = 16;
    tilen_size.base_val = tiling_data.get_basen_size();
    l2_input.l2_vars[1] = tilen_size;
    case1L2TileSolver l2_solver(l2_input, tiling_data);
    if (!l2_solver.Run()) {
    #ifdef DEBUG
      OP_LOGW(OP_NAME, "l2 solver run failed");
    #endif
      return false;
    }
    uint32_t *output = l2_solver.GetL2Tile();
    if ((output == nullptr) || (0 >= 2)) {
    #ifdef DEBUG
        OP_LOGW(OP_NAME, "l2_vars is nullptr or overflow");
    #endif
      return false;
  }
    tiling_data.set_tilem_size(output[0]);
    if ((output == nullptr) || (1 >= 2)) {
    #ifdef DEBUG
        OP_LOGW(OP_NAME, "l2_vars is nullptr or overflow");
    #endif
      return false;
  }
    tiling_data.set_tilen_size(output[1]);
    return true;
  }
  bool ExecuteGeneralSolver(MMTilingData& tiling_data) {
    SolverConfig cfg;
    cfg.top_num = cfg_top_num;
    cfg.search_length = cfg_search_length;
    cfg.iterations = cfg_iterations;
    cfg.simple_ver = cfg_simple_ver;
    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);
    OP_LOGD(OP_NAME, "Record a maximum of %lu solutions.", cfg.top_num);
    OP_LOGD(OP_NAME, "The searching range covers %lu unit(s).", cfg.search_length);
    OP_LOGD(OP_NAME, "The maximum number of iterations is %lu.", cfg.iterations);
    if (cfg.simple_ver) {
        OP_LOGD(OP_NAME, "Using high-efficiency version.");
    } else {
        OP_LOGD(OP_NAME, "Using high-performance version.");
    }
    OP_LOGD(OP_NAME, "The momentum factor is %f.", cfg.momentum_factor);

    // 以下参数若未注明是可修改参数,则不建议修改
    uint64_t k_size = tiling_data.get_k_size();
    k_size = ((k_size + 256 - 1) / 256) * 256;
    uint64_t basek_size = tiling_data.get_basek_size();
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 2;
    OP_LOGD(OP_NAME, "The number of variable is %d(stepka_size_div_align, stepkb_size_base), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(0)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>((k_size / (Max(256, (Max(16, basek_size) * pow(2, (log((16 / (Max(16, basek_size)))) / (log(2))))))))), static_cast<uint64_t>((log((k_size / (Max(16, basek_size)))) / (log(2))))};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(0)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {true, true};
    // 初始化解的个数为0
    int32_t solution_num = 0;
    // 为求解器的输出分配内存
    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];
    if (solution == nullptr)
    {
        OP_LOGW(OP_NAME, "Create solution failed.");
        return false;
    }
    // 通用求解器的输入参数
    SolverInput input;
    input.var_num = num_var;
    input.leq_num = num_leq;
    input.cur_vars = init_vars;
    input.upper_bound = upper_bound;
    input.lower_bound = lower_bound;
    input.update_last = update_last;
    OP_LOGD(OP_NAME, "stepka_size_div_align->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "stepkb_size_base->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    GeneralSolvercase1* solver = new(std::nothrow) GeneralSolvercase1(cfg, tiling_data);
    if (solver != nullptr) {
        // 导入通用求解器的输入参数并完成初始化
        OP_LOGD(OP_NAME, "Start initializing the input.");
        if (solver -> Init(input)) {
            // 运行通用求解器并获取算法的解
            OP_LOGD(OP_NAME, "Intialization finished, start running the solver.");
            if (solver -> Run(solution_num, solution)) {
                solver -> GetResult(solution_num, solution, tiling_data);
                delete solver;
                delete[] solution;
                OP_LOGD(OP_NAME, "The solver executed successfully.");
                return true;
            }
            OP_LOGW(OP_NAME, "Failed to find any solution.");
        }
    }
    if (solver != nullptr) {
        delete solver;
    }
    if (solution != nullptr) {
        delete[] solution;
    }
    OP_LOGW(OP_NAME, "The solver executed failed.");
    return false;
  }

  bool DoTiling(MMTilingData &tiling_data) {
    if (!ExecuteL0Solver(tiling_data)) {
        return false;
    }
    if (!ExecuteL2Solver(tiling_data)) {
        return false;
    }
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1 successfully.");

    return true;
  }

  void DoApiTiling(MMTilingData &tiling_data) {
  }
  int Getl1_size(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double stepka_size = tiling_data.get_stepka_size();
    double stepkb_size = tiling_data.get_stepkb_size();

    return ((4 * basem_size * stepka_size) + (4 * basen_size * stepkb_size));
  }

  int Getl2_size(MMTilingData& tiling_data) {
    double k_size = tiling_data.get_k_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return (((tilem_size + tilen_size) * 2 * k_size) + (2 * tilem_size * tilen_size));
  }

  int Getl0a_size(MMTilingData& tiling_data) {
    double basek_size = tiling_data.get_basek_size();
    double basem_size = tiling_data.get_basem_size();

    return (4 * basek_size * basem_size);
  }

  int Getl0b_size(MMTilingData& tiling_data) {
    double basek_size = tiling_data.get_basek_size();
    double basen_size = tiling_data.get_basen_size();

    return (4 * basek_size * basen_size);
  }

  int Getl0c_size(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();

    return (4 * basem_size * basen_size);
  }

  double GetAIC_MTE2(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double block_dim = tiling_data.get_corenum();
    double k_size = tiling_data.get_k_size();
    double m_size = tiling_data.get_m_size();
    double n_size = tiling_data.get_n_size();
    double stepka_size = tiling_data.get_stepka_size();
    double stepkb_size = tiling_data.get_stepkb_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return ((((((double)(1)/(double)(16) * Max(1, (256 / (basen_size))) * basen_size * stepkb_size) + 210) * stepka_size / (stepkb_size)) + ((double)(1)/(double)(16) * Max(1, (256 / (stepka_size))) * basem_size * stepka_size) + 210) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * k_size * m_size * n_size / (stepka_size * tilem_size * tilen_size));
  }

  double GetAIC_FIXPIPE(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double block_dim = tiling_data.get_corenum();
    double m_size = tiling_data.get_m_size();
    double n_size = tiling_data.get_n_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return ((double)(1)/(double)(8) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (tilem_size * tilen_size));
  }

  double GetAIC_MAC(MMTilingData& tiling_data) {
    double basek_size = tiling_data.get_basek_size();
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double block_dim = tiling_data.get_corenum();
    double k_size = tiling_data.get_k_size();
    double m_size = tiling_data.get_m_size();
    double n_size = tiling_data.get_n_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return ((double)(1)/(double)(4096) * (k_size * k_size) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (basek_size * tilem_size * tilen_size));
  }

  double GetPerf(MMTilingData& tiling_data) {
    double basek_size = tiling_data.get_basek_size();
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double block_dim = tiling_data.get_corenum();
    double k_size = tiling_data.get_k_size();
    double m_size = tiling_data.get_m_size();
    double n_size = tiling_data.get_n_size();
    double stepka_size = tiling_data.get_stepka_size();
    double stepkb_size = tiling_data.get_stepkb_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    double AIC_MTE2 = ((((((double)(1)/(double)(16) * Max(1, (256 / (basen_size))) * basen_size * stepkb_size) + 210) * stepka_size / (stepkb_size)) + ((double)(1)/(double)(16) * Max(1, (256 / (stepka_size))) * basem_size * stepka_size) + 210) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * k_size * m_size * n_size / (stepka_size * tilem_size * tilen_size));
    double AIC_FIXPIPE = ((double)(1)/(double)(8) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (tilem_size * tilen_size));
    double AIC_MAC = ((double)(1)/(double)(4096) * (k_size * k_size) * Max(1, (tilem_size * tilen_size / (basem_size * basen_size * block_dim))) * basem_size * basen_size * m_size * n_size / (basek_size * tilem_size * tilen_size));

    return Max(Max(AIC_FIXPIPE, AIC_MAC), AIC_MTE2);
  }

  void GetTilingData(MMTilingData &tiling_data, MMTilingData &to_tiling) {
    to_tiling.set_block_dim(tiling_data.get_block_dim());
    to_tiling.set_k_size(tiling_data.get_k_size());
    to_tiling.set_m_size(tiling_data.get_m_size());
    to_tiling.set_n_size(tiling_data.get_n_size());
    to_tiling.set_l1_size(tiling_data.get_l1_size());
    to_tiling.set_l2_size(tiling_data.get_l2_size());
    to_tiling.set_l0a_size(tiling_data.get_l0a_size());
    to_tiling.set_l0b_size(tiling_data.get_l0b_size());
    to_tiling.set_l0c_size(tiling_data.get_l0c_size());
    to_tiling.set_corenum(tiling_data.get_corenum());
  }
  void SetTilingData(MMTilingData &from_tiling, MMTilingData &tiling_data) {
    tiling_data.set_basek_size(from_tiling.get_basek_size());
    tiling_data.set_basem_size(from_tiling.get_basem_size());
    tiling_data.set_basen_size(from_tiling.get_basen_size());
    tiling_data.set_stepka_size(from_tiling.get_stepka_size());
    tiling_data.set_stepkb_size(from_tiling.get_stepkb_size());
    tiling_data.set_tilem_size(from_tiling.get_tilem_size());
    tiling_data.set_tilen_size(from_tiling.get_tilen_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_MATMUL_OUTPUT1(from_tiling.get_MATMUL_OUTPUT1());
    tiling_data.set_Q1(from_tiling.get_Q1());
    tiling_data.set_basek_loop_num(from_tiling.get_basek_loop_num());
    tiling_data.set_basek_tail_size(from_tiling.get_basek_tail_size());
    tiling_data.set_basem_loop_num(from_tiling.get_basem_loop_num());
    tiling_data.set_basem_tail_size(from_tiling.get_basem_tail_size());
    tiling_data.set_basen_loop_num(from_tiling.get_basen_loop_num());
    tiling_data.set_basen_tail_size(from_tiling.get_basen_tail_size());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_stepka_loop_num(from_tiling.get_stepka_loop_num());
    tiling_data.set_stepka_tail_size(from_tiling.get_stepka_tail_size());
    tiling_data.set_stepka_tail_tile_stepkb_loop_num(from_tiling.get_stepka_tail_tile_stepkb_loop_num());
    tiling_data.set_stepka_tail_tile_stepkb_tail_size(from_tiling.get_stepka_tail_tile_stepkb_tail_size());
    tiling_data.set_stepkb_loop_num(from_tiling.get_stepkb_loop_num());
    tiling_data.set_stepkb_tail_size(from_tiling.get_stepkb_tail_size());
    tiling_data.set_stepkb_tail_tile_basek_loop_num(from_tiling.get_stepkb_tail_tile_basek_loop_num());
    tiling_data.set_stepkb_tail_tile_basek_tail_size(from_tiling.get_stepkb_tail_tile_basek_tail_size());
    tiling_data.set_tilem_loop_num(from_tiling.get_tilem_loop_num());
    tiling_data.set_tilem_tail_size(from_tiling.get_tilem_tail_size());
    tiling_data.set_tilem_tail_tile_basem_loop_num(from_tiling.get_tilem_tail_tile_basem_loop_num());
    tiling_data.set_tilem_tail_tile_basem_tail_size(from_tiling.get_tilem_tail_tile_basem_tail_size());
    tiling_data.set_tilen_loop_num(from_tiling.get_tilen_loop_num());
    tiling_data.set_tilen_tail_size(from_tiling.get_tilen_tail_size());
    tiling_data.set_tilen_tail_tile_basen_loop_num(from_tiling.get_tilen_tail_tile_basen_loop_num());
    tiling_data.set_tilen_tail_tile_basen_tail_size(from_tiling.get_tilen_tail_tile_basen_tail_size());

  }
  void UpdateGeneralTilingData(MMTilingData& tiling_data) {
    tiling_data.set_block_dim(1);
  }

  void UpdateAxesTilingData(MMTilingData& tiling_data) {
    tiling_data.set_tilen_loop_num(((tiling_data.get_n_size() + tiling_data.get_tilen_size()) - 1) / tiling_data.get_tilen_size());
    tiling_data.set_stepka_loop_num(((tiling_data.get_k_size() + tiling_data.get_stepka_size()) - 1) / tiling_data.get_stepka_size());
    tiling_data.set_stepkb_loop_num(((tiling_data.get_stepka_size() + tiling_data.get_stepkb_size()) - 1) / tiling_data.get_stepkb_size());
    tiling_data.set_basek_loop_num(((tiling_data.get_stepkb_size() + tiling_data.get_basek_size()) - 1) / tiling_data.get_basek_size());
    tiling_data.set_basen_loop_num(((tiling_data.get_tilen_size() + tiling_data.get_basen_size()) - 1) / tiling_data.get_basen_size());
    tiling_data.set_tilem_loop_num(((tiling_data.get_m_size() + tiling_data.get_tilem_size()) - 1) / tiling_data.get_tilem_size());
    tiling_data.set_basem_loop_num(((tiling_data.get_tilem_size() + tiling_data.get_basem_size()) - 1) / tiling_data.get_basem_size());
    tiling_data.set_basen_tail_size((tiling_data.get_tilen_size() % tiling_data.get_basen_size()) == 0 ? tiling_data.get_basen_size() : (tiling_data.get_tilen_size() % tiling_data.get_basen_size()));
    tiling_data.set_stepka_tail_size((tiling_data.get_k_size() % tiling_data.get_stepka_size()) == 0 ? tiling_data.get_stepka_size() : (tiling_data.get_k_size() % tiling_data.get_stepka_size()));
    tiling_data.set_stepkb_tail_size((tiling_data.get_stepka_size() % tiling_data.get_stepkb_size()) == 0 ? tiling_data.get_stepkb_size() : (tiling_data.get_stepka_size() % tiling_data.get_stepkb_size()));
    tiling_data.set_basek_tail_size((tiling_data.get_stepkb_size() % tiling_data.get_basek_size()) == 0 ? tiling_data.get_basek_size() : (tiling_data.get_stepkb_size() % tiling_data.get_basek_size()));
    tiling_data.set_basem_tail_size((tiling_data.get_tilem_size() % tiling_data.get_basem_size()) == 0 ? tiling_data.get_basem_size() : (tiling_data.get_tilem_size() % tiling_data.get_basem_size()));
    tiling_data.set_tilen_tail_size((tiling_data.get_n_size() % tiling_data.get_tilen_size()) == 0 ? tiling_data.get_tilen_size() : (tiling_data.get_n_size() % tiling_data.get_tilen_size()));
    tiling_data.set_tilem_tail_size((tiling_data.get_m_size() % tiling_data.get_tilem_size()) == 0 ? tiling_data.get_tilem_size() : (tiling_data.get_m_size() % tiling_data.get_tilem_size()));
    tiling_data.set_tilem_tail_tile_basem_loop_num(((tiling_data.get_tilem_tail_size() + tiling_data.get_basem_size()) - 1) / tiling_data.get_basem_size());
    tiling_data.set_stepkb_tail_tile_basek_loop_num(((tiling_data.get_stepkb_tail_size() + tiling_data.get_basek_size()) - 1) / tiling_data.get_basek_size());
    tiling_data.set_tilen_tail_tile_basen_loop_num(((tiling_data.get_tilen_tail_size() + tiling_data.get_basen_size()) - 1) / tiling_data.get_basen_size());
    tiling_data.set_stepka_tail_tile_stepkb_loop_num(((tiling_data.get_stepka_tail_size() + tiling_data.get_stepkb_size()) - 1) / tiling_data.get_stepkb_size());
    tiling_data.set_stepkb_tail_tile_basek_tail_size((tiling_data.get_stepkb_tail_size() % tiling_data.get_basek_size()) == 0 ? tiling_data.get_basek_size() : (tiling_data.get_stepkb_tail_size() % tiling_data.get_basek_size()));
    tiling_data.set_tilem_tail_tile_basem_tail_size((tiling_data.get_tilem_tail_size() % tiling_data.get_basem_size()) == 0 ? tiling_data.get_basem_size() : (tiling_data.get_tilem_tail_size() % tiling_data.get_basem_size()));
    tiling_data.set_stepka_tail_tile_stepkb_tail_size((tiling_data.get_stepka_tail_size() % tiling_data.get_stepkb_size()) == 0 ? tiling_data.get_stepkb_size() : (tiling_data.get_stepka_tail_size() % tiling_data.get_stepkb_size()));
    tiling_data.set_tilen_tail_tile_basen_tail_size((tiling_data.get_tilen_tail_size() % tiling_data.get_basen_size()) == 0 ? tiling_data.get_basen_size() : (tiling_data.get_tilen_tail_size() % tiling_data.get_basen_size()));
  }

  void SetQ1(MMTilingData &tiling_data) {
    const auto n_size = tiling_data.get_n_size();
    const auto m_size = tiling_data.get_m_size();
    tiling_data.set_Q1((m_size + n_size));
  }

  void SetMATMUL_OUTPUT1(MMTilingData &tiling_data) {
    const auto n_size = tiling_data.get_n_size();
    const auto m_size = tiling_data.get_m_size();
    tiling_data.set_MATMUL_OUTPUT1((m_size + n_size));
  }

  void ComputeOptionParam(MMTilingData &tiling_data) {
    SetQ1(tiling_data);
    SetMATMUL_OUTPUT1(tiling_data);

  }

  void ExtraTilingData(MMTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1 successfully.");
  }

  void TilingSummary(MMTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set basek_size to %u.", tiling_data.get_basek_size());
    OP_LOGI(OP_NAME, "Set basem_size to %u.", tiling_data.get_basem_size());
    OP_LOGI(OP_NAME, "Set basen_size to %u.", tiling_data.get_basen_size());
    OP_LOGI(OP_NAME, "Set stepka_size to %u.", tiling_data.get_stepka_size());
    OP_LOGI(OP_NAME, "Set stepkb_size to %u.", tiling_data.get_stepkb_size());
    OP_LOGI(OP_NAME, "Set tilem_size to %u.", tiling_data.get_tilem_size());
    OP_LOGI(OP_NAME, "Set tilen_size to %u.", tiling_data.get_tilen_size());
    OP_LOGI(OP_NAME, "The value of AIC_MTE2 is %f.", GetAIC_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIC_FIXPIPE is %f.", GetAIC_FIXPIPE(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIC_MAC is %f.", GetAIC_MAC(tiling_data));
    OP_LOGI(OP_NAME, "The value of l1_size is %d.", Getl1_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l2_size is %d.", Getl2_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0a_size is %d.", Getl0a_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0b_size is %d.", Getl0b_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0c_size is %d.", Getl0c_size(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

// 根据tiling case id创建对应的L0TileSolver子类
class case0L0TileSolver : public L0TileSolver {
  public:
    // 构造函数，接受L0TileInput类型的参数 input
    explicit case0L0TileSolver(L0TileInput &input) : L0TileSolver(input) {};
    // 成员函数声明
    void Setl1_size(const uint32_t &value) { l1_size_ = value; }
    void Setl2_size(const uint32_t &value) { l2_size_ = value; }
    void Setl0a_size(const uint32_t &value) { l0a_size_ = value; }
    void Setl0b_size(const uint32_t &value) { l0b_size_ = value; }
    void Setl0c_size(const uint32_t &value) { l0c_size_ = value; }
    void Setblock_dim(const uint32_t &value) { block_dim_ = value; }
    bool CheckBufferUseValid() override;
  // 定义private类型的成员变量
  private:
    uint32_t l1_size_;
    uint32_t l2_size_;
    uint32_t l0a_size_;
    uint32_t l0b_size_;
    uint32_t l0c_size_;
    uint32_t block_dim_;
    uint32_t k_size{128};
};
  /*
  对创建的L0TileSolver子类的成员函数CheckBufferUseValid进行定义，判断缓存占用是否合法
  i从0开始到待求解l0相关变量的数量，依次递增循环
  @return 如果代求解l0变量的数量为空或i>=待求解l0相关变量的数量，返回false。如果在DEBUG域内输出提示信息
  定义uint32_t类型变量：第i个待求解l0变量的字符串名称，值为第i个待求解l0相关变量的值
  */
bool case0L0TileSolver::CheckBufferUseValid() {
  if ((input_.l0_vars == nullptr) || (0 >= 2)) {
  #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0_vars is nullptr or overflow");
  #endif
      return false;
   }
  uint32_t basem_size = input_.l0_vars[0].value;
  if ((input_.l0_vars == nullptr) || (1 >= 2)) {
  #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0_vars is nullptr or overflow");
  #endif
      return false;
   }
  uint32_t basen_size = input_.l0_vars[1].value;
  /*
  遍历buffer_use_map_中的所有缓存占用表达式，将其转换为字符串类型，赋值给对应的设备类型
  将每种设备类型缓存占用计算值和其设定值对比，如果缓存占用计算值大于设定值，返回false
  @return 遍历所有设备类型后，如果所有设备类型缓存占用计算值都小于设定值，返回true
  */
  uint32_t l0a_size =  (4 * basem_size * k_size);
  if (l0a_size > l0a_size_) {
    return false;
   };
  uint32_t l0b_size =  (4 * basen_size * k_size);
  if (l0b_size > l0b_size_) {
    return false;
   };
  uint32_t l0c_size =  (4 * basem_size * basen_size);
  if (l0c_size > l0c_size_) {
    return false;
   };
  return true;
}
// 根据tiling case id创建对应的L2TileSolver子类
class case0L2TileSolver : public L2TileSolver {
  public:
    // 构造函数，接受L2TileInput类型的参数 input
    explicit case0L2TileSolver(L2TileInput &input, MMTilingData &tiling_data) : L2TileSolver(input), tiling_data_(tiling_data) {};
    // 成员函数声明
    uint64_t GetL2Use() override;
    bool IsClash(const uint32_t idx) override;
    uint32_t k_size{128};
    MMTilingData tiling_data_;
};
/*
对创建的L2TileSolver子类的成员函数GetL2Use进行定义，计算L2的可用内存大小
用input_存储输入数据，并将输入数据对应赋值给tilem_size, tilen_size, k_size
使用公式计算L2的可用内存大小l2_size
@return 返回L2可用内存的大小l2_size
*/
uint64_t case0L2TileSolver::GetL2Use() {
  uint64_t tilem_size = input_.l2_vars[0].value;
  uint64_t tilen_size = input_.l2_vars[1].value;
  uint64_t l2_size = (((tilem_size + tilen_size) * 2 * k_size) + (2 * tilem_size * tilen_size));
  return l2_size;
}
/*
对创建的L2TileSolver子类的成员函数IsClash进行定义，用于检测是否存在读冲突
@param idx 输入参数为idx，用于获取当前tile的编号
@return 如果满足条件blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0,则存在读冲突，返回true
用公式计算尾块个数blocknum_tail
@return 如果满足条件blocknum_tail % (used_corenum_ / 2) == 0,则存在读冲突，返回true
@return 如果以上两个条件均不满足，说明不存在读冲突，返回false
*/
bool case0L2TileSolver::IsClash(const uint32_t idx) {
  if ((used_corenum_ <= 1) || (used_corenum_ % 2 !=0)) {
    return false;
  }
  if (blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0) {
    return true;
  }
  auto blocknum_tail = total_blocknum_[idx] - (tilenum_[idx] - 1) * blocknum_per_tile_[idx];
  if (blocknum_tail % (used_corenum_ / 2) == 0) {
    return true;
  }
  return false;
}
/*
用户可以在派生类中重载Run函数,构造自定义的求解算法,即
  void bool Run(int32_t &solution_num, uint64_t *solutions) override;
其中:
  solution_num:int32_t类型的参数,用来输出实际得到的解的个数
  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,算法将可行解放入该空间
Run函数可以使用下述函数辅助求解:
  bool CheckValid()
    用于检测当前解是否为可行解
  bool UpdateCurVarVal(uint64_t value, int32_t idx)
    将下标为idx的待求解变量改为value,同时更新cons_info_->leqs中的值
  bool RecordBestVarVal()
    待求解变量的当前值所对应的目标函数寻优
Run函数可以使用下述参数辅助求解:
  cons_info_->leqs, double类型的数组, 用于记录不等式约束的函数值, 其下标含义如下:
    cons_info_->leqs[0] = ((4 * Max(128, basen_size) * k_size * stepn_size_div_align) + (4 * Max(16, basem_size) * k_size * stepm_size_div_align) - l1_size)
    cons_info_->leqs[1] = ((tilem_size * tilen_size / (Max(128, basen_size) * Max(16, basem_size) * stepm_size_div_align * stepn_size_div_align)) - block_dim)
    cons_info_->leqs[2] = ((Max(16, basem_size) * stepm_size_div_align) - tilem_size)
    cons_info_->leqs[3] = ((Max(128, basen_size) * stepn_size_div_align) - tilen_size)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase0 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase0(SolverConfig& config, MMTilingData& tiling_data) : GeneralSolver(config) {
            m_size = tiling_data.get_m_size();
            n_size = tiling_data.get_n_size();
            l1_size = tiling_data.get_l1_size();
            block_dim = tiling_data.get_block_dim();
            basem_size = tiling_data.get_basem_size();
            basen_size = tiling_data.get_basen_size();
            tilem_size = tiling_data.get_tilem_size();
            tilen_size = tiling_data.get_tilen_size();
            m_size = ((m_size + 16 - 1) / 16) * 16;
            n_size = ((n_size + 128 - 1) / 128) * 128;
        }

        double GetObj(uint64_t* vars) override;
        double GetSmoothObj(uint64_t* vars) override;
        double GetBuffCost(uint64_t* vars) override;
        bool CheckLocalValid(double* leqs, int32_t idx) override;
        void DisplayVarVal(uint64_t* vars) override;
        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs) override;
        double GetBuffDiff(uint64_t* vars, double* weight) override;
        double GetLeqDiff(uint64_t* vars, double* weight) override;
        double Getblock_dimCost(uint64_t* vars);
        double GetSmoothblock_dimCost(uint64_t* vars);
        double Getl1_sizeCost(uint64_t* vars);
        double GetSmoothl1_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, MMTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, MMTilingData& tiling_data);
    private:
        const int64_t stepm_size_div_align_idx = 0;
        const int64_t stepn_size_div_align_idx = 1;
        uint64_t k_size{128};
        uint64_t m_size;
        uint64_t n_size;
        uint64_t l1_size;
        uint64_t block_dim;
        uint64_t basem_size;
        uint64_t basen_size;
        uint64_t tilem_size;
        uint64_t tilen_size;
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::Getblock_dimCost(uint64_t* vars)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    return ((tilem_size * tilen_size / (Max(128, basen_size) * Max(16, basem_size) * stepm_size_div_align * stepn_size_div_align)) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetSmoothblock_dimCost(uint64_t* vars)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    return ((tilem_size * tilen_size / (Max(128, basen_size) * Max(16, basem_size) * stepm_size_div_align * stepn_size_div_align)) - block_dim);
}

/*
函数名:Getl1_sizeCost(重要函数)
功能描述:
  根据待求解变量值l1_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::Getl1_sizeCost(uint64_t* vars)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    return ((4 * Max(128, basen_size) * k_size * stepn_size_div_align) + (4 * Max(16, basem_size) * k_size * stepm_size_div_align) - l1_size);
}

/*
函数名:GetSmoothl1_sizeCost(重要函数)
功能描述:
  根据待求解变量值l1_size的平滑化缓存占用信息
  与Getl1_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetSmoothl1_sizeCost(uint64_t* vars)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    return ((4 * Max(128, basen_size) * k_size * stepn_size_div_align) + (4 * Max(16, basem_size) * k_size * stepm_size_div_align) - l1_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetObj(uint64_t* vars)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    double AIC_MAC = ((double)(1)/(double)(4096) * basem_size * basen_size * k_size);
    OP_LOGD(OP_NAME, "AIC_MAC = %f", AIC_MAC);
    double AIC_MTE2 = (((double)(1)/(double)(32) * Max(128, basen_size) * k_size * stepn_size_div_align) + ((double)(1)/(double)(32) * Max(16, basem_size) * k_size * stepm_size_div_align));
    OP_LOGD(OP_NAME, "AIC_MTE2 = %f", AIC_MTE2);
    return Max(AIC_MAC, AIC_MTE2);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase0::GetSmoothObj(uint64_t* vars)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    double AIC_MAC = ((double)(1)/(double)(4096) * basem_size * basen_size * k_size);
    double AIC_MTE2 = (((double)(1)/(double)(32) * Max(128, basen_size) * k_size * stepn_size_div_align) + ((double)(1)/(double)(32) * Max(16, basem_size) * k_size * stepm_size_div_align));
    return Max(AIC_MAC, AIC_MTE2);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double l1_size_cost = Getl1_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, l1_size_cost) * Min(0, l1_size_cost)));
}
/*
函数名:GetBuffDiff(重要函数)
功能描述:
  获取缓冲占用加权差分值,计算平滑缓冲占用的差分
  输出的计算公式为sigma_j(delta_{var_i}(g_j(var))) * g_j(var))
  其中g_j为第j个缓冲占用不等式,delta_{var_i}(g_j(var))为g_j(var)沿var_i方向更新一个单位后的变化值
  该函数用于确定变量沿缓冲占用增大的更新方向
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值
*/
double GeneralSolvercase0::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double l1_size_cost = GetSmoothl1_sizeCost(vars);
    l1_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + l1_size_cost;
}
/*
函数名:GetLeqDiff(重要函数)
功能描述:
  获取不等式约束的加权差分值,计算平滑的不等式函数的差分,权值为实际不等式函数值
  输出的计算公式为sigma_j(delta_{var_i}(f_j(var))) * f_j(var))
  其中f_j为第j个不等式约束式,delta_{var_i}(f_j(var))为f_j(var)沿var_i方向更新一个单位后的变化值
  该函数用于确定变量从可行域外侧沿不等式边界方向移动的更新方向
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值
*/
double GeneralSolvercase0::GetLeqDiff(uint64_t* vars, double* weight)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double l1_size_cost = GetSmoothl1_sizeCost(vars);
    l1_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = ((Max(16, basem_size) * stepm_size_div_align) - tilem_size);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = ((Max(128, basen_size) * stepn_size_div_align) - tilen_size);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + l1_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase0::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == stepm_size_div_align_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0 && leqs[2] <= 0;
    } else if (idx == stepn_size_div_align_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double stepm_size_div_align = static_cast<double>(vars[stepm_size_div_align_idx]);
    double stepn_size_div_align = static_cast<double>(vars[stepn_size_div_align_idx]);
    if (idx == stepm_size_div_align_idx) {
        leqs[0] = ((4 * Max(128, basen_size) * k_size * stepn_size_div_align) + (4 * Max(16, basem_size) * k_size * stepm_size_div_align) - l1_size);
        leqs[1] = ((tilem_size * tilen_size / (Max(128, basen_size) * Max(16, basem_size) * stepm_size_div_align * stepn_size_div_align)) - block_dim);
        leqs[2] = ((Max(16, basem_size) * stepm_size_div_align) - tilem_size);
    } else if (idx == stepn_size_div_align_idx) {
        leqs[0] = ((4 * Max(128, basen_size) * k_size * stepn_size_div_align) + (4 * Max(16, basem_size) * k_size * stepm_size_div_align) - l1_size);
        leqs[1] = ((tilem_size * tilen_size / (Max(128, basen_size) * Max(16, basem_size) * stepm_size_div_align * stepn_size_div_align)) - block_dim);
        leqs[3] = ((Max(128, basen_size) * stepn_size_div_align) - tilen_size);
    } else if (idx == -1) {
        leqs[0] = ((4 * Max(128, basen_size) * k_size * stepn_size_div_align) + (4 * Max(16, basem_size) * k_size * stepm_size_div_align) - l1_size);
        leqs[1] = ((tilem_size * tilen_size / (Max(128, basen_size) * Max(16, basem_size) * stepm_size_div_align * stepn_size_div_align)) - block_dim);
        leqs[2] = ((Max(16, basem_size) * stepm_size_div_align) - tilem_size);
        leqs[3] = ((Max(128, basen_size) * stepn_size_div_align) - tilen_size);
    }
}

void GeneralSolvercase0::DisplayVarVal(uint64_t* vars)
{
    uint64_t stepm_size_div_align = vars[stepm_size_div_align_idx];
    uint64_t stepn_size_div_align = vars[stepn_size_div_align_idx];
    OP_LOGD(OP_NAME, "stepm_size = %lu", static_cast<uint64_t>((Max(16, basem_size) * stepm_size_div_align)));
    OP_LOGD(OP_NAME, "stepn_size = %lu", static_cast<uint64_t>((Max(128, basen_size) * stepn_size_div_align)));
}

void GeneralSolvercase0::MapVarVal(uint64_t* vars, MMTilingData& tiling_data)
{
    uint64_t stepm_size_div_align = vars[stepm_size_div_align_idx];
    uint64_t stepn_size_div_align = vars[stepn_size_div_align_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case0 is:");
    tiling_data.set_stepm_size(static_cast<uint64_t>((Max(16, basem_size) * stepm_size_div_align)));
    OP_LOGD(OP_NAME, "stepm_size = %u", tiling_data.get_stepm_size());
    tiling_data.set_stepn_size(static_cast<uint64_t>((Max(128, basen_size) * stepn_size_div_align)));
    OP_LOGD(OP_NAME, "stepn_size = %u", tiling_data.get_stepn_size());
}

void GeneralSolvercase0::GetResult(int32_t solution_num, uint64_t* solution, MMTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case0.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "l1_size = %ld", static_cast<uint64_t>(Getl1_sizeCost(solution) + l1_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case0 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase0Impl : public TilingCaseImpl {
 protected:
/*
定义L0求解器调用函数
@param tiling_data 输入参数为tiling_data
定义求解器接受的输入为L0TileInput类型结构体变量，并将结构体的成员变量赋值为对应值
定义basem_size为L0Var类型的结构体变量，并将相关成员变量赋值为对应值
定义basen_size为L0Var类型的结构体变量，并将相关成员变量赋值为对应值
定义solver为case0L0TileSolver类型的结构体变量，并将此结构体变量的成员变量L1_, L2_, L0A_, L0B_, L0C_, CORENUM_赋值为tiling_data中的设定值
@return 执行L0TileSolver类的主要流程solver.run()，如果solver.run()的返回值不为true,则l0求解器执行失败，函数返回false
用output指针指向solver的输出
i从0开始到L0相关变量个数依次递增
@return 如果output为空指针或者i>=L0相关的变量个数，则函数返回false，否则将basem_size设置为output的第i个值
@return 若上述没有触发返回false的场景，则函数返回true
*/
  bool ExecuteL0Solver(MMTilingData& tiling_data) {
    L0TileInput l0_input;
    l0_input.l0_vars = new(std::nothrow) L0Var[2];
    l0_input.size = 2;
    l0_input.core_num = tiling_data.get_block_dim();
    L0Var basem_size;
    basem_size.max_value = tiling_data.get_m_size();
    basem_size.bind_multicore = true;
  basem_size.align = 16;
    basem_size.is_innermost = true;
    basem_size.prompt_align = 16;
    l0_input.l0_vars[0] = basem_size;
    L0Var basen_size;
    basen_size.max_value = tiling_data.get_n_size();
    basen_size.bind_multicore = true;
  basen_size.align = 16;
    basen_size.is_innermost = true;
    basen_size.prompt_align = 128;
    l0_input.l0_vars[1] = basen_size;
    case0L0TileSolver solver(l0_input);
    solver.Setl1_size(tiling_data.get_l1_size());
    solver.Setl2_size(tiling_data.get_l2_size());
    solver.Setl0a_size(tiling_data.get_l0a_size());
    solver.Setl0b_size(tiling_data.get_l0b_size());
    solver.Setl0c_size(tiling_data.get_l0c_size());
    solver.Setblock_dim(tiling_data.get_block_dim());
    if (!solver.Run()) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "l0 solver run failed");
      #endif
      return false;
    }
    uint32_t *output = solver.GetOutput();
    if ((output == nullptr) || (0 >= 2)) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "output is nullptr or overflow");
      #endif
      return false;
    }
    tiling_data.set_basem_size(output[0]);
    if ((output == nullptr) || (1 >= 2)) {
      #ifdef DEBUG
      OP_LOGW(OP_NAME, "output is nullptr or overflow");
      #endif
      return false;
    }
    tiling_data.set_basen_size(output[1]);
    return true;
  }
/*
定义L2求解器调用函数
@param tiling_data 输入参数为tiling_data
定义求解器的输入为L2TileInput类型结构体变量，并将结构体的成员变量赋值为对应值
定义tilem_size为L2Var类型的结构体变量，并将相关成员变量赋值为对应值
定义tilen_size为L2Var类型的结构体变量，并将相关成员变量赋值为对应值
定义l2_solver为L2TileSolver类型的结构体变量
计算l2的可用内存大小以及是否存在读冲突
@return 执行L2TileSolver类的主要流程l2_solver.run()，如果l2_solver.run()的返回值不为true,则l2求解器执行失败，函数返回false
用output指针指向l2_solver的输出
i从0开始到L2相关变量个数依次递增
@return 如果output为空指针或者i>=L2相关的变量个数，则函数返回false，否则将tilem_size设置为output的第i个值
@return 若上述没有触发返回false的场景，则函数返回true
*/
  bool ExecuteL2Solver(MMTilingData& tiling_data) {
    L2TileInput l2_input;
    l2_input.l2_vars = new(std::nothrow) L2Var[2];
    l2_input.size = 2;
    l2_input.core_num = tiling_data.get_block_dim();
    l2_input.l2_size = EMPIRIC_L2_SIZE;
    L2Var tilem_size;
    tilem_size.max_value = tiling_data.get_m_size();
    tilem_size.align = 16;
    tilem_size.base_val = tiling_data.get_basem_size();
    l2_input.l2_vars[0] = tilem_size;
    L2Var tilen_size;
    tilen_size.max_value = tiling_data.get_n_size();
    tilen_size.align = 16;
    tilen_size.base_val = tiling_data.get_basen_size();
    l2_input.l2_vars[1] = tilen_size;
    case0L2TileSolver l2_solver(l2_input, tiling_data);
    if (!l2_solver.Run()) {
    #ifdef DEBUG
      OP_LOGW(OP_NAME, "l2 solver run failed");
    #endif
      return false;
    }
    uint32_t *output = l2_solver.GetL2Tile();
    if ((output == nullptr) || (0 >= 2)) {
    #ifdef DEBUG
        OP_LOGW(OP_NAME, "l2_vars is nullptr or overflow");
    #endif
      return false;
  }
    tiling_data.set_tilem_size(output[0]);
    if ((output == nullptr) || (1 >= 2)) {
    #ifdef DEBUG
        OP_LOGW(OP_NAME, "l2_vars is nullptr or overflow");
    #endif
      return false;
  }
    tiling_data.set_tilen_size(output[1]);
    return true;
  }
  bool ExecuteGeneralSolver(MMTilingData& tiling_data) {
    SolverConfig cfg;
    cfg.top_num = cfg_top_num;
    cfg.search_length = cfg_search_length;
    cfg.iterations = cfg_iterations;
    cfg.simple_ver = cfg_simple_ver;
    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);
    OP_LOGD(OP_NAME, "Record a maximum of %lu solutions.", cfg.top_num);
    OP_LOGD(OP_NAME, "The searching range covers %lu unit(s).", cfg.search_length);
    OP_LOGD(OP_NAME, "The maximum number of iterations is %lu.", cfg.iterations);
    if (cfg.simple_ver) {
        OP_LOGD(OP_NAME, "Using high-efficiency version.");
    } else {
        OP_LOGD(OP_NAME, "Using high-performance version.");
    }
    OP_LOGD(OP_NAME, "The momentum factor is %f.", cfg.momentum_factor);

    // 以下参数若未注明是可修改参数,则不建议修改
    uint64_t m_size = tiling_data.get_m_size();
    m_size = ((m_size + 16 - 1) / 16) * 16;
    uint64_t n_size = tiling_data.get_n_size();
    n_size = ((n_size + 128 - 1) / 128) * 128;
    uint64_t basem_size = tiling_data.get_basem_size();
    uint64_t basen_size = tiling_data.get_basen_size();
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(stepm_size_div_align, stepn_size_div_align), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>((m_size / (Max(16, basem_size)))), static_cast<uint64_t>((n_size / (Max(128, basen_size))))};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {true, true};
    // 初始化解的个数为0
    int32_t solution_num = 0;
    // 为求解器的输出分配内存
    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];
    if (solution == nullptr)
    {
        OP_LOGW(OP_NAME, "Create solution failed.");
        return false;
    }
    // 通用求解器的输入参数
    SolverInput input;
    input.var_num = num_var;
    input.leq_num = num_leq;
    input.cur_vars = init_vars;
    input.upper_bound = upper_bound;
    input.lower_bound = lower_bound;
    input.update_last = update_last;
    OP_LOGD(OP_NAME, "stepm_size_div_align->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "stepn_size_div_align->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    GeneralSolvercase0* solver = new(std::nothrow) GeneralSolvercase0(cfg, tiling_data);
    if (solver != nullptr) {
        // 导入通用求解器的输入参数并完成初始化
        OP_LOGD(OP_NAME, "Start initializing the input.");
        if (solver -> Init(input)) {
            // 运行通用求解器并获取算法的解
            OP_LOGD(OP_NAME, "Intialization finished, start running the solver.");
            if (solver -> Run(solution_num, solution)) {
                solver -> GetResult(solution_num, solution, tiling_data);
                delete solver;
                delete[] solution;
                OP_LOGD(OP_NAME, "The solver executed successfully.");
                return true;
            }
            OP_LOGW(OP_NAME, "Failed to find any solution.");
        }
    }
    if (solver != nullptr) {
        delete solver;
    }
    if (solution != nullptr) {
        delete[] solution;
    }
    OP_LOGW(OP_NAME, "The solver executed failed.");
    return false;
  }

  bool DoTiling(MMTilingData &tiling_data) {
    if (!ExecuteL0Solver(tiling_data)) {
        return false;
    }
    if (!ExecuteL2Solver(tiling_data)) {
        return false;
    }
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case0.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case0 successfully.");

    return true;
  }

  void DoApiTiling(MMTilingData &tiling_data) {
  }
  int Getl1_size(MMTilingData& tiling_data) {
    double k_size = tiling_data.get_k_size();
    double stepm_size = tiling_data.get_stepm_size();
    double stepn_size = tiling_data.get_stepn_size();

    return ((4 * k_size * stepm_size) + (4 * k_size * stepn_size));
  }

  int Getl2_size(MMTilingData& tiling_data) {
    double k_size = tiling_data.get_k_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return (((tilem_size + tilen_size) * 2 * k_size) + (2 * tilem_size * tilen_size));
  }

  int Getl0a_size(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double k_size = tiling_data.get_k_size();

    return (4 * basem_size * k_size);
  }

  int Getl0b_size(MMTilingData& tiling_data) {
    double basen_size = tiling_data.get_basen_size();
    double k_size = tiling_data.get_k_size();

    return (4 * basen_size * k_size);
  }

  int Getl0c_size(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();

    return (4 * basem_size * basen_size);
  }

  int Getblock_dim(MMTilingData& tiling_data) {
    double stepm_size = tiling_data.get_stepm_size();
    double stepn_size = tiling_data.get_stepn_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return (tilem_size * tilen_size / (stepm_size * stepn_size));
  }

  double GetAIC_MTE2(MMTilingData& tiling_data) {
    double k_size = tiling_data.get_k_size();
    double stepm_size = tiling_data.get_stepm_size();
    double stepn_size = tiling_data.get_stepn_size();

    return (((double)(1)/(double)(32) * k_size * stepm_size) + ((double)(1)/(double)(32) * k_size * stepn_size));
  }

  double GetAIC_MAC(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double k_size = tiling_data.get_k_size();

    return ((double)(1)/(double)(4096) * basem_size * basen_size * k_size);
  }

  double GetPerf(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double k_size = tiling_data.get_k_size();
    double stepm_size = tiling_data.get_stepm_size();
    double stepn_size = tiling_data.get_stepn_size();

    double AIC_MTE2 = (((double)(1)/(double)(32) * k_size * stepm_size) + ((double)(1)/(double)(32) * k_size * stepn_size));
    double AIC_MAC = ((double)(1)/(double)(4096) * basem_size * basen_size * k_size);

    return Max(AIC_MAC, AIC_MTE2);
  }

  void GetTilingData(MMTilingData &tiling_data, MMTilingData &to_tiling) {
    to_tiling.set_m_size(tiling_data.get_m_size());
    to_tiling.set_n_size(tiling_data.get_n_size());
    to_tiling.set_l1_size(tiling_data.get_l1_size());
    to_tiling.set_l2_size(tiling_data.get_l2_size());
    to_tiling.set_l0a_size(tiling_data.get_l0a_size());
    to_tiling.set_l0b_size(tiling_data.get_l0b_size());
    to_tiling.set_l0c_size(tiling_data.get_l0c_size());
    to_tiling.set_block_dim(tiling_data.get_corenum());
    to_tiling.set_corenum(tiling_data.get_corenum());
  }
  void SetTilingData(MMTilingData &from_tiling, MMTilingData &tiling_data) {
    tiling_data.set_basem_size(from_tiling.get_basem_size());
    tiling_data.set_basen_size(from_tiling.get_basen_size());
    tiling_data.set_stepm_size(from_tiling.get_stepm_size());
    tiling_data.set_stepn_size(from_tiling.get_stepn_size());
    tiling_data.set_tilem_size(from_tiling.get_tilem_size());
    tiling_data.set_tilen_size(from_tiling.get_tilen_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_basem_loop_num(from_tiling.get_basem_loop_num());
    tiling_data.set_basem_tail_size(from_tiling.get_basem_tail_size());
    tiling_data.set_basen_loop_num(from_tiling.get_basen_loop_num());
    tiling_data.set_basen_tail_size(from_tiling.get_basen_tail_size());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_stepm_loop_num(from_tiling.get_stepm_loop_num());
    tiling_data.set_stepm_tail_size(from_tiling.get_stepm_tail_size());
    tiling_data.set_stepm_tail_tile_basem_loop_num(from_tiling.get_stepm_tail_tile_basem_loop_num());
    tiling_data.set_stepm_tail_tile_basem_tail_size(from_tiling.get_stepm_tail_tile_basem_tail_size());
    tiling_data.set_stepn_loop_num(from_tiling.get_stepn_loop_num());
    tiling_data.set_stepn_tail_size(from_tiling.get_stepn_tail_size());
    tiling_data.set_stepn_tail_tile_basen_loop_num(from_tiling.get_stepn_tail_tile_basen_loop_num());
    tiling_data.set_stepn_tail_tile_basen_tail_size(from_tiling.get_stepn_tail_tile_basen_tail_size());
    tiling_data.set_test(from_tiling.get_test());
    tiling_data.set_tilem_loop_num(from_tiling.get_tilem_loop_num());
    tiling_data.set_tilem_tail_block_stepm_loop_num(from_tiling.get_tilem_tail_block_stepm_loop_num());
    tiling_data.set_tilem_tail_block_stepm_tail_size(from_tiling.get_tilem_tail_block_stepm_tail_size());
    tiling_data.set_tilem_tail_size(from_tiling.get_tilem_tail_size());
    tiling_data.set_tilen_loop_num(from_tiling.get_tilen_loop_num());
    tiling_data.set_tilen_tail_block_stepn_loop_num(from_tiling.get_tilen_tail_block_stepn_loop_num());
    tiling_data.set_tilen_tail_block_stepn_tail_size(from_tiling.get_tilen_tail_block_stepn_tail_size());
    tiling_data.set_tilen_tail_size(from_tiling.get_tilen_tail_size());

  }
  void UpdateGeneralTilingData(MMTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_tilem_size() + tiling_data.get_stepm_size()) - 1) / tiling_data.get_stepm_size()) * (((tiling_data.get_tilen_size() + tiling_data.get_stepn_size()) - 1) / tiling_data.get_stepn_size()));
  }

  void UpdateAxesTilingData(MMTilingData& tiling_data) {
    tiling_data.set_basem_loop_num(((tiling_data.get_stepm_size() + tiling_data.get_basem_size()) - 1) / tiling_data.get_basem_size());
    tiling_data.set_tilen_loop_num(((tiling_data.get_n_size() + tiling_data.get_tilen_size()) - 1) / tiling_data.get_tilen_size());
    tiling_data.set_stepn_loop_num(((tiling_data.get_tilen_size() + tiling_data.get_stepn_size()) - 1) / tiling_data.get_stepn_size());
    tiling_data.set_basen_loop_num(((tiling_data.get_stepn_size() + tiling_data.get_basen_size()) - 1) / tiling_data.get_basen_size());
    tiling_data.set_tilem_loop_num(((tiling_data.get_m_size() + tiling_data.get_tilem_size()) - 1) / tiling_data.get_tilem_size());
    tiling_data.set_stepm_loop_num(((tiling_data.get_tilem_size() + tiling_data.get_stepm_size()) - 1) / tiling_data.get_stepm_size());
    tiling_data.set_tilem_tail_size((tiling_data.get_m_size() % tiling_data.get_tilem_size()) == 0 ? tiling_data.get_tilem_size() : (tiling_data.get_m_size() % tiling_data.get_tilem_size()));
    tiling_data.set_stepm_tail_size((tiling_data.get_tilem_size() % tiling_data.get_stepm_size()) == 0 ? tiling_data.get_stepm_size() : (tiling_data.get_tilem_size() % tiling_data.get_stepm_size()));
    tiling_data.set_basen_tail_size((tiling_data.get_stepn_size() % tiling_data.get_basen_size()) == 0 ? tiling_data.get_basen_size() : (tiling_data.get_stepn_size() % tiling_data.get_basen_size()));
    tiling_data.set_stepn_tail_size((tiling_data.get_tilen_size() % tiling_data.get_stepn_size()) == 0 ? tiling_data.get_stepn_size() : (tiling_data.get_tilen_size() % tiling_data.get_stepn_size()));
    tiling_data.set_tilen_tail_size((tiling_data.get_n_size() % tiling_data.get_tilen_size()) == 0 ? tiling_data.get_tilen_size() : (tiling_data.get_n_size() % tiling_data.get_tilen_size()));
    tiling_data.set_basem_tail_size((tiling_data.get_stepm_size() % tiling_data.get_basem_size()) == 0 ? tiling_data.get_basem_size() : (tiling_data.get_stepm_size() % tiling_data.get_basem_size()));
    tiling_data.set_tilem_tail_block_stepm_loop_num(((tiling_data.get_tilem_tail_size() + tiling_data.get_stepm_size()) - 1) / tiling_data.get_stepm_size());
    tiling_data.set_stepm_tail_tile_basem_loop_num(((tiling_data.get_stepm_tail_size() + tiling_data.get_basem_size()) - 1) / tiling_data.get_basem_size());
    tiling_data.set_stepn_tail_tile_basen_loop_num(((tiling_data.get_stepn_tail_size() + tiling_data.get_basen_size()) - 1) / tiling_data.get_basen_size());
    tiling_data.set_tilen_tail_block_stepn_loop_num(((tiling_data.get_tilen_tail_size() + tiling_data.get_stepn_size()) - 1) / tiling_data.get_stepn_size());
    tiling_data.set_tilem_tail_block_stepm_tail_size((tiling_data.get_tilem_tail_size() % tiling_data.get_stepm_size()) == 0 ? tiling_data.get_stepm_size() : (tiling_data.get_tilem_tail_size() % tiling_data.get_stepm_size()));
    tiling_data.set_stepn_tail_tile_basen_tail_size((tiling_data.get_stepn_tail_size() % tiling_data.get_basen_size()) == 0 ? tiling_data.get_basen_size() : (tiling_data.get_stepn_tail_size() % tiling_data.get_basen_size()));
    tiling_data.set_stepm_tail_tile_basem_tail_size((tiling_data.get_stepm_tail_size() % tiling_data.get_basem_size()) == 0 ? tiling_data.get_basem_size() : (tiling_data.get_stepm_tail_size() % tiling_data.get_basem_size()));
    tiling_data.set_tilen_tail_block_stepn_tail_size((tiling_data.get_tilen_tail_size() % tiling_data.get_stepn_size()) == 0 ? tiling_data.get_stepn_size() : (tiling_data.get_tilen_tail_size() % tiling_data.get_stepn_size()));
  }

  void ComputeOptionParam(MMTilingData &tiling_data) {

  }

  void ExtraTilingData(MMTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 0.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 0 successfully.");
  }

  void TilingSummary(MMTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set basem_size to %u.", tiling_data.get_basem_size());
    OP_LOGI(OP_NAME, "Set basen_size to %u.", tiling_data.get_basen_size());
    OP_LOGI(OP_NAME, "Set stepm_size to %u.", tiling_data.get_stepm_size());
    OP_LOGI(OP_NAME, "Set stepn_size to %u.", tiling_data.get_stepn_size());
    OP_LOGI(OP_NAME, "Set tilem_size to %u.", tiling_data.get_tilem_size());
    OP_LOGI(OP_NAME, "Set tilen_size to %u.", tiling_data.get_tilen_size());
    OP_LOGI(OP_NAME, "The value of AIC_MTE2 is %f.", GetAIC_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIC_MAC is %f.", GetAIC_MAC(tiling_data));
    OP_LOGI(OP_NAME, "The value of l1_size is %d.", Getl1_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l2_size is %d.", Getl2_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0a_size is %d.", Getl0a_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0b_size is %d.", Getl0b_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0c_size is %d.", Getl0c_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

bool GetTilingImplPtr(int32_t tilingCaseId, TilingCaseImplPtr &tilingCaseImplPtr) {
  switch (tilingCaseId) {
    case 1u:
      tilingCaseImplPtr = std::make_shared<TilingCase1Impl>();
      break;
    case 0u:
      tilingCaseImplPtr = std::make_shared<TilingCase0Impl>();
      break;
    default:
      return false;
  }
  if (tilingCaseImplPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer tilingCaseImplPtr for tilingCaseId [%u] is null.", tilingCaseId);
    return false;
  }
  return true;
}
bool FindPerfBetterTilingbyCaseId(double &obj, MMTilingData &tiling_data, uint32_t tilingCaseId) {
  bool ret;
  double cur_obj;
  TilingCaseImplPtr tilingCaseImplPtr;
  MMTilingData tmp_tiling;
  GetTilingImplPtr(tilingCaseId, tilingCaseImplPtr);
  tilingCaseImplPtr->GetTilingData(tiling_data, tmp_tiling);
  OP_LOGD(OP_NAME, "Construct a backup for tilingCaseId %u.", tilingCaseId);
  ret = tilingCaseImplPtr->GetTiling(tmp_tiling);
  if (!ret) {
    return false;
  }
  cur_obj = tilingCaseImplPtr->GetPerf(tmp_tiling);
    OP_LOGD(OP_NAME, "The optimal objection for tilingCaseId %u is %f.", tilingCaseId, cur_obj);
  if (obj < 0 || cur_obj < obj) {
    OP_LOGD(OP_NAME, "The solution for tilingCaseId %u is better, updating the tiling data.", tilingCaseId);
    tilingCaseImplPtr->SetTilingData(tmp_tiling, tiling_data);
    OP_LOGD(OP_NAME, "Set the output tiling data.");
    obj = cur_obj;
    tiling_data.set_tiling_key(tilingCaseId);
    OP_LOGD(OP_NAME, "Updated the best tilingCaseId to %u.", tilingCaseId);
  }
  return true;
}

bool GetTilingKey(MMTilingData &tiling_data, int32_t tilingCaseId = -1) {
  bool ret = false;
  double obj = -1;
  tiling_data.set_corenum(tiling_data.get_block_dim());
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    uint32_t tilingKeys[2] = {1u, 0u};
    for (const auto &tilingKey : tilingKeys) {
      ret = (FindPerfBetterTilingbyCaseId(obj, tiling_data, tilingKey) || ret);
      OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId %u.", tilingKey);
    }
    if (ret) {
      OP_LOGI(OP_NAME, "Among the templates, tiling case %u is the best choice.", tiling_data.get_tiling_key());
    }
  } else {
    OP_LOGI(OP_NAME, "Calculating the tiling data for tilingCaseId %u.", tilingCaseId);
    TilingCaseImplPtr tilingCaseImplPtr;
    GetTilingImplPtr(tilingCaseId, tilingCaseImplPtr);
    ret = tilingCaseImplPtr->GetTiling(tiling_data);
  }
  if (!ret) {
    OP_LOGE(OP_NAME, "Failed to execute tiling func.");
  }
  return ret;
}

bool GetTiling(MMTilingData &tiling_data, int32_t tilingCaseId) {
  OP_LOGI(OP_NAME, "Start tiling.");
  OP_LOGI(OP_NAME, "Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context.");
  OP_LOGI(OP_NAME, "End tiling.");
  return true;
}

} // namespace optiling

