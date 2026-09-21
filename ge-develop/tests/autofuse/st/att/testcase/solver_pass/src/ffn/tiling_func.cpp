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
#include <chrono>
#include <cstdint>
#include <string>
#include "op_log.h"
#include "tiling_data.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "FFN"

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

struct TilingDataCopy {
  uint32_t K1;
  void set_K1(uint32_t val) { K1 = val; }
  uint32_t get_K1() { return K1; }
  uint32_t N1;
  void set_N1(uint32_t val) { N1 = val; }
  uint32_t get_N1() { return N1; }
  uint32_t N2;
  void set_N2(uint32_t val) { N2 = val; }
  uint32_t get_N2() { return N2; }
  uint32_t base_m1;
  void set_base_m1(uint32_t val) { base_m1 = val; }
  uint32_t get_base_m1() { return base_m1; }
  uint32_t base_m1_loop_num;
  void set_base_m1_loop_num(uint32_t val) { base_m1_loop_num = val; }
  uint32_t get_base_m1_loop_num() { return base_m1_loop_num; }
  uint32_t base_m1_tail_size;
  void set_base_m1_tail_size(uint32_t val) { base_m1_tail_size = val; }
  uint32_t get_base_m1_tail_size() { return base_m1_tail_size; }
  uint32_t base_m1_tail_tile_ub_m_loop_num;
  void set_base_m1_tail_tile_ub_m_loop_num(uint32_t val) { base_m1_tail_tile_ub_m_loop_num = val; }
  uint32_t get_base_m1_tail_tile_ub_m_loop_num() { return base_m1_tail_tile_ub_m_loop_num; }
  uint32_t base_m1_tail_tile_ub_m_tail_size;
  void set_base_m1_tail_tile_ub_m_tail_size(uint32_t val) { base_m1_tail_tile_ub_m_tail_size = val; }
  uint32_t get_base_m1_tail_tile_ub_m_tail_size() { return base_m1_tail_tile_ub_m_tail_size; }
  uint32_t base_m2;
  void set_base_m2(uint32_t val) { base_m2 = val; }
  uint32_t get_base_m2() { return base_m2; }
  uint32_t base_m2_loop_num;
  void set_base_m2_loop_num(uint32_t val) { base_m2_loop_num = val; }
  uint32_t get_base_m2_loop_num() { return base_m2_loop_num; }
  uint32_t base_m2_tail_size;
  void set_base_m2_tail_size(uint32_t val) { base_m2_tail_size = val; }
  uint32_t get_base_m2_tail_size() { return base_m2_tail_size; }
  uint32_t base_n1;
  void set_base_n1(uint32_t val) { base_n1 = val; }
  uint32_t get_base_n1() { return base_n1; }
  uint32_t base_n1_loop_num;
  void set_base_n1_loop_num(uint32_t val) { base_n1_loop_num = val; }
  uint32_t get_base_n1_loop_num() { return base_n1_loop_num; }
  uint32_t base_n1_tail_size;
  void set_base_n1_tail_size(uint32_t val) { base_n1_tail_size = val; }
  uint32_t get_base_n1_tail_size() { return base_n1_tail_size; }
  uint32_t base_n2;
  void set_base_n2(uint32_t val) { base_n2 = val; }
  uint32_t get_base_n2() { return base_n2; }
  uint32_t base_n2_loop_num;
  void set_base_n2_loop_num(uint32_t val) { base_n2_loop_num = val; }
  uint32_t get_base_n2_loop_num() { return base_n2_loop_num; }
  uint32_t base_n2_tail_size;
  void set_base_n2_tail_size(uint32_t val) { base_n2_tail_size = val; }
  uint32_t get_base_n2_tail_size() { return base_n2_tail_size; }
  uint32_t block_dim;
  void set_block_dim(uint32_t val) { block_dim = val; }
  uint32_t get_block_dim() { return block_dim; }
  uint32_t btbuf_size;
  void set_btbuf_size(uint32_t val) { btbuf_size = val; }
  uint32_t get_btbuf_size() { return btbuf_size; }
  uint32_t gm_size;
  void set_gm_size(uint32_t val) { gm_size = val; }
  uint32_t get_gm_size() { return gm_size; }
  uint32_t l0c_size;
  void set_l0c_size(uint32_t val) { l0c_size = val; }
  uint32_t get_l0c_size() { return l0c_size; }
  uint32_t maxTokens;
  void set_maxTokens(uint32_t val) { maxTokens = val; }
  uint32_t get_maxTokens() { return maxTokens; }
  uint32_t output0_single_core_size;
  void set_output0_single_core_size(uint32_t val) { output0_single_core_size = val; }
  uint32_t get_output0_single_core_size() { return output0_single_core_size; }
  uint32_t output0_total_size;
  void set_output0_total_size(uint32_t val) { output0_total_size = val; }
  uint32_t get_output0_total_size() { return output0_total_size; }
  uint32_t singlecore_m1;
  void set_singlecore_m1(uint32_t val) { singlecore_m1 = val; }
  uint32_t get_singlecore_m1() { return singlecore_m1; }
  uint32_t singlecore_m1_loop_num;
  void set_singlecore_m1_loop_num(uint32_t val) { singlecore_m1_loop_num = val; }
  uint32_t get_singlecore_m1_loop_num() { return singlecore_m1_loop_num; }
  uint32_t singlecore_m1_tail_size;
  void set_singlecore_m1_tail_size(uint32_t val) { singlecore_m1_tail_size = val; }
  uint32_t get_singlecore_m1_tail_size() { return singlecore_m1_tail_size; }
  uint32_t singlecore_m1_tail_tile_base_m1_loop_num;
  void set_singlecore_m1_tail_tile_base_m1_loop_num(uint32_t val) { singlecore_m1_tail_tile_base_m1_loop_num = val; }
  uint32_t get_singlecore_m1_tail_tile_base_m1_loop_num() { return singlecore_m1_tail_tile_base_m1_loop_num; }
  uint32_t singlecore_m1_tail_tile_base_m1_tail_size;
  void set_singlecore_m1_tail_tile_base_m1_tail_size(uint32_t val) { singlecore_m1_tail_tile_base_m1_tail_size = val; }
  uint32_t get_singlecore_m1_tail_tile_base_m1_tail_size() { return singlecore_m1_tail_tile_base_m1_tail_size; }
  uint32_t singlecore_m2;
  void set_singlecore_m2(uint32_t val) { singlecore_m2 = val; }
  uint32_t get_singlecore_m2() { return singlecore_m2; }
  uint32_t singlecore_m2_loop_num;
  void set_singlecore_m2_loop_num(uint32_t val) { singlecore_m2_loop_num = val; }
  uint32_t get_singlecore_m2_loop_num() { return singlecore_m2_loop_num; }
  uint32_t singlecore_m2_tail_size;
  void set_singlecore_m2_tail_size(uint32_t val) { singlecore_m2_tail_size = val; }
  uint32_t get_singlecore_m2_tail_size() { return singlecore_m2_tail_size; }
  uint32_t singlecore_m2_tail_tile_base_m2_loop_num;
  void set_singlecore_m2_tail_tile_base_m2_loop_num(uint32_t val) { singlecore_m2_tail_tile_base_m2_loop_num = val; }
  uint32_t get_singlecore_m2_tail_tile_base_m2_loop_num() { return singlecore_m2_tail_tile_base_m2_loop_num; }
  uint32_t singlecore_m2_tail_tile_base_m2_tail_size;
  void set_singlecore_m2_tail_tile_base_m2_tail_size(uint32_t val) { singlecore_m2_tail_tile_base_m2_tail_size = val; }
  uint32_t get_singlecore_m2_tail_tile_base_m2_tail_size() { return singlecore_m2_tail_tile_base_m2_tail_size; }
  uint32_t singlecore_n1;
  void set_singlecore_n1(uint32_t val) { singlecore_n1 = val; }
  uint32_t get_singlecore_n1() { return singlecore_n1; }
  uint32_t singlecore_n1_loop_num;
  void set_singlecore_n1_loop_num(uint32_t val) { singlecore_n1_loop_num = val; }
  uint32_t get_singlecore_n1_loop_num() { return singlecore_n1_loop_num; }
  uint32_t singlecore_n1_tail_size;
  void set_singlecore_n1_tail_size(uint32_t val) { singlecore_n1_tail_size = val; }
  uint32_t get_singlecore_n1_tail_size() { return singlecore_n1_tail_size; }
  uint32_t singlecore_n1_tail_tile_base_n1_loop_num;
  void set_singlecore_n1_tail_tile_base_n1_loop_num(uint32_t val) { singlecore_n1_tail_tile_base_n1_loop_num = val; }
  uint32_t get_singlecore_n1_tail_tile_base_n1_loop_num() { return singlecore_n1_tail_tile_base_n1_loop_num; }
  uint32_t singlecore_n1_tail_tile_base_n1_tail_size;
  void set_singlecore_n1_tail_tile_base_n1_tail_size(uint32_t val) { singlecore_n1_tail_tile_base_n1_tail_size = val; }
  uint32_t get_singlecore_n1_tail_tile_base_n1_tail_size() { return singlecore_n1_tail_tile_base_n1_tail_size; }
  uint32_t singlecore_n2;
  void set_singlecore_n2(uint32_t val) { singlecore_n2 = val; }
  uint32_t get_singlecore_n2() { return singlecore_n2; }
  uint32_t singlecore_n2_loop_num;
  void set_singlecore_n2_loop_num(uint32_t val) { singlecore_n2_loop_num = val; }
  uint32_t get_singlecore_n2_loop_num() { return singlecore_n2_loop_num; }
  uint32_t singlecore_n2_tail_size;
  void set_singlecore_n2_tail_size(uint32_t val) { singlecore_n2_tail_size = val; }
  uint32_t get_singlecore_n2_tail_size() { return singlecore_n2_tail_size; }
  uint32_t singlecore_n2_tail_tile_base_n2_loop_num;
  void set_singlecore_n2_tail_tile_base_n2_loop_num(uint32_t val) { singlecore_n2_tail_tile_base_n2_loop_num = val; }
  uint32_t get_singlecore_n2_tail_tile_base_n2_loop_num() { return singlecore_n2_tail_tile_base_n2_loop_num; }
  uint32_t singlecore_n2_tail_tile_base_n2_tail_size;
  void set_singlecore_n2_tail_tile_base_n2_tail_size(uint32_t val) { singlecore_n2_tail_tile_base_n2_tail_size = val; }
  uint32_t get_singlecore_n2_tail_tile_base_n2_tail_size() { return singlecore_n2_tail_tile_base_n2_tail_size; }
  uint32_t tiling_key;
  void set_tiling_key(uint32_t val) { tiling_key = val; }
  uint32_t get_tiling_key() { return tiling_key; }
  uint32_t ub_m;
  void set_ub_m(uint32_t val) { ub_m = val; }
  uint32_t get_ub_m() { return ub_m; }
  uint32_t ub_m_loop_num;
  void set_ub_m_loop_num(uint32_t val) { ub_m_loop_num = val; }
  uint32_t get_ub_m_loop_num() { return ub_m_loop_num; }
  uint32_t ub_m_tail_size;
  void set_ub_m_tail_size(uint32_t val) { ub_m_tail_size = val; }
  uint32_t get_ub_m_tail_size() { return ub_m_tail_size; }
  uint32_t ub_size;
  void set_ub_size(uint32_t val) { ub_size = val; }
  uint32_t get_ub_size() { return ub_size; }
  uint32_t workspaceSize;
  void set_workspaceSize(uint32_t val) { workspaceSize = val; }
  uint32_t get_workspaceSize() { return workspaceSize; }
};
class TilingCaseImpl {
 public:
  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}
  virtual ~TilingCaseImpl() = default;
  bool GetTiling(FFNTilingData &tiling_data) {
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    ExtraTilingData(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(FFNTilingData &tiling_data) { return 0.0; }
  virtual void GetTilingData(TilingDataCopy &from_tiling, FFNTilingData &to_tiling) {};
  virtual void SetTilingData(FFNTilingData &from_tiling, TilingDataCopy &to_tiling) {};
 protected:
  virtual bool DoTiling(FFNTilingData &tiling_data) = 0;
  virtual void DoApiTiling(FFNTilingData &tiling_data) {}
  virtual void GetWorkSpaceSize(FFNTilingData& tiling_data) {}
  virtual void ExtraTilingData(FFNTilingData &tiling_data) {}
  virtual void TilingSummary(FFNTilingData &tiling_data) = 0;
  uint32_t corenum_;
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;

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
static const bool cfg_simple_ver = true;
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
    uint64_t corenum{0u};
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
        if (vars_ != nullptr) {
            delete[] vars_;
        }
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
    bool GetCoarseLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality);
    bool GetFineLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality);
    bool GetPeerLoc(const UpdateInfo &update_info, Locality &cur_locality);
    bool LocateLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality, Locality &best_locality);
    bool TryLocate(int32_t idx, double init_obj, Locality &best_locality);

    TunePriority GetTunePriority(int32_t idx, double rec_obj, double &cur_obj);
    bool SearchLoc(const UpdateInfo &update_info, uint64_t &step, double &cur_obj, TunePriority &cur_priority);
    bool GetHarmlessLoc(const UpdateInfo &update_info, uint64_t &step, double &cur_obj);
    bool GetDilatedLoc(const UpdateInfo &update_info, uint64_t &step);
    bool TuneLoc(const UpdateInfo &update_info, double cur_obj, uint64_t &step, TunePriority &cur_priority, TunePriority &best_priority);
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
            cur_node->next_var = new_node;
            tail->next_node = new_node;
            tail = tail->next_node;
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
            cur_node->next_val = new_node;
            tail->next_node = new_node;
            tail = tail->next_node;
            cur_node = new_node;
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
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return UpdateDirection::NONE;
    }
    double *weight = new(std::nothrow) double[cons_info_->leq_num];
    if (weight == nullptr) {
        return UpdateDirection::NONE;
    }
    UpdateLeqs(vars, -1, weight);
    double cur_val = GetFuncVal(vars, weight, func_info);
    vars[idx] += 1;
    double next_val = GetFuncVal(vars, weight, func_info);
    vars[idx] -= 1;
    if (!IsEqual(cur_val, next_val))
    {
        delete[] weight;
        return (cur_val > next_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
    }
    if (vars[idx] >= 1)
    {
        vars[idx] -= 1;
        double pre_val = GetFuncVal(vars, weight, func_info);
        vars[idx] += 1;
        if (!IsEqual(cur_val, pre_val))
        {
            delete[] weight;
            return (pre_val > cur_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
        }
    }
    delete[] weight;
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
    if ((var_info_ != nullptr) && (cons_info_ != nullptr) && (momentum_info_ != nullptr))
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
bool GeneralSolver::GetCoarseLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality)
{
    uint64_t update_value;

    int32_t idx = update_info.idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return false;
    }
    uint64_t thres = update_info.thres;
    UpdateDirection update_direction = update_info.update_direction;
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
bool GeneralSolver::GetFineLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality)
{
    uint64_t update_value;
    Locality rec_locality;

    int32_t idx = update_info.idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return false;
    }
    UpdateDirection update_direction = update_info.update_direction;
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
bool GeneralSolver::GetPeerLoc(const UpdateInfo &update_info, Locality &cur_locality)
{
    uint64_t left_value;
    uint64_t right_value;
    uint64_t mid_value;
    Locality rec_locality;
    int32_t idx = update_info.idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return false;
    }
    uint64_t rec_value = var_info_->cur_vars[idx];
    UpdateDirection update_direction = update_info.update_direction;
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
bool GeneralSolver::LocateLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality, Locality &best_locality)
{
    int32_t idx = update_info.idx;
    double init_obj = update_info.init_obj;
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
        UpdateInfo update_info = UpdateInfo(idx, thres, update_direction, init_obj);
        if (GetCoarseLoc(update_info, step, cur_locality))
        {
            if (!LocateLoc(update_info, step, cur_locality, best_locality))
            {
                UpdateCurVarVal(var_info_->rec_vars[idx], idx);
                return false;
            }
            UpdateCurVarVal(var_info_->rec_vars[idx], idx);
        }
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
bool GeneralSolver::SearchLoc(const UpdateInfo &update_info, uint64_t &step, double &cur_obj, TunePriority &cur_priority)
{
    TunePriority rec_priority;
    int32_t idx = update_info.idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return false;
    }
    uint64_t thres = update_info.thres;
    UpdateDirection update_direction = update_info.update_direction;
    double init_obj = update_info.init_obj;
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
bool GeneralSolver::GetHarmlessLoc(const UpdateInfo &update_info, uint64_t &step, double &cur_obj)
{
    double rec_obj;
    int32_t update_value;
    TunePriority rec_priority;
    int32_t idx = update_info.idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return false;
    }
    uint64_t thres = update_info.thres;
    UpdateDirection update_direction = update_info.update_direction;
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
bool GeneralSolver::GetDilatedLoc(const UpdateInfo &update_info, uint64_t &step)
{
    int32_t idx = update_info.idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
        OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
        return false;
    }
    uint64_t update_value;
    uint64_t thres = update_info.thres;
    UpdateDirection update_direction = update_info.update_direction;
    double cur_obj;
    double cur_cons;
    double init_obj = update_info.init_obj;
    double init_cons = update_info.init_cons;
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
bool GeneralSolver::TuneLoc(const UpdateInfo &update_info, double cur_obj, uint64_t &step, TunePriority &cur_priority, TunePriority &best_priority)
{
    if (cur_priority <= best_priority)
    {
        uint64_t update_value;
        int32_t idx = update_info.idx;
        if ((idx < 0) || (idx >= var_info_->var_num)) {
            OP_LOGE(OP_NAME, "idx = %d, var_info_->var_num = %d, idx illegal.", idx, var_info_->var_num);
            return false;
        }
        UpdateDirection update_direction = update_info.update_direction;
        double init_obj = update_info.init_obj;
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
        UpdateInfo update_info = UpdateInfo(idx, thres, update_direction, init_obj, init_cons);
        if (SearchLoc(update_info, step, cur_obj, cur_priority))
        {
            if (!TuneLoc(update_info, cur_obj, step, cur_priority, best_priority))
            {
                return false;
            }
            UpdateCurVarVal(var_info_->rec_vars[idx], idx);
        }
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
    cons_info_->leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size)
    cons_info_->leqs[1] = ((2560 * base_n1_div_align * pow(2, ub_m_base)) - ub_size)
    cons_info_->leqs[2] = (Max((64 * base_n2_div_align), (64 * base_n1_div_align)) - btbuf_size)
    cons_info_->leqs[3] = (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase0 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase0(SolverConfig& config, FFNTilingData& tiling_data) : GeneralSolver(config) {
            K1 = tiling_data.get_K1();
            N1 = tiling_data.get_N1();
            N2 = tiling_data.get_N2();
            maxTokens = tiling_data.get_maxTokens();
            l0c_size = tiling_data.get_l0c_size();
            ub_size = tiling_data.get_ub_size();
            btbuf_size = tiling_data.get_btbuf_size();
            N1 = ((N1 + 16 - 1) / 16) * 16;
            N2 = ((N2 + 16 - 1) / 16) * 16;
            maxTokens = ((maxTokens + 16 - 1) / 16) * 16;
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
        double Getbtbuf_sizeCost(uint64_t* vars);
        double GetSmoothbtbuf_sizeCost(uint64_t* vars);
        double Getl0c_sizeCost(uint64_t* vars);
        double GetSmoothl0c_sizeCost(uint64_t* vars);
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, FFNTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, FFNTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t base_m1_div_align_idx = 0;
        const int64_t base_m2_div_align_idx = 1;
        const int64_t base_n1_div_align_idx = 2;
        const int64_t base_n2_div_align_idx = 3;
        const int64_t singlecore_m1_div_align_idx = 4;
        const int64_t singlecore_m2_div_align_idx = 5;
        const int64_t singlecore_n1_div_align_idx = 6;
        const int64_t singlecore_n2_div_align_idx = 7;
        const int64_t ub_m_base_idx = 8;
        uint64_t K1;
        uint64_t N1;
        uint64_t N2;
        uint64_t maxTokens;
        uint64_t l0c_size;
        uint64_t ub_size;
        uint64_t btbuf_size;
        uint64_t block_dim{0};
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
    double singlecore_m1_div_align = static_cast<double>(vars[singlecore_m1_div_align_idx]);
    double singlecore_m2_div_align = static_cast<double>(vars[singlecore_m2_div_align_idx]);
    double singlecore_n1_div_align = static_cast<double>(vars[singlecore_n1_div_align_idx]);
    double singlecore_n2_div_align = static_cast<double>(vars[singlecore_n2_div_align_idx]);
    return (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
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
    double singlecore_m1_div_align = static_cast<double>(vars[singlecore_m1_div_align_idx]);
    double singlecore_m2_div_align = static_cast<double>(vars[singlecore_m2_div_align_idx]);
    double singlecore_n1_div_align = static_cast<double>(vars[singlecore_n1_div_align_idx]);
    double singlecore_n2_div_align = static_cast<double>(vars[singlecore_n2_div_align_idx]);
    return (Max(((((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * (((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), ((((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * (((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
}

/*
函数名:Getbtbuf_sizeCost(重要函数)
功能描述:
  根据待求解变量值btbuf_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::Getbtbuf_sizeCost(uint64_t* vars)
{
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    return (Max((64 * base_n2_div_align), (64 * base_n1_div_align)) - btbuf_size);
}

/*
函数名:GetSmoothbtbuf_sizeCost(重要函数)
功能描述:
  根据待求解变量值btbuf_size的平滑化缓存占用信息
  与Getbtbuf_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetSmoothbtbuf_sizeCost(uint64_t* vars)
{
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    return (Max((64 * base_n2_div_align), (64 * base_n1_div_align)) - btbuf_size);
}

/*
函数名:Getl0c_sizeCost(重要函数)
功能描述:
  根据待求解变量值l0c_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::Getl0c_sizeCost(uint64_t* vars)
{
    double base_m1_div_align = static_cast<double>(vars[base_m1_div_align_idx]);
    double base_m2_div_align = static_cast<double>(vars[base_m2_div_align_idx]);
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    return (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
}

/*
函数名:GetSmoothl0c_sizeCost(重要函数)
功能描述:
  根据待求解变量值l0c_size的平滑化缓存占用信息
  与Getl0c_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetSmoothl0c_sizeCost(uint64_t* vars)
{
    double base_m1_div_align = static_cast<double>(vars[base_m1_div_align_idx]);
    double base_m2_div_align = static_cast<double>(vars[base_m2_div_align_idx]);
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    return (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::Getub_sizeCost(uint64_t* vars)
{
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    return ((2560 * base_n1_div_align * pow(2, ub_m_base)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetSmoothub_sizeCost(uint64_t* vars)
{
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    return ((2560 * base_n1_div_align * pow(2, ub_m_base)) - ub_size);
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
    double base_m1_div_align = static_cast<double>(vars[base_m1_div_align_idx]);
    double base_m2_div_align = static_cast<double>(vars[base_m2_div_align_idx]);
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    double singlecore_m1_div_align = static_cast<double>(vars[singlecore_m1_div_align_idx]);
    double singlecore_m2_div_align = static_cast<double>(vars[singlecore_m2_div_align_idx]);
    double singlecore_n1_div_align = static_cast<double>(vars[singlecore_n1_div_align_idx]);
    double singlecore_n2_div_align = static_cast<double>(vars[singlecore_n2_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    double AIC_MAC = ((16 * ceiling(((double)(1)/(double)(16) * K1)) * ceiling(((double)(1)/(double)(16) * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align)) * ceiling(base_n1_div_align) * singlecore_m1_div_align * singlecore_n1_div_align / (Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align)) + (ceiling(((double)(1)/(double)(16) * N1)) * ceiling(base_m2_div_align) * ceiling(base_n2_div_align) * singlecore_m2_div_align * singlecore_n2_div_align / (base_m2_div_align * base_n2_div_align)));
    OP_LOGD(OP_NAME, "AIC_MAC = %f", AIC_MAC);
    double AIC_MTE2 = (((((double)(1)/(double)(2) * K1 * Max(1, (16 / (base_n1_div_align))) * base_n1_div_align) + ((double)(1)/(double)(32) * K1 * Max(1, (256 / (K1))) * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align)) * 16 * singlecore_m1_div_align * singlecore_n1_div_align / (Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align)) + ((((double)(1)/(double)(2) * Max(1, (16 / (base_n2_div_align))) * N1 * base_n2_div_align) + ((double)(1)/(double)(2) * Max(1, (256 / (N1))) * N1 * base_m2_div_align)) * singlecore_m2_div_align * singlecore_n2_div_align / (base_m2_div_align * base_n2_div_align)));
    OP_LOGD(OP_NAME, "AIC_MTE2 = %f", AIC_MTE2);
    double AIV_MTE2 = (16 * singlecore_m1_div_align * singlecore_n1_div_align);
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = (8 * singlecore_m1_div_align * singlecore_n1_div_align);
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    double AIV_VEC = (4 * singlecore_m1_div_align * singlecore_n1_div_align);
    OP_LOGD(OP_NAME, "AIV_VEC = %f", AIV_VEC);
    return Max(Max(Max(Max(AIC_MAC, AIV_VEC), AIC_MTE2), AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase0::GetSmoothObj(uint64_t* vars)
{
    double base_m1_div_align = static_cast<double>(vars[base_m1_div_align_idx]);
    double base_m2_div_align = static_cast<double>(vars[base_m2_div_align_idx]);
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    double singlecore_m1_div_align = static_cast<double>(vars[singlecore_m1_div_align_idx]);
    double singlecore_m2_div_align = static_cast<double>(vars[singlecore_m2_div_align_idx]);
    double singlecore_n1_div_align = static_cast<double>(vars[singlecore_n1_div_align_idx]);
    double singlecore_n2_div_align = static_cast<double>(vars[singlecore_n2_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    double AIC_MAC = ((16 * (((double)(1)/(double)(16) * K1)) * (((double)(1)/(double)(16) * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align)) * (base_n1_div_align) * singlecore_m1_div_align * singlecore_n1_div_align / (Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align)) + ((((double)(1)/(double)(16) * N1)) * (base_m2_div_align) * (base_n2_div_align) * singlecore_m2_div_align * singlecore_n2_div_align / (base_m2_div_align * base_n2_div_align)));
    double AIC_MTE2 = (((((double)(1)/(double)(2) * K1 * Max(1, (16 / (base_n1_div_align))) * base_n1_div_align) + ((double)(1)/(double)(32) * K1 * Max(1, (256 / (K1))) * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align)) * 16 * singlecore_m1_div_align * singlecore_n1_div_align / (Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align)) + ((((double)(1)/(double)(2) * Max(1, (16 / (base_n2_div_align))) * N1 * base_n2_div_align) + ((double)(1)/(double)(2) * Max(1, (256 / (N1))) * N1 * base_m2_div_align)) * singlecore_m2_div_align * singlecore_n2_div_align / (base_m2_div_align * base_n2_div_align)));
    double AIV_MTE2 = (16 * singlecore_m1_div_align * singlecore_n1_div_align);
    double AIV_MTE3 = (8 * singlecore_m1_div_align * singlecore_n1_div_align);
    double AIV_VEC = (4 * singlecore_m1_div_align * singlecore_n1_div_align);
    return Max(Max(Max(Max(AIC_MAC, AIV_VEC), AIC_MTE2), AIV_MTE2), AIV_MTE3);
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
    double btbuf_size_cost = Getbtbuf_sizeCost(vars);
    double l0c_size_cost = Getl0c_sizeCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, btbuf_size_cost) * Min(0, btbuf_size_cost)) + (Min(0, l0c_size_cost) * Min(0, l0c_size_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
    block_dim_cost *= weight[3] < 0 ? weight[3] : 0;
    double btbuf_size_cost = GetSmoothbtbuf_sizeCost(vars);
    btbuf_size_cost *= weight[2] < 0 ? weight[2] : 0;
    double l0c_size_cost = GetSmoothl0c_sizeCost(vars);
    l0c_size_cost *= weight[0] < 0 ? weight[0] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[1] < 0 ? weight[1] : 0;
    return block_dim_cost + btbuf_size_cost + l0c_size_cost + ub_size_cost;
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
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[3] > 0 ? weight[3] : 0;
    double btbuf_size_cost = GetSmoothbtbuf_sizeCost(vars);
    btbuf_size_cost *= weight[2] > 0 ? weight[2] : 0;
    double l0c_size_cost = GetSmoothl0c_sizeCost(vars);
    l0c_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[1] > 0 ? weight[1] : 0;
    return block_dim_cost + btbuf_size_cost + l0c_size_cost + ub_size_cost;
}
bool GeneralSolvercase0::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == base_m1_div_align_idx) {
        return leqs[0] <= 0;
    } else if (idx == base_m2_div_align_idx) {
        return leqs[0] <= 0;
    } else if (idx == base_n1_div_align_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0 && leqs[2] <= 0;
    } else if (idx == base_n2_div_align_idx) {
        return leqs[0] <= 0 && leqs[2] <= 0;
    } else if (idx == singlecore_m1_div_align_idx) {
        return leqs[3] <= 0;
    } else if (idx == singlecore_m2_div_align_idx) {
        return leqs[3] <= 0;
    } else if (idx == singlecore_n1_div_align_idx) {
        return leqs[3] <= 0;
    } else if (idx == singlecore_n2_div_align_idx) {
        return leqs[3] <= 0;
    } else if (idx == ub_m_base_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0;
    }
    return true;
}

void GeneralSolvercase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double base_m1_div_align = static_cast<double>(vars[base_m1_div_align_idx]);
    double base_m2_div_align = static_cast<double>(vars[base_m2_div_align_idx]);
    double base_n1_div_align = static_cast<double>(vars[base_n1_div_align_idx]);
    double base_n2_div_align = static_cast<double>(vars[base_n2_div_align_idx]);
    double singlecore_m1_div_align = static_cast<double>(vars[singlecore_m1_div_align_idx]);
    double singlecore_m2_div_align = static_cast<double>(vars[singlecore_m2_div_align_idx]);
    double singlecore_n1_div_align = static_cast<double>(vars[singlecore_n1_div_align_idx]);
    double singlecore_n2_div_align = static_cast<double>(vars[singlecore_n2_div_align_idx]);
    double ub_m_base = static_cast<double>(vars[ub_m_base_idx]);
    if (idx == base_m1_div_align_idx) {
        leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
    } else if (idx == base_m2_div_align_idx) {
        leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
    } else if (idx == base_n1_div_align_idx) {
        leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
        leqs[1] = ((2560 * base_n1_div_align * pow(2, ub_m_base)) - ub_size);
        leqs[2] = (Max((64 * base_n2_div_align), (64 * base_n1_div_align)) - btbuf_size);
    } else if (idx == base_n2_div_align_idx) {
        leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
        leqs[2] = (Max((64 * base_n2_div_align), (64 * base_n1_div_align)) - btbuf_size);
    } else if (idx == singlecore_m1_div_align_idx) {
        leqs[3] = (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
    } else if (idx == singlecore_m2_div_align_idx) {
        leqs[3] = (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
    } else if (idx == singlecore_n1_div_align_idx) {
        leqs[3] = (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
    } else if (idx == singlecore_n2_div_align_idx) {
        leqs[3] = (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
    } else if (idx == ub_m_base_idx) {
        leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
        leqs[1] = ((2560 * base_n1_div_align * pow(2, ub_m_base)) - ub_size);
    } else if (idx == -1) {
        leqs[0] = (Max((64 * Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align * base_n1_div_align), (1024 * base_m2_div_align * base_n2_div_align)) - l0c_size);
        leqs[1] = ((2560 * base_n1_div_align * pow(2, ub_m_base)) - ub_size);
        leqs[2] = (Max((64 * base_n2_div_align), (64 * base_n1_div_align)) - btbuf_size);
        leqs[3] = (Max((ceiling(((double)(1)/(double)(16) * N1 / (singlecore_n1_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m1_div_align)))), (ceiling(((double)(1)/(double)(16) * N2 / (singlecore_n2_div_align))) * ceiling(((double)(1)/(double)(16) * maxTokens / (singlecore_m2_div_align))))) - block_dim);
    }
}

void GeneralSolvercase0::DisplayVarVal(uint64_t* vars)
{
    uint64_t base_m1_div_align = vars[base_m1_div_align_idx];
    uint64_t base_m2_div_align = vars[base_m2_div_align_idx];
    uint64_t base_n1_div_align = vars[base_n1_div_align_idx];
    uint64_t base_n2_div_align = vars[base_n2_div_align_idx];
    uint64_t singlecore_m1_div_align = vars[singlecore_m1_div_align_idx];
    uint64_t singlecore_m2_div_align = vars[singlecore_m2_div_align_idx];
    uint64_t singlecore_n1_div_align = vars[singlecore_n1_div_align_idx];
    uint64_t singlecore_n2_div_align = vars[singlecore_n2_div_align_idx];
    uint64_t ub_m_base = vars[ub_m_base_idx];
    OP_LOGD(OP_NAME, "singlecore_m1 = %lu", static_cast<uint64_t>((16 * singlecore_m1_div_align)));
    OP_LOGD(OP_NAME, "singlecore_m2 = %lu", static_cast<uint64_t>((16 * singlecore_m2_div_align)));
    OP_LOGD(OP_NAME, "base_m1 = %lu", static_cast<uint64_t>((Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align)));
    OP_LOGD(OP_NAME, "base_m2 = %lu", static_cast<uint64_t>((16 * base_m2_div_align)));
    OP_LOGD(OP_NAME, "ub_m = %lu", static_cast<uint64_t>((16 * pow(2, ub_m_base))));
    OP_LOGD(OP_NAME, "singlecore_n1 = %lu", static_cast<uint64_t>((16 * singlecore_n1_div_align)));
    OP_LOGD(OP_NAME, "base_n1 = %lu", static_cast<uint64_t>((16 * base_n1_div_align)));
    OP_LOGD(OP_NAME, "singlecore_n2 = %lu", static_cast<uint64_t>((16 * singlecore_n2_div_align)));
    OP_LOGD(OP_NAME, "base_n2 = %lu", static_cast<uint64_t>((16 * base_n2_div_align)));
}

void GeneralSolvercase0::MapVarVal(uint64_t* vars, FFNTilingData& tiling_data)
{
    uint64_t base_m1_div_align = vars[base_m1_div_align_idx];
    uint64_t base_m2_div_align = vars[base_m2_div_align_idx];
    uint64_t base_n1_div_align = vars[base_n1_div_align_idx];
    uint64_t base_n2_div_align = vars[base_n2_div_align_idx];
    uint64_t singlecore_m1_div_align = vars[singlecore_m1_div_align_idx];
    uint64_t singlecore_m2_div_align = vars[singlecore_m2_div_align_idx];
    uint64_t singlecore_n1_div_align = vars[singlecore_n1_div_align_idx];
    uint64_t singlecore_n2_div_align = vars[singlecore_n2_div_align_idx];
    uint64_t ub_m_base = vars[ub_m_base_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case0 is:");
    tiling_data.set_singlecore_m1(static_cast<uint64_t>((16 * singlecore_m1_div_align)));
    OP_LOGD(OP_NAME, "singlecore_m1 = %u", tiling_data.get_singlecore_m1());
    tiling_data.set_singlecore_m2(static_cast<uint64_t>((16 * singlecore_m2_div_align)));
    OP_LOGD(OP_NAME, "singlecore_m2 = %u", tiling_data.get_singlecore_m2());
    tiling_data.set_base_m1(static_cast<uint64_t>((Max(16, (16 * pow(2, ub_m_base))) * base_m1_div_align)));
    OP_LOGD(OP_NAME, "base_m1 = %u", tiling_data.get_base_m1());
    tiling_data.set_base_m2(static_cast<uint64_t>((16 * base_m2_div_align)));
    OP_LOGD(OP_NAME, "base_m2 = %u", tiling_data.get_base_m2());
    tiling_data.set_ub_m(static_cast<uint64_t>((16 * pow(2, ub_m_base))));
    OP_LOGD(OP_NAME, "ub_m = %u", tiling_data.get_ub_m());
    tiling_data.set_singlecore_n1(static_cast<uint64_t>((16 * singlecore_n1_div_align)));
    OP_LOGD(OP_NAME, "singlecore_n1 = %u", tiling_data.get_singlecore_n1());
    tiling_data.set_base_n1(static_cast<uint64_t>((16 * base_n1_div_align)));
    OP_LOGD(OP_NAME, "base_n1 = %u", tiling_data.get_base_n1());
    tiling_data.set_singlecore_n2(static_cast<uint64_t>((16 * singlecore_n2_div_align)));
    OP_LOGD(OP_NAME, "singlecore_n2 = %u", tiling_data.get_singlecore_n2());
    tiling_data.set_base_n2(static_cast<uint64_t>((16 * base_n2_div_align)));
    OP_LOGD(OP_NAME, "base_n2 = %u", tiling_data.get_base_n2());
}

bool GeneralSolvercase0::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase0::GetResult(int32_t solution_num, uint64_t* solution, FFNTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case0.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "btbuf_size = %ld", static_cast<uint64_t>(Getbtbuf_sizeCost(solution) + btbuf_size));
        OP_LOGD(OP_NAME, "l0c_size = %ld", static_cast<uint64_t>(Getl0c_sizeCost(solution) + l0c_size));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case0 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase0Impl : public TilingCaseImpl {
 public:
  TilingCase0Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  void GetTilingData(TilingDataCopy &from_tiling, FFNTilingData &to_tiling) {
    to_tiling.set_K1(from_tiling.get_K1());
    to_tiling.set_N1(from_tiling.get_N1());
    to_tiling.set_N2(from_tiling.get_N2());
    to_tiling.set_maxTokens(from_tiling.get_maxTokens());
    to_tiling.set_base_m1(from_tiling.get_base_m1());
    to_tiling.set_base_m2(from_tiling.get_base_m2());
    to_tiling.set_base_n1(from_tiling.get_base_n1());
    to_tiling.set_base_n2(from_tiling.get_base_n2());
    to_tiling.set_singlecore_m1(from_tiling.get_singlecore_m1());
    to_tiling.set_singlecore_m2(from_tiling.get_singlecore_m2());
    to_tiling.set_singlecore_n1(from_tiling.get_singlecore_n1());
    to_tiling.set_singlecore_n2(from_tiling.get_singlecore_n2());
    to_tiling.set_ub_m(from_tiling.get_ub_m());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_base_m1_loop_num(from_tiling.get_base_m1_loop_num());
    to_tiling.set_base_m1_tail_size(from_tiling.get_base_m1_tail_size());
    to_tiling.set_base_m1_tail_tile_ub_m_loop_num(from_tiling.get_base_m1_tail_tile_ub_m_loop_num());
    to_tiling.set_base_m1_tail_tile_ub_m_tail_size(from_tiling.get_base_m1_tail_tile_ub_m_tail_size());
    to_tiling.set_base_m2_loop_num(from_tiling.get_base_m2_loop_num());
    to_tiling.set_base_m2_tail_size(from_tiling.get_base_m2_tail_size());
    to_tiling.set_base_n1_loop_num(from_tiling.get_base_n1_loop_num());
    to_tiling.set_base_n1_tail_size(from_tiling.get_base_n1_tail_size());
    to_tiling.set_base_n2_loop_num(from_tiling.get_base_n2_loop_num());
    to_tiling.set_base_n2_tail_size(from_tiling.get_base_n2_tail_size());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_singlecore_m1_loop_num(from_tiling.get_singlecore_m1_loop_num());
    to_tiling.set_singlecore_m1_tail_size(from_tiling.get_singlecore_m1_tail_size());
    to_tiling.set_singlecore_m1_tail_tile_base_m1_loop_num(from_tiling.get_singlecore_m1_tail_tile_base_m1_loop_num());
    to_tiling.set_singlecore_m1_tail_tile_base_m1_tail_size(from_tiling.get_singlecore_m1_tail_tile_base_m1_tail_size());
    to_tiling.set_singlecore_m2_loop_num(from_tiling.get_singlecore_m2_loop_num());
    to_tiling.set_singlecore_m2_tail_size(from_tiling.get_singlecore_m2_tail_size());
    to_tiling.set_singlecore_m2_tail_tile_base_m2_loop_num(from_tiling.get_singlecore_m2_tail_tile_base_m2_loop_num());
    to_tiling.set_singlecore_m2_tail_tile_base_m2_tail_size(from_tiling.get_singlecore_m2_tail_tile_base_m2_tail_size());
    to_tiling.set_singlecore_n1_loop_num(from_tiling.get_singlecore_n1_loop_num());
    to_tiling.set_singlecore_n1_tail_size(from_tiling.get_singlecore_n1_tail_size());
    to_tiling.set_singlecore_n1_tail_tile_base_n1_loop_num(from_tiling.get_singlecore_n1_tail_tile_base_n1_loop_num());
    to_tiling.set_singlecore_n1_tail_tile_base_n1_tail_size(from_tiling.get_singlecore_n1_tail_tile_base_n1_tail_size());
    to_tiling.set_singlecore_n2_loop_num(from_tiling.get_singlecore_n2_loop_num());
    to_tiling.set_singlecore_n2_tail_size(from_tiling.get_singlecore_n2_tail_size());
    to_tiling.set_singlecore_n2_tail_tile_base_n2_loop_num(from_tiling.get_singlecore_n2_tail_tile_base_n2_loop_num());
    to_tiling.set_singlecore_n2_tail_tile_base_n2_tail_size(from_tiling.get_singlecore_n2_tail_tile_base_n2_tail_size());
    to_tiling.set_ub_m_loop_num(from_tiling.get_ub_m_loop_num());
    to_tiling.set_ub_m_tail_size(from_tiling.get_ub_m_tail_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  void SetTilingData(FFNTilingData &from_tiling, TilingDataCopy &to_tiling) {
    to_tiling.set_K1(from_tiling.get_K1());
    to_tiling.set_N1(from_tiling.get_N1());
    to_tiling.set_N2(from_tiling.get_N2());
    to_tiling.set_maxTokens(from_tiling.get_maxTokens());
    to_tiling.set_base_m1(from_tiling.get_base_m1());
    to_tiling.set_base_m2(from_tiling.get_base_m2());
    to_tiling.set_base_n1(from_tiling.get_base_n1());
    to_tiling.set_base_n2(from_tiling.get_base_n2());
    to_tiling.set_singlecore_m1(from_tiling.get_singlecore_m1());
    to_tiling.set_singlecore_m2(from_tiling.get_singlecore_m2());
    to_tiling.set_singlecore_n1(from_tiling.get_singlecore_n1());
    to_tiling.set_singlecore_n2(from_tiling.get_singlecore_n2());
    to_tiling.set_ub_m(from_tiling.get_ub_m());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_base_m1_loop_num(from_tiling.get_base_m1_loop_num());
    to_tiling.set_base_m1_tail_size(from_tiling.get_base_m1_tail_size());
    to_tiling.set_base_m1_tail_tile_ub_m_loop_num(from_tiling.get_base_m1_tail_tile_ub_m_loop_num());
    to_tiling.set_base_m1_tail_tile_ub_m_tail_size(from_tiling.get_base_m1_tail_tile_ub_m_tail_size());
    to_tiling.set_base_m2_loop_num(from_tiling.get_base_m2_loop_num());
    to_tiling.set_base_m2_tail_size(from_tiling.get_base_m2_tail_size());
    to_tiling.set_base_n1_loop_num(from_tiling.get_base_n1_loop_num());
    to_tiling.set_base_n1_tail_size(from_tiling.get_base_n1_tail_size());
    to_tiling.set_base_n2_loop_num(from_tiling.get_base_n2_loop_num());
    to_tiling.set_base_n2_tail_size(from_tiling.get_base_n2_tail_size());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_singlecore_m1_loop_num(from_tiling.get_singlecore_m1_loop_num());
    to_tiling.set_singlecore_m1_tail_size(from_tiling.get_singlecore_m1_tail_size());
    to_tiling.set_singlecore_m1_tail_tile_base_m1_loop_num(from_tiling.get_singlecore_m1_tail_tile_base_m1_loop_num());
    to_tiling.set_singlecore_m1_tail_tile_base_m1_tail_size(from_tiling.get_singlecore_m1_tail_tile_base_m1_tail_size());
    to_tiling.set_singlecore_m2_loop_num(from_tiling.get_singlecore_m2_loop_num());
    to_tiling.set_singlecore_m2_tail_size(from_tiling.get_singlecore_m2_tail_size());
    to_tiling.set_singlecore_m2_tail_tile_base_m2_loop_num(from_tiling.get_singlecore_m2_tail_tile_base_m2_loop_num());
    to_tiling.set_singlecore_m2_tail_tile_base_m2_tail_size(from_tiling.get_singlecore_m2_tail_tile_base_m2_tail_size());
    to_tiling.set_singlecore_n1_loop_num(from_tiling.get_singlecore_n1_loop_num());
    to_tiling.set_singlecore_n1_tail_size(from_tiling.get_singlecore_n1_tail_size());
    to_tiling.set_singlecore_n1_tail_tile_base_n1_loop_num(from_tiling.get_singlecore_n1_tail_tile_base_n1_loop_num());
    to_tiling.set_singlecore_n1_tail_tile_base_n1_tail_size(from_tiling.get_singlecore_n1_tail_tile_base_n1_tail_size());
    to_tiling.set_singlecore_n2_loop_num(from_tiling.get_singlecore_n2_loop_num());
    to_tiling.set_singlecore_n2_tail_size(from_tiling.get_singlecore_n2_tail_size());
    to_tiling.set_singlecore_n2_tail_tile_base_n2_loop_num(from_tiling.get_singlecore_n2_tail_tile_base_n2_loop_num());
    to_tiling.set_singlecore_n2_tail_tile_base_n2_tail_size(from_tiling.get_singlecore_n2_tail_tile_base_n2_tail_size());
    to_tiling.set_ub_m_loop_num(from_tiling.get_ub_m_loop_num());
    to_tiling.set_ub_m_tail_size(from_tiling.get_ub_m_tail_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  bool ExecuteGeneralSolver(FFNTilingData& tiling_data) {
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
    uint64_t N1 = tiling_data.get_N1();
    N1 = ((N1 + 16 - 1) / 16) * 16;
    uint64_t N2 = tiling_data.get_N2();
    N2 = ((N2 + 16 - 1) / 16) * 16;
    uint64_t maxTokens = tiling_data.get_maxTokens();
    maxTokens = ((maxTokens + 16 - 1) / 16) * 16;
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 9;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(base_m1_div_align, base_m2_div_align, base_n1_div_align, base_n2_div_align, singlecore_m1_div_align, singlecore_m2_div_align, singlecore_n1_div_align, singlecore_n2_div_align, ub_m_base), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(((double)(1)/(double)(16) * N1)), static_cast<uint64_t>(((double)(1)/(double)(16) * N2)), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(0)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(((double)(1)/(double)(16) * maxTokens)), static_cast<uint64_t>(((double)(1)/(double)(16) * maxTokens)), static_cast<uint64_t>(((double)(1)/(double)(16) * N1)), static_cast<uint64_t>(((double)(1)/(double)(16) * N2)), static_cast<uint64_t>(((double)(1)/(double)(16) * maxTokens)), static_cast<uint64_t>(((double)(1)/(double)(16) * maxTokens)), static_cast<uint64_t>(((double)(1)/(double)(16) * N1)), static_cast<uint64_t>(((double)(1)/(double)(16) * N2)), static_cast<uint64_t>((log(((double)(1)/(double)(16) * maxTokens)) / (log(2))))};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(0)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, false, true, true, false, false, false, false, false};
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
    input.corenum = corenum_;
    input.cur_vars = init_vars;
    input.upper_bound = upper_bound;
    input.lower_bound = lower_bound;
    input.update_last = update_last;
    OP_LOGD(OP_NAME, "base_m1_div_align->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "base_m2_div_align->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);
    OP_LOGD(OP_NAME, "base_n1_div_align->init value: %lu, range: [%lu, %lu].", init_vars[2], lower_bound[2], upper_bound[2]);
    OP_LOGD(OP_NAME, "base_n2_div_align->init value: %lu, range: [%lu, %lu].", init_vars[3], lower_bound[3], upper_bound[3]);
    OP_LOGD(OP_NAME, "singlecore_m1_div_align->init value: %lu, range: [%lu, %lu].", init_vars[4], lower_bound[4], upper_bound[4]);
    OP_LOGD(OP_NAME, "singlecore_m2_div_align->init value: %lu, range: [%lu, %lu].", init_vars[5], lower_bound[5], upper_bound[5]);
    OP_LOGD(OP_NAME, "singlecore_n1_div_align->init value: %lu, range: [%lu, %lu].", init_vars[6], lower_bound[6], upper_bound[6]);
    OP_LOGD(OP_NAME, "singlecore_n2_div_align->init value: %lu, range: [%lu, %lu].", init_vars[7], lower_bound[7], upper_bound[7]);
    OP_LOGD(OP_NAME, "ub_m_base->init value: %lu, range: [%lu, %lu].", init_vars[8], lower_bound[8], upper_bound[8]);

    std::shared_ptr<GeneralSolvercase0> solver = std::make_shared<GeneralSolvercase0>(cfg, tiling_data);
    if (solver != nullptr) {
        // 导入通用求解器的输入参数并完成初始化
        OP_LOGD(OP_NAME, "Start initializing the input.");
        if (solver -> Init(input)) {
            // 运行通用求解器并获取算法的解
            OP_LOGD(OP_NAME, "Intialization finished, start running the solver.");
            if (solver -> Run(solution_num, solution)) {
                solver -> GetResult(solution_num, solution, tiling_data);
                delete[] solution;
                OP_LOGD(OP_NAME, "The solver executed successfully.");
                return true;
            }
            OP_LOGW(OP_NAME, "Failed to find any solution.");
        }
    }
    if (solution != nullptr) {
        delete[] solution;
    }
    OP_LOGW(OP_NAME, "The solver executed failed.");
    return false;
  }

  bool DoTiling(FFNTilingData &tiling_data) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case0.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case0 successfully.");

    return true;
  }

  int Getl0c_size(FFNTilingData& tiling_data) {
    double base_m1 = tiling_data.get_base_m1();
    double base_m2 = tiling_data.get_base_m2();
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();

    return Max((4 * base_m1 * base_n1), (4 * base_m2 * base_n2));
  }

  int Getub_size(FFNTilingData& tiling_data) {
    double base_n1 = tiling_data.get_base_n1();
    double ub_m = tiling_data.get_ub_m();

    return (10 * base_n1 * ub_m);
  }

  int Getbtbuf_size(FFNTilingData& tiling_data) {
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();

    return Max((4 * base_n1), (4 * base_n2));
  }

  int Getblock_dim(FFNTilingData& tiling_data) {
    double N1 = tiling_data.get_N1();
    double N2 = tiling_data.get_N2();
    double maxTokens = tiling_data.get_maxTokens();
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_m2 = tiling_data.get_singlecore_m2();
    double singlecore_n1 = tiling_data.get_singlecore_n1();
    double singlecore_n2 = tiling_data.get_singlecore_n2();

    return Max((ceiling((N2 / (singlecore_n2))) * ceiling((maxTokens / (singlecore_m2)))), (ceiling((N1 / (singlecore_n1))) * ceiling((maxTokens / (singlecore_m1)))));
  }

  double GetAIC_MTE2(FFNTilingData& tiling_data) {
    double K1 = tiling_data.get_K1();
    double N1 = tiling_data.get_N1();
    double base_m1 = tiling_data.get_base_m1();
    double base_m2 = tiling_data.get_base_m2();
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_m2 = tiling_data.get_singlecore_m2();
    double singlecore_n1 = tiling_data.get_singlecore_n1();
    double singlecore_n2 = tiling_data.get_singlecore_n2();

    return (((((double)(1)/(double)(32) * K1 * Max(1, (256 / (K1))) * base_m1) + ((double)(1)/(double)(32) * K1 * Max(1, (256 / (base_n1))) * base_n1)) * singlecore_m1 * singlecore_n1 / (base_m1 * base_n1)) + ((((double)(1)/(double)(32) * Max(1, (256 / (N1))) * N1 * base_m2) + ((double)(1)/(double)(32) * Max(1, (256 / (base_n2))) * N1 * base_n2)) * singlecore_m2 * singlecore_n2 / (base_m2 * base_n2)));
  }

  double GetAIC_MAC(FFNTilingData& tiling_data) {
    double K1 = tiling_data.get_K1();
    double N1 = tiling_data.get_N1();
    double base_m1 = tiling_data.get_base_m1();
    double base_m2 = tiling_data.get_base_m2();
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_m2 = tiling_data.get_singlecore_m2();
    double singlecore_n1 = tiling_data.get_singlecore_n1();
    double singlecore_n2 = tiling_data.get_singlecore_n2();

    return ((ceiling(((double)(1)/(double)(16) * K1)) * ceiling(((double)(1)/(double)(16) * base_m1)) * ceiling(((double)(1)/(double)(16) * base_n1)) * singlecore_m1 * singlecore_n1 / (base_m1 * base_n1)) + (ceiling(((double)(1)/(double)(16) * N1)) * ceiling(((double)(1)/(double)(16) * base_m2)) * ceiling(((double)(1)/(double)(16) * base_n2)) * singlecore_m2 * singlecore_n2 / (base_m2 * base_n2)));
  }

  double GetAIV_MTE2(FFNTilingData& tiling_data) {
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_n1 = tiling_data.get_singlecore_n1();

    return ((double)(1)/(double)(16) * singlecore_m1 * singlecore_n1);
  }

  double GetAIV_MTE3(FFNTilingData& tiling_data) {
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_n1 = tiling_data.get_singlecore_n1();

    return ((double)(1)/(double)(32) * singlecore_m1 * singlecore_n1);
  }

  double GetAIV_VEC(FFNTilingData& tiling_data) {
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_n1 = tiling_data.get_singlecore_n1();

    return ((double)(1)/(double)(64) * singlecore_m1 * singlecore_n1);
  }

  double GetPerf(FFNTilingData& tiling_data) {
    double K1 = tiling_data.get_K1();
    double N1 = tiling_data.get_N1();
    double base_m1 = tiling_data.get_base_m1();
    double base_m2 = tiling_data.get_base_m2();
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();
    double singlecore_m1 = tiling_data.get_singlecore_m1();
    double singlecore_m2 = tiling_data.get_singlecore_m2();
    double singlecore_n1 = tiling_data.get_singlecore_n1();
    double singlecore_n2 = tiling_data.get_singlecore_n2();

    double AIC_MTE2 = (((((double)(1)/(double)(32) * K1 * Max(1, (256 / (K1))) * base_m1) + ((double)(1)/(double)(32) * K1 * Max(1, (256 / (base_n1))) * base_n1)) * singlecore_m1 * singlecore_n1 / (base_m1 * base_n1)) + ((((double)(1)/(double)(32) * Max(1, (256 / (N1))) * N1 * base_m2) + ((double)(1)/(double)(32) * Max(1, (256 / (base_n2))) * N1 * base_n2)) * singlecore_m2 * singlecore_n2 / (base_m2 * base_n2)));
    double AIC_MAC = ((ceiling(((double)(1)/(double)(16) * K1)) * ceiling(((double)(1)/(double)(16) * base_m1)) * ceiling(((double)(1)/(double)(16) * base_n1)) * singlecore_m1 * singlecore_n1 / (base_m1 * base_n1)) + (ceiling(((double)(1)/(double)(16) * N1)) * ceiling(((double)(1)/(double)(16) * base_m2)) * ceiling(((double)(1)/(double)(16) * base_n2)) * singlecore_m2 * singlecore_n2 / (base_m2 * base_n2)));
    double AIV_MTE2 = ((double)(1)/(double)(16) * singlecore_m1 * singlecore_n1);
    double AIV_MTE3 = ((double)(1)/(double)(32) * singlecore_m1 * singlecore_n1);
    double AIV_VEC = ((double)(1)/(double)(64) * singlecore_m1 * singlecore_n1);

    return Max(Max(Max(Max(AIC_MAC, AIV_VEC), AIC_MTE2), AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(FFNTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_maxTokens() + tiling_data.get_singlecore_m1()) - 1) / tiling_data.get_singlecore_m1()) * (((tiling_data.get_maxTokens() + tiling_data.get_singlecore_m2()) - 1) / tiling_data.get_singlecore_m2()) * (((tiling_data.get_N1() + tiling_data.get_singlecore_n1()) - 1) / tiling_data.get_singlecore_n1()) * (((tiling_data.get_N2() + tiling_data.get_singlecore_n2()) - 1) / tiling_data.get_singlecore_n2()));
  }

  void UpdateAxesTilingData(FFNTilingData& tiling_data) {
    tiling_data.set_ub_m_loop_num(((tiling_data.get_base_m1() + tiling_data.get_ub_m()) - 1) / tiling_data.get_ub_m());
    tiling_data.set_singlecore_n2_loop_num(((tiling_data.get_N2() + tiling_data.get_singlecore_n2()) - 1) / tiling_data.get_singlecore_n2());
    tiling_data.set_base_n2_loop_num(((tiling_data.get_singlecore_n2() + tiling_data.get_base_n2()) - 1) / tiling_data.get_base_n2());
    tiling_data.set_singlecore_n1_loop_num(((tiling_data.get_N1() + tiling_data.get_singlecore_n1()) - 1) / tiling_data.get_singlecore_n1());
    tiling_data.set_base_n1_loop_num(((tiling_data.get_singlecore_n1() + tiling_data.get_base_n1()) - 1) / tiling_data.get_base_n1());
    tiling_data.set_singlecore_m2_loop_num(((tiling_data.get_maxTokens() + tiling_data.get_singlecore_m2()) - 1) / tiling_data.get_singlecore_m2());
    tiling_data.set_base_m2_loop_num(((tiling_data.get_singlecore_m2() + tiling_data.get_base_m2()) - 1) / tiling_data.get_base_m2());
    tiling_data.set_singlecore_m1_loop_num(((tiling_data.get_maxTokens() + tiling_data.get_singlecore_m1()) - 1) / tiling_data.get_singlecore_m1());
    tiling_data.set_base_m1_loop_num(((tiling_data.get_singlecore_m1() + tiling_data.get_base_m1()) - 1) / tiling_data.get_base_m1());
    tiling_data.set_singlecore_m1_tail_size((tiling_data.get_maxTokens() % tiling_data.get_singlecore_m1()) == 0 ? tiling_data.get_singlecore_m1() : (tiling_data.get_maxTokens() % tiling_data.get_singlecore_m1()));
    tiling_data.set_base_m2_tail_size((tiling_data.get_singlecore_m2() % tiling_data.get_base_m2()) == 0 ? tiling_data.get_base_m2() : (tiling_data.get_singlecore_m2() % tiling_data.get_base_m2()));
    tiling_data.set_ub_m_tail_size((tiling_data.get_base_m1() % tiling_data.get_ub_m()) == 0 ? tiling_data.get_ub_m() : (tiling_data.get_base_m1() % tiling_data.get_ub_m()));
    tiling_data.set_singlecore_n2_tail_size((tiling_data.get_N2() % tiling_data.get_singlecore_n2()) == 0 ? tiling_data.get_singlecore_n2() : (tiling_data.get_N2() % tiling_data.get_singlecore_n2()));
    tiling_data.set_singlecore_m2_tail_size((tiling_data.get_maxTokens() % tiling_data.get_singlecore_m2()) == 0 ? tiling_data.get_singlecore_m2() : (tiling_data.get_maxTokens() % tiling_data.get_singlecore_m2()));
    tiling_data.set_base_n1_tail_size((tiling_data.get_singlecore_n1() % tiling_data.get_base_n1()) == 0 ? tiling_data.get_base_n1() : (tiling_data.get_singlecore_n1() % tiling_data.get_base_n1()));
    tiling_data.set_base_m1_tail_size((tiling_data.get_singlecore_m1() % tiling_data.get_base_m1()) == 0 ? tiling_data.get_base_m1() : (tiling_data.get_singlecore_m1() % tiling_data.get_base_m1()));
    tiling_data.set_base_n2_tail_size((tiling_data.get_singlecore_n2() % tiling_data.get_base_n2()) == 0 ? tiling_data.get_base_n2() : (tiling_data.get_singlecore_n2() % tiling_data.get_base_n2()));
    tiling_data.set_singlecore_n1_tail_size((tiling_data.get_N1() % tiling_data.get_singlecore_n1()) == 0 ? tiling_data.get_singlecore_n1() : (tiling_data.get_N1() % tiling_data.get_singlecore_n1()));
    tiling_data.set_singlecore_m1_tail_tile_base_m1_loop_num(((tiling_data.get_singlecore_m1_tail_size() + tiling_data.get_base_m1()) - 1) / tiling_data.get_base_m1());
    tiling_data.set_singlecore_n2_tail_tile_base_n2_loop_num(((tiling_data.get_singlecore_n2_tail_size() + tiling_data.get_base_n2()) - 1) / tiling_data.get_base_n2());
    tiling_data.set_base_m1_tail_tile_ub_m_loop_num(((tiling_data.get_base_m1_tail_size() + tiling_data.get_ub_m()) - 1) / tiling_data.get_ub_m());
    tiling_data.set_singlecore_n1_tail_tile_base_n1_loop_num(((tiling_data.get_singlecore_n1_tail_size() + tiling_data.get_base_n1()) - 1) / tiling_data.get_base_n1());
    tiling_data.set_singlecore_m2_tail_tile_base_m2_loop_num(((tiling_data.get_singlecore_m2_tail_size() + tiling_data.get_base_m2()) - 1) / tiling_data.get_base_m2());
    tiling_data.set_singlecore_n1_tail_tile_base_n1_tail_size((tiling_data.get_singlecore_n1_tail_size() % tiling_data.get_base_n1()) == 0 ? tiling_data.get_base_n1() : (tiling_data.get_singlecore_n1_tail_size() % tiling_data.get_base_n1()));
    tiling_data.set_base_m1_tail_tile_ub_m_tail_size((tiling_data.get_base_m1_tail_size() % tiling_data.get_ub_m()) == 0 ? tiling_data.get_ub_m() : (tiling_data.get_base_m1_tail_size() % tiling_data.get_ub_m()));
    tiling_data.set_singlecore_n2_tail_tile_base_n2_tail_size((tiling_data.get_singlecore_n2_tail_size() % tiling_data.get_base_n2()) == 0 ? tiling_data.get_base_n2() : (tiling_data.get_singlecore_n2_tail_size() % tiling_data.get_base_n2()));
    tiling_data.set_singlecore_m2_tail_tile_base_m2_tail_size((tiling_data.get_singlecore_m2_tail_size() % tiling_data.get_base_m2()) == 0 ? tiling_data.get_base_m2() : (tiling_data.get_singlecore_m2_tail_size() % tiling_data.get_base_m2()));
    tiling_data.set_singlecore_m1_tail_tile_base_m1_tail_size((tiling_data.get_singlecore_m1_tail_size() % tiling_data.get_base_m1()) == 0 ? tiling_data.get_base_m1() : (tiling_data.get_singlecore_m1_tail_size() % tiling_data.get_base_m1()));
  }

  void ComputeOptionParam(FFNTilingData &tiling_data) {

  }

  void ExtraTilingData(FFNTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 0.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 0 successfully.");
  }

  void GetWorkSpaceSize(FFNTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 0.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 0.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(FFNTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set base_m1 to %u.", tiling_data.get_base_m1());
    OP_LOGI(OP_NAME, "Set base_m2 to %u.", tiling_data.get_base_m2());
    OP_LOGI(OP_NAME, "Set base_n1 to %u.", tiling_data.get_base_n1());
    OP_LOGI(OP_NAME, "Set base_n2 to %u.", tiling_data.get_base_n2());
    OP_LOGI(OP_NAME, "Set singlecore_m1 to %u.", tiling_data.get_singlecore_m1());
    OP_LOGI(OP_NAME, "Set singlecore_m2 to %u.", tiling_data.get_singlecore_m2());
    OP_LOGI(OP_NAME, "Set singlecore_n1 to %u.", tiling_data.get_singlecore_n1());
    OP_LOGI(OP_NAME, "Set singlecore_n2 to %u.", tiling_data.get_singlecore_n2());
    OP_LOGI(OP_NAME, "Set ub_m to %u.", tiling_data.get_ub_m());
    OP_LOGI(OP_NAME, "The value of l0c_size is %d.", Getl0c_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of btbuf_size is %d.", Getbtbuf_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIC_MTE2 is %f.", GetAIC_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIC_MAC is %f.", GetAIC_MAC(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_VEC is %f.", GetAIV_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 0u) {
    tilingCaseImplPtr = std::make_shared<TilingCase0Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
bool FindPerfBetterTilingbyCaseId(uint32_t corenum, double &obj, TilingDataCopy &tmp_tiling, FFNTilingData &tiling_data, uint32_t tilingCaseId) {
  double cur_obj;
  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
  if (tilingCaseImplPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
    return false;
  }
  if (tilingCaseImplPtr->GetTiling(tiling_data)) {
    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);
    OP_LOGD(OP_NAME, "The optimal objection for tilingCaseId %u is %f.", tilingCaseId, cur_obj);
    if (obj < 0 || cur_obj < obj) {
      OP_LOGD(OP_NAME, "The solution for tilingCaseId %u is better, updating the tiling data.", tilingCaseId);
      tiling_data.set_tiling_key(tilingCaseId);
      tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);
      OP_LOGD(OP_NAME, "Set the output tiling data.");
      obj = cur_obj;
      OP_LOGD(OP_NAME, "Updated the best tilingCaseId to %u.", tilingCaseId);
    } else {
      tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
    }
    return true;
  }
  return false;
}

bool GetTilingKey(FFNTilingData &tiling_data, int32_t tilingCaseId = -1) {
  bool ret = false;
  double obj = -1;
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    TilingDataCopy tmp_tiling;
    uint32_t tilingKeys[1] = {0u};
    for (const auto &tilingKey : tilingKeys) {
      ret = (FindPerfBetterTilingbyCaseId(corenum, obj, tmp_tiling, tiling_data, tilingKey) || ret);
      OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId %u.", tilingKey);
    }
    if (ret) {
      OP_LOGI(OP_NAME, "Among the templates, tiling case %u is the best choice.", tiling_data.get_tiling_key());
    }
  } else {
    OP_LOGI(OP_NAME, "Calculating the tiling data for tilingCaseId %u.", tilingCaseId);
    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
    if (tilingCaseImplPtr == nullptr) {
      OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
      return false;
    }
    ret = tilingCaseImplPtr->GetTiling(tiling_data);
  }
  if (!ret) {
    OP_LOGE(OP_NAME, "Failed to execute tiling func.");
  }
  return ret;
}

bool GetTiling(FFNTilingData &tiling_data, int32_t tilingCaseId) {
  OP_LOGI(OP_NAME, "Start tiling. Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context. End tiling.");
  return true;
}

} // namespace optiling

