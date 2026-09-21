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
#include "runtime2_util.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "AddLayerNorm"

namespace {
enum DurationType {
  TILING_FUNC_DURATION_TOTAL = 0,
  TILING_FUNC_DURATION_DOTILING,
  TILING_FUNC_DURATION_MAX,
};

struct DurationDef {
  std::string name;
};

DurationDef g_duration_def[TILING_FUNC_DURATION_MAX] = {
  {"TILING_FUNC_DURATION_TOTAL"},
  {"TILING_FUNC_DURATION_DOTILING"},
};

class Duration {
 public:
  Duration(const std::string &name): name_(name) {}

  ~Duration() {
    Print();
  }
  void Begin() {
    call_start_ = Now();
  }

  void End() {
    auto now = Now();
    uint64_t duration = now - call_start_;
    total_count_++;
    total_time_ += duration;
    if (now > max_time_) max_time_ = now;
    if (now < min_time_) min_time_ = now;
  }

private:
  void Print() {
    if (total_count_ == 0ULL) return;
    OP_LOGI(OP_NAME, "Duration record: name[%s], total_count[%lu], total_time[%lu], max_time[%lu], min_time[%lu], average_time[%lu].",
      name_.c_str(), total_count_, total_time_, max_time_, min_time_,
      static_cast<uint64_t>(total_count_ / total_count_));
  } 

  uint64_t Now() {
    static auto zero = std::chrono::system_clock::now();
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now() - zero).count();
    return static_cast<uint64_t>(now);
  }

  std::string name_;
  uint64_t total_count_ = 0ULL;
  uint64_t total_time_ = 0ULL;
  uint64_t max_time_ = 0ULL;
  uint64_t min_time_ = UINT64_MAX;
  uint64_t call_start_ = 0ULL;
};

struct DurationInfo {
  std::unique_ptr<Duration> stat;
};

class DurationManager {
public:
  static DurationManager &GetInstance() {
    static DurationManager ins;
    return ins;
  }

  DurationManager() {
    for (uint32_t index = 0U; index < static_cast<uint32_t>(TILING_FUNC_DURATION_MAX); index++) {
      AddDuration(index, g_duration_def[index].name);
    }
  }
  
  void AddDuration(const uint32_t type, const std::string &name) {
    duration_infos_[type].stat = std::unique_ptr<Duration>(new(std::nothrow) Duration(name));
    if (duration_infos_[type].stat == nullptr) {
      OP_LOGW(OP_NAME, "Create Duration failed.");
    }
  }

  void Begin(const DurationType type) {
    const auto &stat = duration_infos_[type].stat;
    if (stat == nullptr) {
      return;
    }
    stat->Begin();
  }

  void End(const DurationType type) {
    const auto &stat = duration_infos_[type].stat;
    if (stat == nullptr) {
      return;
    }
    stat->End();
  }
private:
  DurationInfo duration_infos_[TILING_FUNC_DURATION_MAX];
};

static inline void DurationBegin(const DurationType type) {
  DurationManager::GetInstance().Begin(type);
}

static inline void DurationEnd(const DurationType type) {
  DurationManager::GetInstance().End(type);
}

class DurationGuard {
public:
  DurationGuard(const DurationType type) : type_(type)
  {
    DurationBegin(type);
  }

  ~DurationGuard() {
    DurationEnd(type_);
  }
private:
  DurationType type_;
};

#define DURATION_GUARD(type) DurationGuard g_duration##__COUNTER__(type);
} // namespace

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
  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}
  virtual ~TilingCaseImpl() = default;
  bool GetTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
DURATION_GUARD(TILING_FUNC_DURATION_TOTAL)
    if (!GetShapeAttrsInfo(tiling_data, context)) {
      OP_LOGW(OP_NAME, "Failed to get shape attrs.");
      return false;
    }
    if (!CheckIsCapable(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to check capable.");
      return false;
    }
    if (!DoTiling(tiling_data, context)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    GetWorkSpaceSize(tiling_data);
    ExtraTilingData(tiling_data, context);
    TilingSummary(tiling_data);
    return true;
  }
  virtual void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) = 0;
  virtual void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) = 0;
  virtual double GetPerf(graph_normalTilingData &tiling_data) { return 0.0; }
 protected:
  virtual bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) = 0;
  virtual bool CheckIsCapable(graph_normalTilingData &tiling_data) = 0;
  virtual bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) = 0;
  virtual void DoApiTiling(graph_normalTilingData &tiling_data) {}
  virtual void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {}
  virtual void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {}
  virtual void TilingSummary(graph_normalTilingData &tiling_data) = 0;
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

bool GetPlatformInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer platformInfoPtr is null.");
    return false;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  if ((aivNum == 0) || (aicNum == 0) || (ub_size == 0)) {
    OP_LOGE(OP_NAME, "Get incorrect platform value.");
    return false;
  } 
  OP_LOGD(OP_NAME, "PlatformInfo is valid.");
  tiling_data.set_block_dim(ascendcPlatform.GetCoreNumAiv());
  OP_LOGD(OP_NAME, "Set block dim to %d.", tiling_data.get_block_dim());
  tiling_data.set_ub_size(ub_size);
  OP_LOGD(OP_NAME, "Set ub_size to %d.", tiling_data.get_ub_size());

  return true;
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
    cons_info_->leqs[0] = ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size)
    cons_info_->leqs[1] = (Max(0, ceiling((A / (nbo_size)))) - block_dim)
    cons_info_->leqs[2] = (nbo_size - A)
    cons_info_->leqs[3] = (nio_size - nbo_size)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1101 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1101(SolverConfig& config, graph_normalTilingData& tiling_data) : GeneralSolver(config) {
            A = tiling_data.get_A();
            R = tiling_data.get_R();
            ub_size = tiling_data.get_ub_size();
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
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t nbo_size_idx = 0;
        const int64_t nio_size_idx = 1;
        uint64_t BL{8};
        uint64_t A;
        uint64_t R;
        uint64_t ub_size;
        uint64_t block_dim{0};
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1101::Getblock_dimCost(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    return (Max(0, ceiling((A / (nbo_size)))) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1101::GetSmoothblock_dimCost(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    return (Max(0, ((A / (nbo_size)))) - block_dim);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1101::Getub_sizeCost(uint64_t* vars)
{
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    return ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1101::GetSmoothub_sizeCost(uint64_t* vars)
{
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    return ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1101::GetObj(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    double AICORE_VEC = (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ceiling((nbo_size / (nio_size))));
    OP_LOGD(OP_NAME, "AICORE_VEC = %f", AICORE_VEC);
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 3 * ceiling((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 2 * ceiling((nbo_size / (nio_size)))));
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ceiling((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ceiling((nbo_size / (nio_size)))));
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1101::GetSmoothObj(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    double AICORE_VEC = (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ((nbo_size / (nio_size))));
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 3 * ((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 2 * ((nbo_size / (nio_size)))));
    double AIV_MTE3 = ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ((nbo_size / (nio_size)))));
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1101::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
double GeneralSolvercase1101::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + ub_size_cost;
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
double GeneralSolvercase1101::GetLeqDiff(uint64_t* vars, double* weight)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = (nbo_size - A);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = (nio_size - nbo_size);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + ub_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase1101::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == nbo_size_idx) {
        return leqs[1] <= 0 && leqs[2] <= 0 && leqs[3] <= 0;
    } else if (idx == nio_size_idx) {
        return leqs[0] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase1101::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    if (idx == nbo_size_idx) {
        leqs[1] = (Max(0, ceiling((A / (nbo_size)))) - block_dim);
        leqs[2] = (nbo_size - A);
        leqs[3] = (nio_size - nbo_size);
    } else if (idx == nio_size_idx) {
        leqs[0] = ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
        leqs[3] = (nio_size - nbo_size);
    } else if (idx == -1) {
        leqs[0] = ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
        leqs[1] = (Max(0, ceiling((A / (nbo_size)))) - block_dim);
        leqs[2] = (nbo_size - A);
        leqs[3] = (nio_size - nbo_size);
    }
}

void GeneralSolvercase1101::DisplayVarVal(uint64_t* vars)
{
    uint64_t nbo_size = vars[nbo_size_idx];
    uint64_t nio_size = vars[nio_size_idx];
    OP_LOGD(OP_NAME, "nio_size = %lu", static_cast<uint64_t>(nio_size));
    OP_LOGD(OP_NAME, "nbo_size = %lu", static_cast<uint64_t>(nbo_size));
}

void GeneralSolvercase1101::MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data)
{
    uint64_t nbo_size = vars[nbo_size_idx];
    uint64_t nio_size = vars[nio_size_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1101 is:");
    tiling_data.set_nio_size(static_cast<uint64_t>(nio_size));
    OP_LOGD(OP_NAME, "nio_size = %u", tiling_data.get_nio_size());
    tiling_data.set_nbo_size(static_cast<uint64_t>(nbo_size));
    OP_LOGD(OP_NAME, "nbo_size = %u", tiling_data.get_nbo_size());
}

bool GeneralSolvercase1101::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase1101::GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1101.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1101 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1101Impl : public TilingCaseImpl {
 public:
  TilingCase1101Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != 5) {
      OP_LOGW(OP_NAME, "context->GetComputeNodeInfo()->GetInputsNum() != 5, current value is [%lu], invalid input num.", context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }

  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(0)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(1)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(2)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(3)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(4)->GetDataType());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }

  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(0)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(1)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(2)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(3)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(4)->GetStorageFormat());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }

  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input0 is too small.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input1 is too small.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input2 is too small.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input3 is too small.");
      return false;
    }
    if (input4_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input4 is too small.");
      return false;
    }
    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input4_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }

  bool TilingVarsShapeCheck(gert::TilingContext *context) {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 2; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input4.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = context->GetInputShape(4)->GetStorageShape().GetDim(input4_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool CheckInput0Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 0 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput1Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 1 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput2Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 2 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput3Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 3 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput4Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 4 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool TilingVarsCoverCheck(gert::TilingContext *context) {
    if (!CheckInput0Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 0 cannot cover all axis.");
      return false;
    }
    if (!CheckInput1Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 1 cannot cover all axis.");
      return false;
    }
    if (!CheckInput2Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 2 cannot cover all axis.");
      return false;
    }
    if (!CheckInput3Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 3 cannot cover all axis.");
      return false;
    }
    if (!CheckInput4Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 4 cannot cover all axis.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsCoverCheck success.");
    return true;
  }


  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsCoverCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }

  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    if (additional_output_ptr == nullptr) {
      OP_LOGW(OP_NAME, "Attr for additional_output_ptr is null.");
      return false;
    }
    int32_t additional_output = *additional_output_ptr;
    if ((additional_output < 1) || (additional_output > 1)) {
      OP_LOGW(OP_NAME, "(additional_output < 1) || (additional_output > 1), invalid optional att.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (TilingInputVarsCheck(context) != true) {
      return false;
    }
    if (TilingAttrCheck(context) != true) {
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1101.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1101.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(tiling_data.get_A());
    to_tiling.set_R(tiling_data.get_R());
    to_tiling.set_ub_size(tiling_data.get_ub_size());
  }
  void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) {
    tiling_data.set_A(from_tiling.get_A());
    tiling_data.set_R(from_tiling.get_R());
    tiling_data.set_BL(8);
    tiling_data.set_nbo_size(from_tiling.get_nbo_size());
    tiling_data.set_nio_size(from_tiling.get_nio_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_A_aligned_size(from_tiling.get_A_aligned_size());
    tiling_data.set_Q0(from_tiling.get_Q0());
    tiling_data.set_Q1(from_tiling.get_Q1());
    tiling_data.set_Q2(from_tiling.get_Q2());
    tiling_data.set_Q3(from_tiling.get_Q3());
    tiling_data.set_Q4(from_tiling.get_Q4());
    tiling_data.set_Q5(from_tiling.get_Q5());
    tiling_data.set_Q6(from_tiling.get_Q6());
    tiling_data.set_Q7(from_tiling.get_Q7());
    tiling_data.set_Q8(from_tiling.get_Q8());
    tiling_data.set_Q9(from_tiling.get_Q9());
    tiling_data.set_R_aligned_size(from_tiling.get_R_aligned_size());
    tiling_data.set_additional_output(from_tiling.get_additional_output());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_nbo_loop_num(from_tiling.get_nbo_loop_num());
    tiling_data.set_nbo_tail_size(from_tiling.get_nbo_tail_size());
    tiling_data.set_nbo_tail_tile_nio_loop_num(from_tiling.get_nbo_tail_tile_nio_loop_num());
    tiling_data.set_nbo_tail_tile_nio_tail_size(from_tiling.get_nbo_tail_tile_nio_tail_size());
    tiling_data.set_nio_loop_num(from_tiling.get_nio_loop_num());
    tiling_data.set_nio_tail_size(from_tiling.get_nio_tail_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    tiling_data.set_output1_total_size(from_tiling.get_output1_total_size());
    tiling_data.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    tiling_data.set_output2_total_size(from_tiling.get_output2_total_size());
    tiling_data.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    tiling_data.set_output3_total_size(from_tiling.get_output3_total_size());

    tiling_data.set_workspaceSize(from_tiling.get_workspaceSize());
  }
  bool ExecuteGeneralSolver(graph_normalTilingData& tiling_data) {
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
    uint64_t A = tiling_data.get_A();
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(nbo_size, nio_size), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(A), static_cast<uint64_t>(A)};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, false};
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
    OP_LOGD(OP_NAME, "nbo_size->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "nio_size->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    std::shared_ptr<GeneralSolvercase1101> solver = std::make_shared<GeneralSolvercase1101>(cfg, tiling_data);
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

  bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1101.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1101 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nio_size = tiling_data.get_nio_size();

    return ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double nbo_size = tiling_data.get_nbo_size();

    return Max(0, ceiling((A / (nbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 3 * ceiling((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 2 * ceiling((nbo_size / (nio_size)))));
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ceiling((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ceiling((nbo_size / (nio_size)))));
  }

  double GetAICORE_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ceiling((nbo_size / (nio_size))));
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 3 * ceiling((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 2 * ceiling((nbo_size / (nio_size)))));
    double AIV_MTE3 = ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ceiling((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ceiling((nbo_size / (nio_size)))));
    double AICORE_VEC = (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ceiling((nbo_size / (nio_size))));

    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_nio_loop_num(((tiling_data.get_nbo_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_loop_num(((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size());
    tiling_data.set_nio_tail_size((tiling_data.get_nbo_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_size() % tiling_data.get_nio_size()));
    tiling_data.set_nbo_tail_size((tiling_data.get_A() % tiling_data.get_nbo_size()) == 0 ? tiling_data.get_nbo_size() : (tiling_data.get_A() % tiling_data.get_nbo_size()));
    tiling_data.set_nbo_tail_tile_nio_loop_num(((tiling_data.get_nbo_tail_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_tail_tile_nio_tail_size((tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q0((2 * R * nio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q1((2 * R * nio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q2((2 * R * nio_size));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q3((4 * nio_size));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q4((2 * R * nio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q5((4 * R * nio_size));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q6(Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q7((4 * nio_size));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q8((2 * R));
  }

  void SetQ9(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q9((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);
    SetQ9(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1101.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1101 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1101.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1101 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1101.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1101.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set nbo_size to %u.", tiling_data.get_nbo_size());
    OP_LOGI(OP_NAME, "Set nio_size to %u.", tiling_data.get_nio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AICORE_VEC is %f.", GetAICORE_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

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
    cons_info_->leqs[0] = ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size)
    cons_info_->leqs[1] = (Max(0, ceiling((A / (nbo_size)))) - block_dim)
    cons_info_->leqs[2] = (nbo_size - A)
    cons_info_->leqs[3] = (nio_size - nbo_size)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1102 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1102(SolverConfig& config, graph_normalTilingData& tiling_data) : GeneralSolver(config) {
            A = tiling_data.get_A();
            R = tiling_data.get_R();
            ub_size = tiling_data.get_ub_size();
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
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t nbo_size_idx = 0;
        const int64_t nio_size_idx = 1;
        uint64_t BL{8};
        uint64_t A;
        uint64_t R;
        uint64_t ub_size;
        uint64_t block_dim{0};
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1102::Getblock_dimCost(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    return (Max(0, ceiling((A / (nbo_size)))) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1102::GetSmoothblock_dimCost(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    return (Max(0, ((A / (nbo_size)))) - block_dim);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1102::Getub_sizeCost(uint64_t* vars)
{
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    return ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1102::GetSmoothub_sizeCost(uint64_t* vars)
{
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    return ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1102::GetObj(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    double AICORE_VEC = (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ceiling((nbo_size / (nio_size))));
    OP_LOGD(OP_NAME, "AICORE_VEC = %f", AICORE_VEC);
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 2 * ceiling((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 3 * ceiling((nbo_size / (nio_size)))));
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ceiling((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ceiling((nbo_size / (nio_size)))));
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1102::GetSmoothObj(uint64_t* vars)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    double AICORE_VEC = (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ((nbo_size / (nio_size))));
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 2 * ((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 3 * ((nbo_size / (nio_size)))));
    double AIV_MTE3 = ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ((nbo_size / (nio_size)))));
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1102::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
double GeneralSolvercase1102::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + ub_size_cost;
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
double GeneralSolvercase1102::GetLeqDiff(uint64_t* vars, double* weight)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = (nbo_size - A);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = (nio_size - nbo_size);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + ub_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase1102::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == nbo_size_idx) {
        return leqs[1] <= 0 && leqs[2] <= 0 && leqs[3] <= 0;
    } else if (idx == nio_size_idx) {
        return leqs[0] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase1102::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double nbo_size = static_cast<double>(vars[nbo_size_idx]);
    double nio_size = static_cast<double>(vars[nio_size_idx]);
    if (idx == nbo_size_idx) {
        leqs[1] = (Max(0, ceiling((A / (nbo_size)))) - block_dim);
        leqs[2] = (nbo_size - A);
        leqs[3] = (nio_size - nbo_size);
    } else if (idx == nio_size_idx) {
        leqs[0] = ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
        leqs[3] = (nio_size - nbo_size);
    } else if (idx == -1) {
        leqs[0] = ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)) - ub_size);
        leqs[1] = (Max(0, ceiling((A / (nbo_size)))) - block_dim);
        leqs[2] = (nbo_size - A);
        leqs[3] = (nio_size - nbo_size);
    }
}

void GeneralSolvercase1102::DisplayVarVal(uint64_t* vars)
{
    uint64_t nbo_size = vars[nbo_size_idx];
    uint64_t nio_size = vars[nio_size_idx];
    OP_LOGD(OP_NAME, "nbo_size = %lu", static_cast<uint64_t>(nbo_size));
    OP_LOGD(OP_NAME, "nio_size = %lu", static_cast<uint64_t>(nio_size));
}

void GeneralSolvercase1102::MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data)
{
    uint64_t nbo_size = vars[nbo_size_idx];
    uint64_t nio_size = vars[nio_size_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1102 is:");
    tiling_data.set_nbo_size(static_cast<uint64_t>(nbo_size));
    OP_LOGD(OP_NAME, "nbo_size = %u", tiling_data.get_nbo_size());
    tiling_data.set_nio_size(static_cast<uint64_t>(nio_size));
    OP_LOGD(OP_NAME, "nio_size = %u", tiling_data.get_nio_size());
}

bool GeneralSolvercase1102::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase1102::GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1102.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1102 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1102Impl : public TilingCaseImpl {
 public:
  TilingCase1102Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != 5) {
      OP_LOGW(OP_NAME, "context->GetComputeNodeInfo()->GetInputsNum() != 5, current value is [%lu], invalid input num.", context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }

  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(0)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(1)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(2)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(3)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(4)->GetDataType());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }

  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(0)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(1)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(2)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(3)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(4)->GetStorageFormat());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }

  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input0 is too small.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input1 is too small.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input2 is too small.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input3 is too small.");
      return false;
    }
    if (input4_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input4 is too small.");
      return false;
    }
    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input4_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }

  bool TilingVarsShapeCheck(gert::TilingContext *context) {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 1; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool CheckInput0Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 0 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput1Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 1 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput2Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 2 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput3Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 3 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput4Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 4 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool TilingVarsCoverCheck(gert::TilingContext *context) {
    if (!CheckInput0Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 0 cannot cover all axis.");
      return false;
    }
    if (!CheckInput1Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 1 cannot cover all axis.");
      return false;
    }
    if (!CheckInput2Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 2 cannot cover all axis.");
      return false;
    }
    if (!CheckInput3Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 3 cannot cover all axis.");
      return false;
    }
    if (!CheckInput4Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 4 cannot cover all axis.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsCoverCheck success.");
    return true;
  }


  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsCoverCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }

  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    if (additional_output_ptr == nullptr) {
      OP_LOGW(OP_NAME, "Attr for additional_output_ptr is null.");
      return false;
    }
    int32_t additional_output = *additional_output_ptr;
    if ((additional_output < 1) || (additional_output > 1)) {
      OP_LOGW(OP_NAME, "(additional_output < 1) || (additional_output > 1), invalid optional att.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (TilingInputVarsCheck(context) != true) {
      return false;
    }
    if (TilingAttrCheck(context) != true) {
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1102.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1102.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(tiling_data.get_A());
    to_tiling.set_R(tiling_data.get_R());
    to_tiling.set_ub_size(tiling_data.get_ub_size());
  }
  void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) {
    tiling_data.set_A(from_tiling.get_A());
    tiling_data.set_R(from_tiling.get_R());
    tiling_data.set_BL(8);
    tiling_data.set_nbo_size(from_tiling.get_nbo_size());
    tiling_data.set_nio_size(from_tiling.get_nio_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_A_aligned_size(from_tiling.get_A_aligned_size());
    tiling_data.set_Q10(from_tiling.get_Q10());
    tiling_data.set_Q11(from_tiling.get_Q11());
    tiling_data.set_Q12(from_tiling.get_Q12());
    tiling_data.set_Q13(from_tiling.get_Q13());
    tiling_data.set_Q14(from_tiling.get_Q14());
    tiling_data.set_Q15(from_tiling.get_Q15());
    tiling_data.set_Q16(from_tiling.get_Q16());
    tiling_data.set_Q17(from_tiling.get_Q17());
    tiling_data.set_Q18(from_tiling.get_Q18());
    tiling_data.set_Q19(from_tiling.get_Q19());
    tiling_data.set_R_aligned_size(from_tiling.get_R_aligned_size());
    tiling_data.set_additional_output(from_tiling.get_additional_output());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_nbo_loop_num(from_tiling.get_nbo_loop_num());
    tiling_data.set_nbo_tail_size(from_tiling.get_nbo_tail_size());
    tiling_data.set_nbo_tail_tile_nio_loop_num(from_tiling.get_nbo_tail_tile_nio_loop_num());
    tiling_data.set_nbo_tail_tile_nio_tail_size(from_tiling.get_nbo_tail_tile_nio_tail_size());
    tiling_data.set_nio_loop_num(from_tiling.get_nio_loop_num());
    tiling_data.set_nio_tail_size(from_tiling.get_nio_tail_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    tiling_data.set_output1_total_size(from_tiling.get_output1_total_size());
    tiling_data.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    tiling_data.set_output2_total_size(from_tiling.get_output2_total_size());
    tiling_data.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    tiling_data.set_output3_total_size(from_tiling.get_output3_total_size());

    tiling_data.set_workspaceSize(from_tiling.get_workspaceSize());
  }
  bool ExecuteGeneralSolver(graph_normalTilingData& tiling_data) {
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
    uint64_t A = tiling_data.get_A();
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(nbo_size, nio_size), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(A), static_cast<uint64_t>(A)};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, false};
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
    OP_LOGD(OP_NAME, "nbo_size->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "nio_size->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    std::shared_ptr<GeneralSolvercase1102> solver = std::make_shared<GeneralSolvercase1102>(cfg, tiling_data);
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

  bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1102.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1102 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nio_size = tiling_data.get_nio_size();

    return ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double nbo_size = tiling_data.get_nbo_size();

    return Max(0, ceiling((A / (nbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 2 * ceiling((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 3 * ceiling((nbo_size / (nio_size)))));
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ceiling((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ceiling((nbo_size / (nio_size)))));
  }

  double GetAICORE_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ceiling((nbo_size / (nio_size))));
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R * nio_size) + 11.5) * 2 * ceiling((nbo_size / (nio_size)))) + (((((1.12000000476837 / ((41.4000015258789 + R))) + 0.889999985694885) * 0.019542701500562 * R) + 11.5) * 3 * ceiling((nbo_size / (nio_size)))));
    double AIV_MTE3 = ((((0.0174154702434844 * R * nio_size) + 0.219999998807907) * 2 * ceiling((nbo_size / (nio_size)))) + (((0.0346654997578466 * nio_size) + 1.03999996185303) * 2 * ceiling((nbo_size / (nio_size)))));
    double AICORE_VEC = (((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ceiling((nbo_size / (nio_size))));

    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_nio_loop_num(((tiling_data.get_nbo_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_loop_num(((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size());
    tiling_data.set_nio_tail_size((tiling_data.get_nbo_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_size() % tiling_data.get_nio_size()));
    tiling_data.set_nbo_tail_size((tiling_data.get_A() % tiling_data.get_nbo_size()) == 0 ? tiling_data.get_nbo_size() : (tiling_data.get_A() % tiling_data.get_nbo_size()));
    tiling_data.set_nbo_tail_tile_nio_loop_num(((tiling_data.get_nbo_tail_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_tail_tile_nio_tail_size((tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()));
  }

  void SetQ10(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q10((2 * R * nio_size));
  }

  void SetQ11(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q11((2 * R * nio_size));
  }

  void SetQ12(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q12((2 * R));
  }

  void SetQ13(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q13((4 * nio_size));
  }

  void SetQ14(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q14((2 * R * nio_size));
  }

  void SetQ15(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q15((4 * R * nio_size));
  }

  void SetQ16(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q16(Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  void SetQ17(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q17((4 * nio_size));
  }

  void SetQ18(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q18((2 * R));
  }

  void SetQ19(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q19((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ10(tiling_data);
    SetQ11(tiling_data);
    SetQ12(tiling_data);
    SetQ13(tiling_data);
    SetQ14(tiling_data);
    SetQ15(tiling_data);
    SetQ16(tiling_data);
    SetQ17(tiling_data);
    SetQ18(tiling_data);
    SetQ19(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1102.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1102 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1102.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1102 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1102.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1102.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set nbo_size to %u.", tiling_data.get_nbo_size());
    OP_LOGI(OP_NAME, "Set nio_size to %u.", tiling_data.get_nio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AICORE_VEC is %f.", GetAICORE_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

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
    cons_info_->leqs[0] = ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size)
    cons_info_->leqs[1] = (Max(0, ceiling((A / (sbo_size)))) - block_dim)
    cons_info_->leqs[2] = (sbo_size - A)
    cons_info_->leqs[3] = ((16 * sio_size_div_align) - R)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1111 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1111(SolverConfig& config, graph_normalTilingData& tiling_data) : GeneralSolver(config) {
            A = tiling_data.get_A();
            R = tiling_data.get_R();
            ub_size = tiling_data.get_ub_size();
            R = ((R + 16 - 1) / 16) * 16;
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
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t sbo_size_idx = 0;
        const int64_t sio_size_div_align_idx = 1;
        uint64_t A;
        uint64_t R;
        uint64_t ub_size;
        uint64_t block_dim{0};
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1111::Getblock_dimCost(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    return (Max(0, ceiling((A / (sbo_size)))) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1111::GetSmoothblock_dimCost(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    return (Max(0, ((A / (sbo_size)))) - block_dim);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1111::Getub_sizeCost(uint64_t* vars)
{
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    return ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1111::GetSmoothub_sizeCost(uint64_t* vars)
{
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    return ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1111::GetObj(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    double AICORE_VEC = (((128 * sio_size_div_align / (((16 * sio_size_div_align) + -1))) + 4) * 3 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    OP_LOGD(OP_NAME, "AICORE_VEC = %f", AICORE_VEC);
    double AIV_MTE2 = (((((1.12000000476837 / (((16 * sio_size_div_align) + 41.4000015258789))) + 0.889999985694885) * 0.312683224008991 * sio_size_div_align) + 11.5) * 5 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = ((((0.27864752389575 * sio_size_div_align) + 0.219999998807907) * 2 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size) + (2.14933092322175 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size));
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1111::GetSmoothObj(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    double AICORE_VEC = (((128 * sio_size_div_align / (((16 * sio_size_div_align) + -1))) + 4) * 3 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    double AIV_MTE2 = (((((1.12000000476837 / (((16 * sio_size_div_align) + 41.4000015258789))) + 0.889999985694885) * 0.312683224008991 * sio_size_div_align) + 11.5) * 5 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    double AIV_MTE3 = ((((0.27864752389575 * sio_size_div_align) + 0.219999998807907) * 2 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size) + (2.14933092322175 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size));
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1111::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
double GeneralSolvercase1111::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + ub_size_cost;
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
double GeneralSolvercase1111::GetLeqDiff(uint64_t* vars, double* weight)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = (sbo_size - A);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = ((16 * sio_size_div_align) - R);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + ub_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase1111::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == sbo_size_idx) {
        return leqs[1] <= 0 && leqs[2] <= 0;
    } else if (idx == sio_size_div_align_idx) {
        return leqs[0] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase1111::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    if (idx == sbo_size_idx) {
        leqs[1] = (Max(0, ceiling((A / (sbo_size)))) - block_dim);
        leqs[2] = (sbo_size - A);
    } else if (idx == sio_size_div_align_idx) {
        leqs[0] = ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
        leqs[3] = ((16 * sio_size_div_align) - R);
    } else if (idx == -1) {
        leqs[0] = ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
        leqs[1] = (Max(0, ceiling((A / (sbo_size)))) - block_dim);
        leqs[2] = (sbo_size - A);
        leqs[3] = ((16 * sio_size_div_align) - R);
    }
}

void GeneralSolvercase1111::DisplayVarVal(uint64_t* vars)
{
    uint64_t sbo_size = vars[sbo_size_idx];
    uint64_t sio_size_div_align = vars[sio_size_div_align_idx];
    OP_LOGD(OP_NAME, "sbo_size = %lu", static_cast<uint64_t>(sbo_size));
    OP_LOGD(OP_NAME, "sio_size = %lu", static_cast<uint64_t>((16 * sio_size_div_align)));
}

void GeneralSolvercase1111::MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data)
{
    uint64_t sbo_size = vars[sbo_size_idx];
    uint64_t sio_size_div_align = vars[sio_size_div_align_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1111 is:");
    tiling_data.set_sbo_size(static_cast<uint64_t>(sbo_size));
    OP_LOGD(OP_NAME, "sbo_size = %u", tiling_data.get_sbo_size());
    tiling_data.set_sio_size(static_cast<uint64_t>((16 * sio_size_div_align)));
    OP_LOGD(OP_NAME, "sio_size = %u", tiling_data.get_sio_size());
}

bool GeneralSolvercase1111::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase1111::GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1111.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1111 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1111Impl : public TilingCaseImpl {
 public:
  TilingCase1111Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != 5) {
      OP_LOGW(OP_NAME, "context->GetComputeNodeInfo()->GetInputsNum() != 5, current value is [%lu], invalid input num.", context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }

  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(0)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(1)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(2)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(3)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(4)->GetDataType());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }

  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(0)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(1)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(2)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(3)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(4)->GetStorageFormat());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }

  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input0 is too small.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input1 is too small.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input2 is too small.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input3 is too small.");
      return false;
    }
    if (input4_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input4 is too small.");
      return false;
    }
    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input4_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }

  bool TilingVarsShapeCheck(gert::TilingContext *context) {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 2; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input4.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = context->GetInputShape(4)->GetStorageShape().GetDim(input4_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool CheckInput0Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 0 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput1Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 1 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput2Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 2 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput3Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 3 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput4Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 4 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool TilingVarsCoverCheck(gert::TilingContext *context) {
    if (!CheckInput0Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 0 cannot cover all axis.");
      return false;
    }
    if (!CheckInput1Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 1 cannot cover all axis.");
      return false;
    }
    if (!CheckInput2Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 2 cannot cover all axis.");
      return false;
    }
    if (!CheckInput3Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 3 cannot cover all axis.");
      return false;
    }
    if (!CheckInput4Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 4 cannot cover all axis.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsCoverCheck success.");
    return true;
  }


  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsCoverCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }

  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    if (additional_output_ptr == nullptr) {
      OP_LOGW(OP_NAME, "Attr for additional_output_ptr is null.");
      return false;
    }
    int32_t additional_output = *additional_output_ptr;
    if ((additional_output < 1) || (additional_output > 1)) {
      OP_LOGW(OP_NAME, "(additional_output < 1) || (additional_output > 1), invalid optional att.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (TilingInputVarsCheck(context) != true) {
      return false;
    }
    if (TilingAttrCheck(context) != true) {
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1111.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1111.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(tiling_data.get_A());
    to_tiling.set_R(tiling_data.get_R());
    to_tiling.set_ub_size(tiling_data.get_ub_size());
  }
  void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) {
    tiling_data.set_A(from_tiling.get_A());
    tiling_data.set_R(from_tiling.get_R());
    tiling_data.set_sbo_size(from_tiling.get_sbo_size());
    tiling_data.set_sio_size(from_tiling.get_sio_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_A_aligned_size(from_tiling.get_A_aligned_size());
    tiling_data.set_Q20(from_tiling.get_Q20());
    tiling_data.set_Q21(from_tiling.get_Q21());
    tiling_data.set_Q22(from_tiling.get_Q22());
    tiling_data.set_Q23(from_tiling.get_Q23());
    tiling_data.set_Q24(from_tiling.get_Q24());
    tiling_data.set_Q25(from_tiling.get_Q25());
    tiling_data.set_Q26(from_tiling.get_Q26());
    tiling_data.set_Q27(from_tiling.get_Q27());
    tiling_data.set_Q28(from_tiling.get_Q28());
    tiling_data.set_R_aligned_size(from_tiling.get_R_aligned_size());
    tiling_data.set_additional_output(from_tiling.get_additional_output());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    tiling_data.set_output1_total_size(from_tiling.get_output1_total_size());
    tiling_data.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    tiling_data.set_output2_total_size(from_tiling.get_output2_total_size());
    tiling_data.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    tiling_data.set_output3_total_size(from_tiling.get_output3_total_size());
    tiling_data.set_sbo_loop_num(from_tiling.get_sbo_loop_num());
    tiling_data.set_sbo_tail_size(from_tiling.get_sbo_tail_size());
    tiling_data.set_sio_loop_num(from_tiling.get_sio_loop_num());
    tiling_data.set_sio_tail_size(from_tiling.get_sio_tail_size());

    tiling_data.set_workspaceSize(from_tiling.get_workspaceSize());
  }
  bool ExecuteGeneralSolver(graph_normalTilingData& tiling_data) {
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
    uint64_t A = tiling_data.get_A();
    uint64_t R = tiling_data.get_R();
    R = ((R + 16 - 1) / 16) * 16;
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(sbo_size, sio_size_div_align), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(A), static_cast<uint64_t>(((double)(1)/(double)(16) * R))};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, true};
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
    OP_LOGD(OP_NAME, "sbo_size->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "sio_size_div_align->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    std::shared_ptr<GeneralSolvercase1111> solver = std::make_shared<GeneralSolvercase1111>(cfg, tiling_data);
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

  bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1111.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1111 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sio_size = tiling_data.get_sio_size();

    return ((4 * R) + (8 * sio_size) + 8 + Max((2 * R), (4 * R)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double sbo_size = tiling_data.get_sbo_size();

    return Max(0, ceiling((A / (sbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return (((((1.12000000476837 / ((41.4000015258789 + sio_size))) + 0.889999985694885) * 0.019542701500562 * sio_size) + 11.5) * 5 * ceiling((R / (sio_size))) * sbo_size);
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return ((((0.0174154702434844 * sio_size) + 0.219999998807907) * 2 * ceiling((R / (sio_size))) * sbo_size) + (2.14933092322175 * ceiling((R / (sio_size))) * sbo_size));
  }

  double GetAICORE_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return (((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * ceiling((R / (sio_size))) * sbo_size);
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    double AIV_MTE2 = (((((1.12000000476837 / ((41.4000015258789 + sio_size))) + 0.889999985694885) * 0.019542701500562 * sio_size) + 11.5) * 5 * ceiling((R / (sio_size))) * sbo_size);
    double AIV_MTE3 = ((((0.0174154702434844 * sio_size) + 0.219999998807907) * 2 * ceiling((R / (sio_size))) * sbo_size) + (2.14933092322175 * ceiling((R / (sio_size))) * sbo_size));
    double AICORE_VEC = (((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * ceiling((R / (sio_size))) * sbo_size);

    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_sbo_loop_num(((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size());
    tiling_data.set_sio_loop_num(((tiling_data.get_R() + tiling_data.get_sio_size()) - 1) / tiling_data.get_sio_size());
    tiling_data.set_sbo_tail_size((tiling_data.get_A() % tiling_data.get_sbo_size()) == 0 ? tiling_data.get_sbo_size() : (tiling_data.get_A() % tiling_data.get_sbo_size()));
    tiling_data.set_sio_tail_size((tiling_data.get_R() % tiling_data.get_sio_size()) == 0 ? tiling_data.get_sio_size() : (tiling_data.get_R() % tiling_data.get_sio_size()));
  }

  void SetQ20(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q20((2 * sio_size));
  }

  void SetQ21(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q21((2 * sio_size));
  }

  void SetQ22(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q22((2 * sio_size));
  }

  void SetQ23(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q23(4);
  }

  void SetQ24(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q24((2 * sio_size));
  }

  void SetQ25(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q25(Max((2 * R), (4 * R)));
  }

  void SetQ26(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q26(4);
  }

  void SetQ27(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q27((2 * R));
  }

  void SetQ28(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q28((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ20(tiling_data);
    SetQ21(tiling_data);
    SetQ22(tiling_data);
    SetQ23(tiling_data);
    SetQ24(tiling_data);
    SetQ25(tiling_data);
    SetQ26(tiling_data);
    SetQ27(tiling_data);
    SetQ28(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1111.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1111 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1111.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1111 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1111.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1111.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set sbo_size to %u.", tiling_data.get_sbo_size());
    OP_LOGI(OP_NAME, "Set sio_size to %u.", tiling_data.get_sio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AICORE_VEC is %f.", GetAICORE_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

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
    cons_info_->leqs[0] = ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size)
    cons_info_->leqs[1] = (Max(0, ceiling((A / (sbo_size)))) - block_dim)
    cons_info_->leqs[2] = (sbo_size - A)
    cons_info_->leqs[3] = ((16 * sio_size_div_align) - R)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1112 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1112(SolverConfig& config, graph_normalTilingData& tiling_data) : GeneralSolver(config) {
            A = tiling_data.get_A();
            R = tiling_data.get_R();
            ub_size = tiling_data.get_ub_size();
            R = ((R + 16 - 1) / 16) * 16;
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
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t sbo_size_idx = 0;
        const int64_t sio_size_div_align_idx = 1;
        uint64_t A;
        uint64_t R;
        uint64_t ub_size;
        uint64_t block_dim{0};
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1112::Getblock_dimCost(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    return (Max(0, ceiling((A / (sbo_size)))) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1112::GetSmoothblock_dimCost(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    return (Max(0, ((A / (sbo_size)))) - block_dim);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1112::Getub_sizeCost(uint64_t* vars)
{
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    return ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1112::GetSmoothub_sizeCost(uint64_t* vars)
{
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    return ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1112::GetObj(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    double AICORE_VEC = (((128 * sio_size_div_align / (((16 * sio_size_div_align) + -1))) + 4) * 3 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    OP_LOGD(OP_NAME, "AICORE_VEC = %f", AICORE_VEC);
    double AIV_MTE2 = (((((1.12000000476837 / (((16 * sio_size_div_align) + 41.4000015258789))) + 0.889999985694885) * 0.312683224008991 * sio_size_div_align) + 11.5) * 5 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = ((((0.27864752389575 * sio_size_div_align) + 0.219999998807907) * 2 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size) + (2.14933092322175 * ceiling(((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size));
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1112::GetSmoothObj(uint64_t* vars)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    double AICORE_VEC = (((128 * sio_size_div_align / (((16 * sio_size_div_align) + -1))) + 4) * 3 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    double AIV_MTE2 = (((((1.12000000476837 / (((16 * sio_size_div_align) + 41.4000015258789))) + 0.889999985694885) * 0.312683224008991 * sio_size_div_align) + 11.5) * 5 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size);
    double AIV_MTE3 = ((((0.27864752389575 * sio_size_div_align) + 0.219999998807907) * 2 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size) + (2.14933092322175 * (((double)(1)/(double)(16) * R / (sio_size_div_align))) * sbo_size));
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1112::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
double GeneralSolvercase1112::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + ub_size_cost;
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
double GeneralSolvercase1112::GetLeqDiff(uint64_t* vars, double* weight)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = (sbo_size - A);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = ((16 * sio_size_div_align) - R);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + ub_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase1112::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == sbo_size_idx) {
        return leqs[1] <= 0 && leqs[2] <= 0;
    } else if (idx == sio_size_div_align_idx) {
        return leqs[0] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase1112::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double sbo_size = static_cast<double>(vars[sbo_size_idx]);
    double sio_size_div_align = static_cast<double>(vars[sio_size_div_align_idx]);
    if (idx == sbo_size_idx) {
        leqs[1] = (Max(0, ceiling((A / (sbo_size)))) - block_dim);
        leqs[2] = (sbo_size - A);
    } else if (idx == sio_size_div_align_idx) {
        leqs[0] = ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
        leqs[3] = ((16 * sio_size_div_align) - R);
    } else if (idx == -1) {
        leqs[0] = ((128 * sio_size_div_align) + (4 * R) + 8 + Max((2 * R), (4 * R)) - ub_size);
        leqs[1] = (Max(0, ceiling((A / (sbo_size)))) - block_dim);
        leqs[2] = (sbo_size - A);
        leqs[3] = ((16 * sio_size_div_align) - R);
    }
}

void GeneralSolvercase1112::DisplayVarVal(uint64_t* vars)
{
    uint64_t sbo_size = vars[sbo_size_idx];
    uint64_t sio_size_div_align = vars[sio_size_div_align_idx];
    OP_LOGD(OP_NAME, "sio_size = %lu", static_cast<uint64_t>((16 * sio_size_div_align)));
    OP_LOGD(OP_NAME, "sbo_size = %lu", static_cast<uint64_t>(sbo_size));
}

void GeneralSolvercase1112::MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data)
{
    uint64_t sbo_size = vars[sbo_size_idx];
    uint64_t sio_size_div_align = vars[sio_size_div_align_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1112 is:");
    tiling_data.set_sio_size(static_cast<uint64_t>((16 * sio_size_div_align)));
    OP_LOGD(OP_NAME, "sio_size = %u", tiling_data.get_sio_size());
    tiling_data.set_sbo_size(static_cast<uint64_t>(sbo_size));
    OP_LOGD(OP_NAME, "sbo_size = %u", tiling_data.get_sbo_size());
}

bool GeneralSolvercase1112::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase1112::GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1112.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1112 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1112Impl : public TilingCaseImpl {
 public:
  TilingCase1112Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != 5) {
      OP_LOGW(OP_NAME, "context->GetComputeNodeInfo()->GetInputsNum() != 5, current value is [%lu], invalid input num.", context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }

  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(0)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(1)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(2)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(3)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(4)->GetDataType());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }

  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(0)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(1)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(2)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(3)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(4)->GetStorageFormat());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }

  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input0 is too small.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input1 is too small.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input2 is too small.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input3 is too small.");
      return false;
    }
    if (input4_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input4 is too small.");
      return false;
    }
    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input4_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }

  bool TilingVarsShapeCheck(gert::TilingContext *context) {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 1; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool CheckInput0Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 0 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput1Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 1 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput2Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 2 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput3Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 3 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput4Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 4 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool TilingVarsCoverCheck(gert::TilingContext *context) {
    if (!CheckInput0Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 0 cannot cover all axis.");
      return false;
    }
    if (!CheckInput1Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 1 cannot cover all axis.");
      return false;
    }
    if (!CheckInput2Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 2 cannot cover all axis.");
      return false;
    }
    if (!CheckInput3Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 3 cannot cover all axis.");
      return false;
    }
    if (!CheckInput4Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 4 cannot cover all axis.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsCoverCheck success.");
    return true;
  }


  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsCoverCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }

  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    if (additional_output_ptr == nullptr) {
      OP_LOGW(OP_NAME, "Attr for additional_output_ptr is null.");
      return false;
    }
    int32_t additional_output = *additional_output_ptr;
    if ((additional_output < 1) || (additional_output > 1)) {
      OP_LOGW(OP_NAME, "(additional_output < 1) || (additional_output > 1), invalid optional att.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (TilingInputVarsCheck(context) != true) {
      return false;
    }
    if (TilingAttrCheck(context) != true) {
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1112.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1112.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(tiling_data.get_A());
    to_tiling.set_R(tiling_data.get_R());
    to_tiling.set_ub_size(tiling_data.get_ub_size());
  }
  void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) {
    tiling_data.set_A(from_tiling.get_A());
    tiling_data.set_R(from_tiling.get_R());
    tiling_data.set_sbo_size(from_tiling.get_sbo_size());
    tiling_data.set_sio_size(from_tiling.get_sio_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_A_aligned_size(from_tiling.get_A_aligned_size());
    tiling_data.set_Q29(from_tiling.get_Q29());
    tiling_data.set_Q30(from_tiling.get_Q30());
    tiling_data.set_Q31(from_tiling.get_Q31());
    tiling_data.set_Q32(from_tiling.get_Q32());
    tiling_data.set_Q33(from_tiling.get_Q33());
    tiling_data.set_Q34(from_tiling.get_Q34());
    tiling_data.set_Q35(from_tiling.get_Q35());
    tiling_data.set_Q36(from_tiling.get_Q36());
    tiling_data.set_Q37(from_tiling.get_Q37());
    tiling_data.set_R_aligned_size(from_tiling.get_R_aligned_size());
    tiling_data.set_additional_output(from_tiling.get_additional_output());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    tiling_data.set_output1_total_size(from_tiling.get_output1_total_size());
    tiling_data.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    tiling_data.set_output2_total_size(from_tiling.get_output2_total_size());
    tiling_data.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    tiling_data.set_output3_total_size(from_tiling.get_output3_total_size());
    tiling_data.set_sbo_loop_num(from_tiling.get_sbo_loop_num());
    tiling_data.set_sbo_tail_size(from_tiling.get_sbo_tail_size());
    tiling_data.set_sio_loop_num(from_tiling.get_sio_loop_num());
    tiling_data.set_sio_tail_size(from_tiling.get_sio_tail_size());

    tiling_data.set_workspaceSize(from_tiling.get_workspaceSize());
  }
  bool ExecuteGeneralSolver(graph_normalTilingData& tiling_data) {
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
    uint64_t A = tiling_data.get_A();
    uint64_t R = tiling_data.get_R();
    R = ((R + 16 - 1) / 16) * 16;
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(sbo_size, sio_size_div_align), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(A), static_cast<uint64_t>(((double)(1)/(double)(16) * R))};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, true};
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
    OP_LOGD(OP_NAME, "sbo_size->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "sio_size_div_align->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    std::shared_ptr<GeneralSolvercase1112> solver = std::make_shared<GeneralSolvercase1112>(cfg, tiling_data);
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

  bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1112.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1112 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sio_size = tiling_data.get_sio_size();

    return ((4 * R) + (8 * sio_size) + 8 + Max((2 * R), (4 * R)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double sbo_size = tiling_data.get_sbo_size();

    return Max(0, ceiling((A / (sbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return (((((1.12000000476837 / ((41.4000015258789 + sio_size))) + 0.889999985694885) * 0.019542701500562 * sio_size) + 11.5) * 5 * ceiling((R / (sio_size))) * sbo_size);
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return ((((0.0174154702434844 * sio_size) + 0.219999998807907) * 2 * ceiling((R / (sio_size))) * sbo_size) + (2.14933092322175 * ceiling((R / (sio_size))) * sbo_size));
  }

  double GetAICORE_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return (((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * ceiling((R / (sio_size))) * sbo_size);
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    double AIV_MTE2 = (((((1.12000000476837 / ((41.4000015258789 + sio_size))) + 0.889999985694885) * 0.019542701500562 * sio_size) + 11.5) * 5 * ceiling((R / (sio_size))) * sbo_size);
    double AIV_MTE3 = ((((0.0174154702434844 * sio_size) + 0.219999998807907) * 2 * ceiling((R / (sio_size))) * sbo_size) + (2.14933092322175 * ceiling((R / (sio_size))) * sbo_size));
    double AICORE_VEC = (((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * ceiling((R / (sio_size))) * sbo_size);

    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_sbo_loop_num(((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size());
    tiling_data.set_sio_loop_num(((tiling_data.get_R() + tiling_data.get_sio_size()) - 1) / tiling_data.get_sio_size());
    tiling_data.set_sbo_tail_size((tiling_data.get_A() % tiling_data.get_sbo_size()) == 0 ? tiling_data.get_sbo_size() : (tiling_data.get_A() % tiling_data.get_sbo_size()));
    tiling_data.set_sio_tail_size((tiling_data.get_R() % tiling_data.get_sio_size()) == 0 ? tiling_data.get_sio_size() : (tiling_data.get_R() % tiling_data.get_sio_size()));
  }

  void SetQ29(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q29((2 * sio_size));
  }

  void SetQ30(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q30((2 * sio_size));
  }

  void SetQ31(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q31((2 * sio_size));
  }

  void SetQ32(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q32(4);
  }

  void SetQ33(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q33((2 * sio_size));
  }

  void SetQ34(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q34(Max((2 * R), (4 * R)));
  }

  void SetQ35(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q35(4);
  }

  void SetQ36(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q36((2 * R));
  }

  void SetQ37(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q37((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ29(tiling_data);
    SetQ30(tiling_data);
    SetQ31(tiling_data);
    SetQ32(tiling_data);
    SetQ33(tiling_data);
    SetQ34(tiling_data);
    SetQ35(tiling_data);
    SetQ36(tiling_data);
    SetQ37(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1112.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1112 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1112.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1112 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1112.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1112.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set sbo_size to %u.", tiling_data.get_sbo_size());
    OP_LOGI(OP_NAME, "Set sio_size to %u.", tiling_data.get_sio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AICORE_VEC is %f.", GetAICORE_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

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
    cons_info_->leqs[0] = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size)
    cons_info_->leqs[1] = (Max(0, ceiling((A / (wbo_size)))) - block_dim)
    cons_info_->leqs[2] = (wbo_size - A)
    cons_info_->leqs[3] = (wio_size - R)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1151 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1151(SolverConfig& config, graph_normalTilingData& tiling_data) : GeneralSolver(config) {
            A = tiling_data.get_A();
            R = tiling_data.get_R();
            ub_size = tiling_data.get_ub_size();
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
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t wbo_size_idx = 0;
        const int64_t wio_size_idx = 1;
        uint64_t A;
        uint64_t R;
        uint64_t ub_size;
        uint64_t block_dim{0};
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1151::Getblock_dimCost(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    return (Max(0, ceiling((A / (wbo_size)))) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1151::GetSmoothblock_dimCost(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    return (Max(0, ((A / (wbo_size)))) - block_dim);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1151::Getub_sizeCost(uint64_t* vars)
{
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1151::GetSmoothub_sizeCost(uint64_t* vars)
{
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1151::GetObj(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    double AICORE_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ceiling((R / (wio_size))) * wbo_size));
    OP_LOGD(OP_NAME, "AICORE_VEC = %f", AICORE_VEC);
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ceiling((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ceiling((R / (wio_size))) * wbo_size));
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ceiling((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ceiling((R / (wio_size))) * wbo_size) + (2.14933092322175 * ceiling((R / (wio_size))) * wbo_size));
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1151::GetSmoothObj(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    double AICORE_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ((R / (wio_size))) * wbo_size));
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ((R / (wio_size))) * wbo_size));
    double AIV_MTE3 = ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ((R / (wio_size))) * wbo_size) + (2.14933092322175 * ((R / (wio_size))) * wbo_size));
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1151::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
double GeneralSolvercase1151::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + ub_size_cost;
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
double GeneralSolvercase1151::GetLeqDiff(uint64_t* vars, double* weight)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = (wbo_size - A);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = (wio_size - R);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + ub_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase1151::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == wbo_size_idx) {
        return leqs[1] <= 0 && leqs[2] <= 0;
    } else if (idx == wio_size_idx) {
        return leqs[0] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase1151::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    if (idx == wbo_size_idx) {
        leqs[1] = (Max(0, ceiling((A / (wbo_size)))) - block_dim);
        leqs[2] = (wbo_size - A);
    } else if (idx == wio_size_idx) {
        leqs[0] = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
        leqs[3] = (wio_size - R);
    } else if (idx == -1) {
        leqs[0] = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
        leqs[1] = (Max(0, ceiling((A / (wbo_size)))) - block_dim);
        leqs[2] = (wbo_size - A);
        leqs[3] = (wio_size - R);
    }
}

void GeneralSolvercase1151::DisplayVarVal(uint64_t* vars)
{
    uint64_t wbo_size = vars[wbo_size_idx];
    uint64_t wio_size = vars[wio_size_idx];
    OP_LOGD(OP_NAME, "wbo_size = %lu", static_cast<uint64_t>(wbo_size));
    OP_LOGD(OP_NAME, "wio_size = %lu", static_cast<uint64_t>(wio_size));
}

void GeneralSolvercase1151::MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data)
{
    uint64_t wbo_size = vars[wbo_size_idx];
    uint64_t wio_size = vars[wio_size_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1151 is:");
    tiling_data.set_wbo_size(static_cast<uint64_t>(wbo_size));
    OP_LOGD(OP_NAME, "wbo_size = %u", tiling_data.get_wbo_size());
    tiling_data.set_wio_size(static_cast<uint64_t>(wio_size));
    OP_LOGD(OP_NAME, "wio_size = %u", tiling_data.get_wio_size());
}

bool GeneralSolvercase1151::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase1151::GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1151.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1151 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1151Impl : public TilingCaseImpl {
 public:
  TilingCase1151Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != 5) {
      OP_LOGW(OP_NAME, "context->GetComputeNodeInfo()->GetInputsNum() != 5, current value is [%lu], invalid input num.", context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }

  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(0)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(1)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(2)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(3)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(4)->GetDataType());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }

  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(0)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(1)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(2)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(3)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(4)->GetStorageFormat());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }

  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input0 is too small.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input1 is too small.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input2 is too small.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input3 is too small.");
      return false;
    }
    if (input4_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input4 is too small.");
      return false;
    }
    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input4_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }

  bool TilingVarsShapeCheck(gert::TilingContext *context) {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 2; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input4.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = context->GetInputShape(4)->GetStorageShape().GetDim(input4_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool CheckInput0Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 0 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput1Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 1 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput2Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 2 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput3Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 3 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput4Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 4 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool TilingVarsCoverCheck(gert::TilingContext *context) {
    if (!CheckInput0Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 0 cannot cover all axis.");
      return false;
    }
    if (!CheckInput1Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 1 cannot cover all axis.");
      return false;
    }
    if (!CheckInput2Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 2 cannot cover all axis.");
      return false;
    }
    if (!CheckInput3Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 3 cannot cover all axis.");
      return false;
    }
    if (!CheckInput4Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 4 cannot cover all axis.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsCoverCheck success.");
    return true;
  }


  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsCoverCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }

  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    if (additional_output_ptr == nullptr) {
      OP_LOGW(OP_NAME, "Attr for additional_output_ptr is null.");
      return false;
    }
    int32_t additional_output = *additional_output_ptr;
    if ((additional_output < 1) || (additional_output > 1)) {
      OP_LOGW(OP_NAME, "(additional_output < 1) || (additional_output > 1), invalid optional att.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (TilingInputVarsCheck(context) != true) {
      return false;
    }
    if (TilingAttrCheck(context) != true) {
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1151.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1151.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(tiling_data.get_A());
    to_tiling.set_R(tiling_data.get_R());
    to_tiling.set_ub_size(tiling_data.get_ub_size());
  }
  void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) {
    tiling_data.set_A(from_tiling.get_A());
    tiling_data.set_R(from_tiling.get_R());
    tiling_data.set_wbo_size(from_tiling.get_wbo_size());
    tiling_data.set_wio_size(from_tiling.get_wio_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_A_aligned_size(from_tiling.get_A_aligned_size());
    tiling_data.set_Q38(from_tiling.get_Q38());
    tiling_data.set_Q39(from_tiling.get_Q39());
    tiling_data.set_Q40(from_tiling.get_Q40());
    tiling_data.set_Q41(from_tiling.get_Q41());
    tiling_data.set_Q42(from_tiling.get_Q42());
    tiling_data.set_Q43(from_tiling.get_Q43());
    tiling_data.set_Q44(from_tiling.get_Q44());
    tiling_data.set_Q45(from_tiling.get_Q45());
    tiling_data.set_Q46(from_tiling.get_Q46());
    tiling_data.set_R_aligned_size(from_tiling.get_R_aligned_size());
    tiling_data.set_additional_output(from_tiling.get_additional_output());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    tiling_data.set_output1_total_size(from_tiling.get_output1_total_size());
    tiling_data.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    tiling_data.set_output2_total_size(from_tiling.get_output2_total_size());
    tiling_data.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    tiling_data.set_output3_total_size(from_tiling.get_output3_total_size());
    tiling_data.set_wbo_loop_num(from_tiling.get_wbo_loop_num());
    tiling_data.set_wbo_tail_size(from_tiling.get_wbo_tail_size());
    tiling_data.set_wio_loop_num(from_tiling.get_wio_loop_num());
    tiling_data.set_wio_tail_size(from_tiling.get_wio_tail_size());

    tiling_data.set_workspaceSize(from_tiling.get_workspaceSize());
  }
  bool ExecuteGeneralSolver(graph_normalTilingData& tiling_data) {
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
    uint64_t A = tiling_data.get_A();
    uint64_t R = tiling_data.get_R();
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(wbo_size, wio_size), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(A), static_cast<uint64_t>(R)};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, true};
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
    OP_LOGD(OP_NAME, "wbo_size->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "wio_size->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    std::shared_ptr<GeneralSolvercase1151> solver = std::make_shared<GeneralSolvercase1151>(cfg, tiling_data);
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

  bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1151.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1151 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double wio_size = tiling_data.get_wio_size();

    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double wbo_size = tiling_data.get_wbo_size();

    return Max(0, ceiling((A / (wbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ceiling((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ceiling((R / (wio_size))) * wbo_size));
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ceiling((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ceiling((R / (wio_size))) * wbo_size) + (2.14933092322175 * ceiling((R / (wio_size))) * wbo_size));
  }

  double GetAICORE_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((4 * wio_size / ((-1 + wio_size))) + 4) * ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ceiling((R / (wio_size))) * wbo_size));
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ceiling((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ceiling((R / (wio_size))) * wbo_size));
    double AIV_MTE3 = ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ceiling((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ceiling((R / (wio_size))) * wbo_size) + (2.14933092322175 * ceiling((R / (wio_size))) * wbo_size));
    double AICORE_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ceiling((R / (wio_size))) * wbo_size));

    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_wbo_loop_num(((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size());
    tiling_data.set_wio_loop_num(((tiling_data.get_R() + tiling_data.get_wio_size()) - 1) / tiling_data.get_wio_size());
    tiling_data.set_wbo_tail_size((tiling_data.get_A() % tiling_data.get_wbo_size()) == 0 ? tiling_data.get_wbo_size() : (tiling_data.get_A() % tiling_data.get_wbo_size()));
    tiling_data.set_wio_tail_size((tiling_data.get_R() % tiling_data.get_wio_size()) == 0 ? tiling_data.get_wio_size() : (tiling_data.get_R() % tiling_data.get_wio_size()));
  }

  void SetQ38(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q38((2 * wio_size));
  }

  void SetQ39(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q39((2 * wio_size));
  }

  void SetQ40(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q40((2 * wio_size));
  }

  void SetQ41(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q41((2 * wio_size));
  }

  void SetQ42(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q42((4 * wio_size));
  }

  void SetQ43(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q43(Max((2 * wio_size), (4 * wio_size)));
  }

  void SetQ44(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q44((4 * wio_size));
  }

  void SetQ45(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q45(4);
  }

  void SetQ46(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q46(4);
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ38(tiling_data);
    SetQ39(tiling_data);
    SetQ40(tiling_data);
    SetQ41(tiling_data);
    SetQ42(tiling_data);
    SetQ43(tiling_data);
    SetQ44(tiling_data);
    SetQ45(tiling_data);
    SetQ46(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1151.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1151 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1151.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1151 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1151.");
    double A = static_cast<double>(tiling_data.get_A());
    double R = static_cast<double>(tiling_data.get_R());
    double wbo_size = static_cast<double>(tiling_data.get_wbo_size());
    tiling_data.set_workspaceSize(static_cast<uint32_t>((4 * Max(0, ceiling((A / (wbo_size)))) * R * wbo_size)));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1151.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set wbo_size to %u.", tiling_data.get_wbo_size());
    OP_LOGI(OP_NAME, "Set wio_size to %u.", tiling_data.get_wio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AICORE_VEC is %f.", GetAICORE_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

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
    cons_info_->leqs[0] = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size)
    cons_info_->leqs[1] = (Max(0, ceiling((A / (wbo_size)))) - block_dim)
    cons_info_->leqs[2] = (wbo_size - A)
    cons_info_->leqs[3] = (wio_size - R)
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase1152 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase1152(SolverConfig& config, graph_normalTilingData& tiling_data) : GeneralSolver(config) {
            A = tiling_data.get_A();
            R = tiling_data.get_R();
            ub_size = tiling_data.get_ub_size();
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
        double Getub_sizeCost(uint64_t* vars);
        double GetSmoothub_sizeCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data);
        bool Init(const SolverInput &input);
    private:
        const int64_t wbo_size_idx = 0;
        const int64_t wio_size_idx = 1;
        uint64_t A;
        uint64_t R;
        uint64_t ub_size;
        uint64_t block_dim{0};
};
/*
函数名:Getblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1152::Getblock_dimCost(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    return (Max(0, ceiling((A / (wbo_size)))) - block_dim);
}

/*
函数名:GetSmoothblock_dimCost(重要函数)
功能描述:
  根据待求解变量值block_dim的平滑化缓存占用信息
  与Getblock_dimCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1152::GetSmoothblock_dimCost(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    return (Max(0, ((A / (wbo_size)))) - block_dim);
}

/*
函数名:Getub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1152::Getub_sizeCost(uint64_t* vars)
{
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
}

/*
函数名:GetSmoothub_sizeCost(重要函数)
功能描述:
  根据待求解变量值ub_size的平滑化缓存占用信息
  与Getub_sizeCost函数相比,整除运算被替换为浮点数的除法运算
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1152::GetSmoothub_sizeCost(uint64_t* vars)
{
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1152::GetObj(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    double AICORE_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ceiling((R / (wio_size))) * wbo_size));
    OP_LOGD(OP_NAME, "AICORE_VEC = %f", AICORE_VEC);
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ceiling((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ceiling((R / (wio_size))) * wbo_size));
    OP_LOGD(OP_NAME, "AIV_MTE2 = %f", AIV_MTE2);
    double AIV_MTE3 = ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ceiling((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ceiling((R / (wio_size))) * wbo_size) + (2.14933092322175 * ceiling((R / (wio_size))) * wbo_size));
    OP_LOGD(OP_NAME, "AIV_MTE3 = %f", AIV_MTE3);
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetSmoothObj(重要函数)
功能描述:
  根据待求解变量值输出平滑化目标函数
  与GetObj函数相比,整除运算被替换为浮点数的除法运算
*/
double GeneralSolvercase1152::GetSmoothObj(uint64_t* vars)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    double AICORE_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ((R / (wio_size))) * wbo_size));
    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ((R / (wio_size))) * wbo_size));
    double AIV_MTE3 = ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ((R / (wio_size))) * wbo_size) + (2.14933092322175 * ((R / (wio_size))) * wbo_size));
    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
  该函数用于量化解在缓存占用方面的质量
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase1152::GetBuffCost(uint64_t* vars)
{
    double block_dim_cost = Getblock_dimCost(vars);
    double ub_size_cost = Getub_sizeCost(vars);
    return ((Min(0, block_dim_cost) * Min(0, block_dim_cost)) + (Min(0, ub_size_cost) * Min(0, ub_size_cost)));
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
double GeneralSolvercase1152::GetBuffDiff(uint64_t* vars, double* weight)
{
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] < 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] < 0 ? weight[0] : 0;
    return block_dim_cost + ub_size_cost;
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
double GeneralSolvercase1152::GetLeqDiff(uint64_t* vars, double* weight)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    double block_dim_cost = GetSmoothblock_dimCost(vars);
    block_dim_cost *= weight[1] > 0 ? weight[1] : 0;
    double ub_size_cost = GetSmoothub_sizeCost(vars);
    ub_size_cost *= weight[0] > 0 ? weight[0] : 0;
    double leq1_cost = (wbo_size - A);
    leq1_cost *= weight[2] > 0 ? weight[2] : 0;
    double leq2_cost = (wio_size - R);
    leq2_cost *= weight[3] > 0 ? weight[3] : 0;
    return block_dim_cost + ub_size_cost + leq1_cost + leq2_cost;
}
bool GeneralSolvercase1152::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == wbo_size_idx) {
        return leqs[1] <= 0 && leqs[2] <= 0;
    } else if (idx == wio_size_idx) {
        return leqs[0] <= 0 && leqs[3] <= 0;
    }
    return true;
}

void GeneralSolvercase1152::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double wbo_size = static_cast<double>(vars[wbo_size_idx]);
    double wio_size = static_cast<double>(vars[wio_size_idx]);
    if (idx == wbo_size_idx) {
        leqs[1] = (Max(0, ceiling((A / (wbo_size)))) - block_dim);
        leqs[2] = (wbo_size - A);
    } else if (idx == wio_size_idx) {
        leqs[0] = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
        leqs[3] = (wio_size - R);
    } else if (idx == -1) {
        leqs[0] = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)) - ub_size);
        leqs[1] = (Max(0, ceiling((A / (wbo_size)))) - block_dim);
        leqs[2] = (wbo_size - A);
        leqs[3] = (wio_size - R);
    }
}

void GeneralSolvercase1152::DisplayVarVal(uint64_t* vars)
{
    uint64_t wbo_size = vars[wbo_size_idx];
    uint64_t wio_size = vars[wio_size_idx];
    OP_LOGD(OP_NAME, "wbo_size = %lu", static_cast<uint64_t>(wbo_size));
    OP_LOGD(OP_NAME, "wio_size = %lu", static_cast<uint64_t>(wio_size));
}

void GeneralSolvercase1152::MapVarVal(uint64_t* vars, graph_normalTilingData& tiling_data)
{
    uint64_t wbo_size = vars[wbo_size_idx];
    uint64_t wio_size = vars[wio_size_idx];
    OP_LOGD(OP_NAME, "The output of the solver for tilingCaseId case1152 is:");
    tiling_data.set_wbo_size(static_cast<uint64_t>(wbo_size));
    OP_LOGD(OP_NAME, "wbo_size = %u", tiling_data.get_wbo_size());
    tiling_data.set_wio_size(static_cast<uint64_t>(wio_size));
    OP_LOGD(OP_NAME, "wio_size = %u", tiling_data.get_wio_size());
}

bool GeneralSolvercase1152::Init(const SolverInput &input) {
    block_dim = input.corenum;
    return GeneralSolver::Init(input);
}
void GeneralSolvercase1152::GetResult(int32_t solution_num, uint64_t* solution, graph_normalTilingData& tiling_data)
{
    if (solution_num > 0) {
        OP_LOGD(OP_NAME, "Filling tilingdata for case1152.");
        OP_LOGD(OP_NAME, "Estimate the occupy.");
        OP_LOGD(OP_NAME, "block_dim = %ld", static_cast<uint64_t>(Getblock_dimCost(solution) + block_dim));
        OP_LOGD(OP_NAME, "ub_size = %ld", static_cast<uint64_t>(Getub_sizeCost(solution) + ub_size));
        OP_LOGD(OP_NAME, "Simulate the cost.");
        OP_LOGD(OP_NAME, "Objective value for case1152 is %f.", GetObj(solution));
        MapVarVal(solution, tiling_data);
    }
}


class TilingCase1152Impl : public TilingCaseImpl {
 public:
  TilingCase1152Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != 5) {
      OP_LOGW(OP_NAME, "context->GetComputeNodeInfo()->GetInputsNum() != 5, current value is [%lu], invalid input num.", context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }

  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(0)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(1)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(2)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(3)->GetDataType());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetDataType()) != 1, current value is [%u], invalid input dtype.", context->GetInputTensor(4)->GetDataType());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }

  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(0)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(1)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(2)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(3)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(3)->GetStorageFormat());
      return false;
    }
    if (static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2) {
      OP_LOGW(OP_NAME, "static_cast<uint32_t>(context->GetInputTensor(4)->GetStorageFormat()) != 2, current value is [%u], invalid input format.", context->GetInputTensor(4)->GetStorageFormat());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }

  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input0 is too small.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Shape dim for input1 is too small.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input2 is too small.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input3 is too small.");
      return false;
    }
    if (input4_size < 1) {
      OP_LOGW(OP_NAME, "Shape dim for input4 is too small.");
      return false;
    }
    if (input0_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input1_size < 2) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input2_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input3_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }
    if (input4_size < 1) {
      OP_LOGW(OP_NAME, "Irrational shape dim for continous axis map.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }

  bool TilingVarsShapeCheck(gert::TilingContext *context) {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 1; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool CheckInput0Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 0 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput1Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 1 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 2; ++i) {
      axis_map[i] = true;
    }
    axis_map[input_size - 1] = true;
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput2Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 2 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput3Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 3 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool CheckInput4Cover(gert::TilingContext *context) {
    uint64_t input_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();
    uint32_t cover_cnt = 0;
    bool *axis_map = new(std::nothrow) bool[input_size];
    if (axis_map == nullptr) {
      OP_LOGW(OP_NAME, "Create axis_map for input 4 failed.");
      return false;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      axis_map[i] = false;
    }
    for (uint32_t i = 0; i <= input_size - 1; ++i) {
      axis_map[i] = true;
    }
    for (uint32_t i = 0; i < input_size; ++i) {
      cover_cnt += axis_map[i] ? 1 : 0;
    }
    delete[] axis_map;
    return cover_cnt == input_size;
  }

  bool TilingVarsCoverCheck(gert::TilingContext *context) {
    if (!CheckInput0Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 0 cannot cover all axis.");
      return false;
    }
    if (!CheckInput1Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 1 cannot cover all axis.");
      return false;
    }
    if (!CheckInput2Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 2 cannot cover all axis.");
      return false;
    }
    if (!CheckInput3Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 3 cannot cover all axis.");
      return false;
    }
    if (!CheckInput4Cover(context)) {
      OP_LOGW(OP_NAME, "Map for input 4 cannot cover all axis.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsCoverCheck success.");
    return true;
  }


  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    if (!TilingVarsCoverCheck(context)) {
      OP_LOGW(OP_NAME, "TilingInputVarsCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }

  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    if (additional_output_ptr == nullptr) {
      OP_LOGW(OP_NAME, "Attr for additional_output_ptr is null.");
      return false;
    }
    int32_t additional_output = *additional_output_ptr;
    if ((additional_output < 1) || (additional_output > 1)) {
      OP_LOGW(OP_NAME, "(additional_output < 1) || (additional_output > 1), invalid optional att.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (TilingInputVarsCheck(context) != true) {
      return false;
    }
    if (TilingAttrCheck(context) != true) {
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1152.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1152.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  void GetTilingData(graph_normalTilingData &tiling_data, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(tiling_data.get_A());
    to_tiling.set_R(tiling_data.get_R());
    to_tiling.set_ub_size(tiling_data.get_ub_size());
  }
  void SetTilingData(graph_normalTilingData &from_tiling, graph_normalTilingData &tiling_data) {
    tiling_data.set_A(from_tiling.get_A());
    tiling_data.set_R(from_tiling.get_R());
    tiling_data.set_wbo_size(from_tiling.get_wbo_size());
    tiling_data.set_wio_size(from_tiling.get_wio_size());
    tiling_data.set_block_dim(from_tiling.get_block_dim());
    tiling_data.set_A_aligned_size(from_tiling.get_A_aligned_size());
    tiling_data.set_Q47(from_tiling.get_Q47());
    tiling_data.set_Q48(from_tiling.get_Q48());
    tiling_data.set_Q49(from_tiling.get_Q49());
    tiling_data.set_Q50(from_tiling.get_Q50());
    tiling_data.set_Q51(from_tiling.get_Q51());
    tiling_data.set_Q52(from_tiling.get_Q52());
    tiling_data.set_Q53(from_tiling.get_Q53());
    tiling_data.set_Q54(from_tiling.get_Q54());
    tiling_data.set_Q55(from_tiling.get_Q55());
    tiling_data.set_R_aligned_size(from_tiling.get_R_aligned_size());
    tiling_data.set_additional_output(from_tiling.get_additional_output());
    tiling_data.set_gm_size(from_tiling.get_gm_size());
    tiling_data.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    tiling_data.set_output0_total_size(from_tiling.get_output0_total_size());
    tiling_data.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    tiling_data.set_output1_total_size(from_tiling.get_output1_total_size());
    tiling_data.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    tiling_data.set_output2_total_size(from_tiling.get_output2_total_size());
    tiling_data.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    tiling_data.set_output3_total_size(from_tiling.get_output3_total_size());
    tiling_data.set_wbo_loop_num(from_tiling.get_wbo_loop_num());
    tiling_data.set_wbo_tail_size(from_tiling.get_wbo_tail_size());
    tiling_data.set_wio_loop_num(from_tiling.get_wio_loop_num());
    tiling_data.set_wio_tail_size(from_tiling.get_wio_tail_size());

    tiling_data.set_workspaceSize(from_tiling.get_workspaceSize());
  }
  bool ExecuteGeneralSolver(graph_normalTilingData& tiling_data) {
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
    uint64_t A = tiling_data.get_A();
    uint64_t R = tiling_data.get_R();
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 2;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 4;
    OP_LOGD(OP_NAME, "The number of variable is %d(wbo_size, wio_size), the number of constraints is %d.", num_var, num_leq);
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(A), static_cast<uint64_t>(R)};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, true};
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
    OP_LOGD(OP_NAME, "wbo_size->init value: %lu, range: [%lu, %lu].", init_vars[0], lower_bound[0], upper_bound[0]);
    OP_LOGD(OP_NAME, "wio_size->init value: %lu, range: [%lu, %lu].", init_vars[1], lower_bound[1], upper_bound[1]);

    std::shared_ptr<GeneralSolvercase1152> solver = std::make_shared<GeneralSolvercase1152>(cfg, tiling_data);
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

  bool DoTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!ExecuteGeneralSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute general solver for tilingCaseId case1152.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute general solver for tilingCaseId case1152 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double wio_size = tiling_data.get_wio_size();

    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double wbo_size = tiling_data.get_wbo_size();

    return Max(0, ceiling((A / (wbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ceiling((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ceiling((R / (wio_size))) * wbo_size));
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ceiling((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ceiling((R / (wio_size))) * wbo_size) + (2.14933092322175 * ceiling((R / (wio_size))) * wbo_size));
  }

  double GetAICORE_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((4 * wio_size / ((-1 + wio_size))) + 4) * ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ceiling((R / (wio_size))) * wbo_size));
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    double AIV_MTE2 = ((((((1.12000000476837 / ((41.4000015258789 + wio_size))) + 0.889999985694885) * 0.019542701500562 * wio_size) + 11.5) * 5 * ceiling((R / (wio_size))) * wbo_size) + (((((5.01000022888184 / ((27240.689453125 + wio_size))) + 1051.66003417969) * 3.28717696022805e-05 * wio_size) + 13.1199998855591) * ceiling((R / (wio_size))) * wbo_size));
    double AIV_MTE3 = ((((0.0174154702434844 * wio_size) + 0.219999998807907) * 2 * ceiling((R / (wio_size))) * wbo_size) + (((0.0346654997578466 * wio_size) + 1.03999996185303) * ceiling((R / (wio_size))) * wbo_size) + (2.14933092322175 * ceiling((R / (wio_size))) * wbo_size));
    double AICORE_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ceiling((R / (wio_size))) * wbo_size));

    return Max(Max(AICORE_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_wbo_loop_num(((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size());
    tiling_data.set_wio_loop_num(((tiling_data.get_R() + tiling_data.get_wio_size()) - 1) / tiling_data.get_wio_size());
    tiling_data.set_wbo_tail_size((tiling_data.get_A() % tiling_data.get_wbo_size()) == 0 ? tiling_data.get_wbo_size() : (tiling_data.get_A() % tiling_data.get_wbo_size()));
    tiling_data.set_wio_tail_size((tiling_data.get_R() % tiling_data.get_wio_size()) == 0 ? tiling_data.get_wio_size() : (tiling_data.get_R() % tiling_data.get_wio_size()));
  }

  void SetQ47(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q47((2 * wio_size));
  }

  void SetQ48(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q48((2 * wio_size));
  }

  void SetQ49(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q49((2 * wio_size));
  }

  void SetQ50(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q50((2 * wio_size));
  }

  void SetQ51(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q51((4 * wio_size));
  }

  void SetQ52(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q52(Max((2 * wio_size), (4 * wio_size)));
  }

  void SetQ53(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q53((4 * wio_size));
  }

  void SetQ54(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q54(4);
  }

  void SetQ55(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q55(4);
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ47(tiling_data);
    SetQ48(tiling_data);
    SetQ49(tiling_data);
    SetQ50(tiling_data);
    SetQ51(tiling_data);
    SetQ52(tiling_data);
    SetQ53(tiling_data);
    SetQ54(tiling_data);
    SetQ55(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1152.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1152 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1152.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1152 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1152.");
    double A = static_cast<double>(tiling_data.get_A());
    double R = static_cast<double>(tiling_data.get_R());
    double wbo_size = static_cast<double>(tiling_data.get_wbo_size());
    tiling_data.set_workspaceSize(static_cast<uint32_t>((4 * Max(0, ceiling((A / (wbo_size)))) * R * wbo_size)));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1152.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling key to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set wbo_size to %u.", tiling_data.get_wbo_size());
    OP_LOGI(OP_NAME, "Set wio_size to %u.", tiling_data.get_wio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AICORE_VEC is %f.", GetAICORE_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 1101u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1101Impl>(corenum);
  } else if (tilingCaseId == 1102u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1102Impl>(corenum);
  } else if (tilingCaseId == 1111u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1111Impl>(corenum);
  } else if (tilingCaseId == 1112u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1112Impl>(corenum);
  } else if (tilingCaseId == 1151u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1151Impl>(corenum);
  } else if (tilingCaseId == 1152u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1152Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
bool FindPerfBetterTilingbyCaseId(uint32_t corenum, double &obj, graph_normalTilingData &tiling_data, gert::TilingContext *context, uint32_t tilingCaseId) {
  double cur_obj;
  graph_normalTilingData tmp_tiling;
  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
  if (tilingCaseImplPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
    return false;
  }
  tilingCaseImplPtr->GetTilingData(tiling_data, tmp_tiling);
  OP_LOGD(OP_NAME, "Construct a backup for tilingCaseId %u.", tilingCaseId);
  if (tilingCaseImplPtr->GetTiling(tmp_tiling, context)) {
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
  return false;
}

bool GetTilingKey(graph_normalTilingData &tiling_data, gert::TilingContext *context, int32_t tilingCaseId = -1) {
  bool ret = false;
  double obj = -1;
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    uint32_t tilingKeys[6] = {1101u, 1102u, 1111u, 1112u, 1151u, 1152u};
    for (const auto &tilingKey : tilingKeys) {
      ret = (FindPerfBetterTilingbyCaseId(corenum, obj, tiling_data, context, tilingKey) || ret);
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
    DurationBegin(TILING_FUNC_DURATION_DOTILING);
    ret = tilingCaseImplPtr->GetTiling(tiling_data, context);
    DurationEnd(TILING_FUNC_DURATION_DOTILING);
  }
  if (!ret) {
    OP_LOGE(OP_NAME, "Failed to execute tiling func.");
  }
  return ret;
}

bool PostTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
// 用户可以自定义函数以修改tiling_data数据与回填的tiling数据
  int64_t workspaceSize = tiling_data.get_workspaceSize();
  int64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
  context->SetTilingKey(tiling_data.get_tiling_key());
  tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());
  size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
  size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
  currentWorkSpace[0] = workspaceSize + sysWorkspaceSize;
  return true;
}

ge::graphStatus GetCtxTiling(gert::TilingContext *context, int32_t tilingCaseId) {
  graph_normalTilingData tiling_data;
  OP_LOGI(OP_NAME, "Start context tiling.");
  if (!GetPlatformInfo(tiling_data, context)) {
    OP_LOGE(OP_NAME, "Get platform info Failed.");
    return ge::GRAPH_FAILED;
  }
  OP_LOGI(OP_NAME, "Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, context, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return ge::GRAPH_FAILED;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context.");
  if (PostTiling(tiling_data, context) != true) {
    OP_LOGE(OP_NAME, "PostTiling Failed.");
    return ge::GRAPH_FAILED;
  }
  OP_LOGI(OP_NAME, "End context tiling.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTiling(gert::TilingContext *context) {
  return GetCtxTiling(context, -1);
}

static ge::graphStatus TilingPrepare4AddLayerNorm(gert::TilingParseContext*) {
  return ge::GRAPH_SUCCESS;
}
struct AddLayerNormCompileInfo {};

IMPL_OP(AddLayerNorm).Tiling(GetTiling).TilingParse<AddLayerNormCompileInfo>(TilingPrepare4AddLayerNorm);
} // namespace optiling

