/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_data.h"
#include <memory>
#include <cmath>
#include <cstdlib>
#include <memory.h>
#include <iostream>
#include <algorithm>
#ifdef OPEN_TILING_CTX
#include "exe_graph/runtime/tiling_context.h"
#endif
#ifdef DEBUG
#define ATT_LOG(log)
  do {
    std::cout << "[ERROR]" << log << std::endl;
  } while (0)
#else
#define ATT_LOG(log)
#endif
namespace optiling {
/*
(可修改变量)用于控制通用求解器求解质量的超参数
cfg_top_num:保留目标函数最优的前top_num个解,用户可以打印这些解并从中选取较优项(默认值为5)
cfg_search_length:在可行域内执行局部搜索的搜索范围,当搜索范围内存在更优的解时会将该解视为候选
  搜索范围越大,越有可能获取更优的解,但求解耗时更长(默认值为1)
cfg_iterations:启发式求解算法的迭代轮次上限,算法最多执行iterations次,并在满足早停逻辑时提前退出
  在不满足早停逻辑的前提下,设置更大的iterations算法有机会取得更好的解,但求解耗时更长(默认值为500)
cfg_get_log:在每轮循环时输出tiling_data当前状态信息
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
static const uint64_t cfg_iterations = 3;
static const bool cfg_get_log = false;
static const bool cfg_simple_ver = false;
static const double cfg_momentum_factor = 0.9;
}
#include <cstdint>
#include <iostream>

#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
namespace optiling {
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
            throw "Create head failed!";
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
    bool get_log{true};
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
            throw "var_num = 0!";
        }
        var_num_ = var_num;
        obj_ = obj;
        cons_ = cons;
        vars_ = new(std::nothrow) uint64_t[var_num];
        if (vars_ == nullptr)
        {
            throw "Create vars_ failed!";
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
            throw "top_num = 0!";
        }
        solution_num_ = 0;
        top_n_ = top_num;
        var_num_ = var_num;
        solution_ = new(std::nothrow) VarVal *[top_num];
        if (solution_ == nullptr)
        {
            throw "Create solution_ failed!";
        }
    }
    ~Result()
    {
        for (int32_t i = 0; i < solution_num_; i++)
        {
            delete solution_[i];
        }
        delete[] solution_;
    }
    bool AddVarVal(uint64_t *vars, double obj, double cons);
    bool GetResult(int32_t &solution_num, uint64_t *solution);

private:
    int32_t top_n_{0};
    int32_t var_num_{0};
    int32_t solution_num_{0};
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
            throw "Create upper_bound failed!";
        }
        lower_bound = new(std::nothrow) uint64_t[input.var_num];
        if (lower_bound == nullptr)
        {
            throw "Create lower_bound failed!";
        }
        history_vars = new(std::nothrow) uint64_t[input.var_num];
        if (history_vars == nullptr)
        {
            throw "Create history_vars failed!";
        }
        rec_vars = new(std::nothrow) uint64_t[input.var_num];
        if (rec_vars == nullptr)
        {
            throw "Create rec_vars failed!";
        }
        cur_vars = new(std::nothrow) uint64_t[input.var_num];
        if (cur_vars == nullptr)
        {
            throw "Create cur_vars failed!";
        }
        target_val = new(std::nothrow) uint64_t[input.var_num];
        if (target_val == nullptr)
        {
            throw "Create target_val failed!";
        }
        update_last = new(std::nothrow) bool[input.var_num];
        if (update_last == nullptr)
        {
            throw "Create update_last failed!";
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
            throw "num_leq = 0!";
        }
        leq_num = num_leq;
        leqs = new(std::nothrow) double[leq_num];
        if (leqs == nullptr)
        {
            throw "Create leqs failed!";
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
            throw "var_num = 0!";
        }
        momentum = new(std::nothrow) double[var_num];
        if (momentum == nullptr)
        {
            throw "Create momentum failed!";
        }
        cur_value = new(std::nothrow) double[var_num];
        if (cur_value == nullptr)
        {
            throw "Create cur_value failed!";
        }
        is_valid = new(std::nothrow) bool[var_num];
        if (is_valid == nullptr)
        {
            throw "Create is_valid failed!";
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
    explicit GeneralSolver(const SolverConfig &config)
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

    double GetFuncVal(uint64_t *vars, FuncInfo func_info);
    UpdateDirection GetDescent(uint64_t *vars, int32_t idx, FuncInfo func_info);

    virtual void DisplayVarVal(uint64_t *vars) = 0;
    virtual double GetObj(uint64_t *vars) = 0;
    virtual double GetBuffCost(uint64_t *vars) = 0;
    virtual double GetLeqCost(uint64_t *vars) = 0;
    virtual bool CheckLocalValid(double *leqs, int32_t idx) = 0;
    virtual void UpdateLeqs(uint64_t *vars, int32_t idx, double *leqs) = 0;

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
    bool RecordBestVarVal(double obj);
    bool is_feasible_{false};
    bool has_feasible_{false};

    SolverConfig solver_config_;
    Result *result_{nullptr};
    VarInfo *var_info_{nullptr};
    ConsInfo *cons_info_{nullptr};
    Momentum *momentum_info_{nullptr};
    VisitedNode *visited_node_{nullptr};
};

template<typename T>
inline T SMAX(T a, T b)
{
    return (a > b) ? a : b;
}

template<typename T>
inline T SMIN(T a, T b)
{
    return (a < b) ? a : b;
}

inline bool IsEqual(double a, double b)
{
    const double epsilon = 0.001;
    double abs = (a > b) ? (a - b) : (b - a);
    return abs < epsilon;
}

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
    for (int32_t i = 0; i < depth; i++)
    {
        if (!cur_node->next_var)
        {
            new_node = new(std::nothrow) Node(vars[i]);
            if (new_node == nullptr)
            {
                ATT_LOG("Create new_node failed!");
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
                ATT_LOG("Create new_node failed!");
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
        ATT_LOG("Too much solutions!");
        return false;
    }
    int32_t cnt_num = 0;
    int32_t temp_idx = 0;
    double cur_obj;
    double cur_cons;
    bool has_add = false;
    solution_num_ = SMIN(solution_num_ + 1, top_n_);
    VarVal *new_vars = new(std::nothrow) VarVal(var_num_, obj, cons, vars);
    if (new_vars == nullptr)
    {
        ATT_LOG("Create new_vars failed!");
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
        ATT_LOG("Create temp failed!");
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
    }

    for (int32_t i = 0; i < rec_num; i++)
    {
        delete temp[i];
    }
    delete[] temp;
    delete new_vars;

    return cnt_num == solution_num_;
}

bool Result::GetResult(int32_t &solution_num, uint64_t *solution)
{
    for (int32_t i = 0; i < solution_num_; i++)
    {
        solution_[i]->GetVars(solution + i * var_num_);
    }
    solution_num = solution_num_;
    return true;
}

double GeneralSolver::GetFuncVal(uint64_t *vars, FuncInfo func_info)
{
    if (func_info == FuncInfo::BUFFER)
    {
        return GetBuffCost(vars);
    }
    else if (func_info == FuncInfo::LEQ)
    {
        return GetLeqCost(vars);
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
        ATT_LOG("idx illegal!");
        return UpdateDirection::NONE;
    }
    double cur_val = GetFuncVal(vars, func_info);
    if (vars[idx] + 1 <= var_info_->upper_bound[idx])
    {
        vars[idx] += 1;
        double next_val = GetFuncVal(vars, func_info);
        vars[idx] -= 1;
        if (!IsEqual(cur_val, next_val))
        {
            return (cur_val > next_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
        }
    }
    if (vars[idx] >= var_info_->lower_bound[idx] + 1)
    {
        vars[idx] -= 1;
        double pre_val = GetFuncVal(vars, func_info);
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
        ATT_LOG("Create visited_node_ failed!");
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
        ATT_LOG("Create result_ failed!");
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
        ATT_LOG("idx illegal!");
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
        ATT_LOG("idx illegal!");
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
        ATT_LOG("idx illegal!");
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
        double update_value = init_obj - GetObj(var_info_->cur_vars);
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
            ATT_LOG("Create update_info failed!");
            return false;
        }
        if (GetCoarseLoc(update_info, step, cur_locality))
        {
            if (!LocateLoc(update_info, step, cur_locality, best_locality))
            {
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
    Locality best_locality = Locality::REJECT;
    double init_obj = GetObj(var_info_->cur_vars);
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
        return false;
    }
    UpdateBestVar();
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
    cur_obj = GetObj(var_info_->cur_vars);
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
        ATT_LOG("idx illegal!");
        return false;
    }
    uint64_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    double init_obj = update_info->init_obj;
    while (step < SMIN(thres, solver_config_.search_length))
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
        ATT_LOG("idx illegal!");
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
        ATT_LOG("idx illegal!");
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
        cur_obj = GetObj(var_info_->cur_vars);
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
            ATT_LOG("idx illegal!");
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
            ATT_LOG("Create update_info failed!");
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
    double init_obj = GetObj(var_info_->cur_vars);
    double init_cons = GetBuffCost(var_info_->cur_vars);
    if (!RecordBestVarVal(init_obj))
    {
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
        return false;
    }
    UpdateBestVar();
    return true;
}

bool GeneralSolver::RecordBestVarVal(double obj)
{
    if (is_feasible_)
    {
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
  solution_num:int32_t类型的参数,用来输出实际得到的解的个数
  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,求解算法获取到的可行解放入该空间
*/
bool GeneralSolver::Run(int32_t &solution_num, uint64_t *solutions)
{
    Node* cur_node;
    uint64_t iter = 1;
    has_feasible_ = false;
    while (iter <= solver_config_.iterations)
    {
        if (solver_config_.get_log)
        {
            std::cout << "iter:" << iter << std::endl;
            DisplayVarVal(var_info_->cur_vars);
            std::cout << std::endl;
        }
        Initialize(iter);
        if (!is_feasible_)
        {
            if (!LocateRegion())
            {
                break;
            }
        }
        else
        {
            if (SearchVars(var_info_->cur_vars))
            {
                break;
            }
            cur_node = visited_node_->GetVarVal(var_info_->cur_vars);
            if (cur_node == nullptr) {
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
} // namespace optiling

namespace optiling {
using namespace std;
class TilingCaseImpl {
 public:
#ifdef OPEN_TILING_CTX

  virtual bool DoTiling(double &obj, TilingData &tiling_data, gert::TilingContext *context) = 0;
#else
  virtual bool DoTiling(double &obj, TilingData &tiling_data) = 0;
#endif
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;
#ifdef OPEN_TILING_CTX
void Getbmm1Tiling(TilingData &tiling_data, gert::TilingContext *context) {
	auto ascendcPlatform = plat_ascendc::PlatformAscendC(context->GetPlatformInfo());
	MatmulApiTiling tiling(ascendcPlatform);
	tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
	tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
	tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
	tiling.SetShape(tiling_data.s1t_size, tiling_data.s2t_size, tiling_data.D);
	tiling.SetOriShape(tiling_data.s1t_size, tiling_data.s2t_size, tiling_data.D);
	tiling.SetBufferSpace(-1, -1, -1);
	tiling.SetFixSplit(-1, -1, -1);
	tiling.GetTiling(tiling_data.bmm1_tiling);
}

void Getbmm2Tiling(TilingData &tiling_data, gert::TilingContext *context) {
	auto ascendcPlatform = plat_ascendc::PlatformAscendC(context->GetPlatformInfo());
	MatmulApiTiling tiling(ascendcPlatform);
	tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
	tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
	tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
	tiling.SetShape(tiling_data.s1tt_size, tiling_data.D, tiling_data.s2t_size);
	tiling.SetOriShape(tiling_data.s1tt_size, tiling_data.D, tiling_data.s2t_size);
	tiling.SetBufferSpace(-1, -1, -1);
	tiling.SetFixSplit(-1, -1, -1);
	tiling.GetTiling(tiling_data.bmm2_tiling);
}

void GetflashSoftmaxTiling(TilingData &tiling_data) {
	ge::Shape x1_orig_shape= ge::Shape({tiling_data.s1tt_size,tiling_data.s2t_size});
	uint32_t x1_dtype_size = 4;
	uint32_t x2_dtype_size = 4;
	uint32_t x3_size = ((tiling_data.s1tt_size * tiling_data.s2t_size ) * 4);
	bool flashSoftmaxisUpdate = false;
	bool flashSoftmaxisBasicBlock = false;
	SoftMaxFlashV2TilingFunc(x1_orig_shape, x1_dtype_size, x2_dtype_size, x3_size, tiling_data.flashSoftmax_tiling, flashSoftmaxisUpdate, flashSoftmaxisBasicBlock);
}

#endif
void Gencase0AxesTilingData(TilingData& tiling_data) {
	tiling_data.set_bngs1Tb_tail_size(((tiling_data.get_B() * tiling_data.get_G() * tiling_data.get_N() * tiling_data.get_S1() / (tiling_data.get_s1t_size())) % tiling_data.get_bngs1Tb_size()));
	tiling_data.set_bngs1Tb_loop_num((((tiling_data.get_B() * tiling_data.get_G() * tiling_data.get_N() * tiling_data.get_S1() / (tiling_data.get_s1t_size())) + tiling_data.get_bngs1Tb_size()) - 1) / tiling_data.get_bngs1Tb_size());
	tiling_data.set_s1tt2_tail_size((tiling_data.get_s1t_size() % tiling_data.get_s1tt2_size()));
	tiling_data.set_s1tt2_loop_num(((tiling_data.get_s1t_size() + tiling_data.get_s1tt2_size()) - 1) / tiling_data.get_s1tt2_size());
	tiling_data.set_s1tt_size_aligned_size((tiling_data.get_s1tt_size() - 1) * 8 / 8 + 8);
	tiling_data.set_s1tt_size_tail_size((tiling_data.get_s1t_size() % tiling_data.get_s1tt_size()));
	tiling_data.set_s1tt_size_loop_num(((tiling_data.get_s1t_size() + tiling_data.get_s1tt_size()) - 1) / tiling_data.get_s1tt_size());
	tiling_data.set_s2t_size_aligned_size((tiling_data.get_s2t_size() - 1) * 256 / 256 + 256);
	tiling_data.set_s2t_size_tail_size((tiling_data.get_S2() % tiling_data.get_s2t_size()));
	tiling_data.set_s2t_size_loop_num(((tiling_data.get_S2() + tiling_data.get_s2t_size()) - 1) / tiling_data.get_s2t_size());
	tiling_data.set_s1t_size_aligned_size((tiling_data.get_s1t_size() - 1) * 128 / 128 + 128);
	tiling_data.set_s1t_size_tail_size((tiling_data.get_S1() % tiling_data.get_s1t_size()));
	tiling_data.set_s1t_size_loop_num(((tiling_data.get_S1() + tiling_data.get_s1t_size()) - 1) / tiling_data.get_s1t_size());
}

void Gencase0GeneralTilingData(TilingData& tiling_data) {
	tiling_data.set_block_dim(((((tiling_data.get_B() * tiling_data.get_G() * tiling_data.get_N() * tiling_data.get_S1() / (tiling_data.get_s1t_size())) + tiling_data.get_bngs1Tb_size()) - 1) / tiling_data.get_bngs1Tb_size()));
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
  bool RecordBestVarVal(double obj)
    待求解变量的当前值所对应的目标函数寻优
Run函数可以使用下述参数辅助求解:
  cons_info_->leqs, double类型的数组, 用于记录不等式约束的函数值, 其下标含义如下:
    cons_info_->leqs[0] = ((1024 * D * s1t_size_div_align) + (393216 * pow(2, s2t_size_base) * s1t_size_div_align) + (8 * D * s1tt2_size) - GM)
    cons_info_->leqs[1] = ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB)
    cons_info_->leqs[2] = (Max(0, ((double)(1)/(double)(128) * B * G * N * S1 / (bngs1Tb_size * s1t_size_div_align))) - CORENUM)
    cons_info_->leqs[3] = ((128 * s1t_size_div_align) - S1)
    cons_info_->leqs[4] = (bngs1Tb_size - ((double)(1)/(double)(128) * B * G * N * S1 / (s1t_size_div_align)))
    cons_info_->leqs[5] = ((8 * s1tt_size_div_align) - (128 * s1t_size_div_align))
    cons_info_->leqs[6] = (s1tt2_size - (128 * s1t_size_div_align))
  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:
  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界
  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界
*/
class GeneralSolvercase0 : public GeneralSolver
{
    public:
        explicit GeneralSolvercase0(const SolverConfig& config, const TilingData& tiling_data) : GeneralSolver(config) {
            B = tiling_data.get_B();
            D = tiling_data.get_D();
            G = tiling_data.get_G();
            N = tiling_data.get_N();
            S1 = tiling_data.get_S1();
            S2 = tiling_data.get_S2();
            GM = tiling_data.get_GM();
            UB = tiling_data.get_UB();
            CORENUM = tiling_data.get_CORENUM();
            S1 = ((S1 + 128 - 1) / 128) * 128;
            S2 = ((S2 + 256 - 1) / 256) * 256;
        }

        double GetObj(uint64_t* vars) override;
        double GetBuffCost(uint64_t* vars) override;
        double GetLeqCost(uint64_t* vars) override;
        bool CheckLocalValid(double* leqs, int32_t idx) override;
        void DisplayVarVal(uint64_t* vars) override;
        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs) override;
        double GetCORENUMCost(uint64_t* vars);
        double GetGMCost(uint64_t* vars);
        double GetUBCost(uint64_t* vars);
        void MapVarVal(uint64_t* vars, TilingData& tiling_data);
        void GetResult(int32_t solution_num, uint64_t* solution, double& obj, TilingData& tiling_data);
    private:
        const uint64_t bngs1Tb_size_idx = 0;
        const uint64_t s1t_size_div_align_idx = 1;
        const uint64_t s1tt2_size_idx = 2;
        const uint64_t s1tt_size_div_align_idx = 3;
        const uint64_t s2t_size_base_idx = 4;
        uint64_t BL{8};
        uint64_t B;
        uint64_t D;
        uint64_t G;
        uint64_t N;
        uint64_t S1;
        uint64_t S2;
        uint64_t GM;
        uint64_t UB;
        uint64_t CORENUM;
};
/*
函数名:GetCORENUMCost(重要函数)
功能描述:
  根据待求解变量值CORENUM缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetCORENUMCost(uint64_t* vars)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    return (Max(0, ((double)(1)/(double)(128) * B * G * N * S1 / (bngs1Tb_size * s1t_size_div_align))) - CORENUM);
}

/*
函数名:GetGMCost(重要函数)
功能描述:
  根据待求解变量值GM缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetGMCost(uint64_t* vars)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    return ((1024 * D * s1t_size_div_align) + (393216 * pow(2, s2t_size_base) * s1t_size_div_align) + (8 * D * s1tt2_size) - GM);
}

/*
函数名:GetUBCost(重要函数)
功能描述:
  根据待求解变量值UB缓存占用信息(occupy-buff)
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetUBCost(uint64_t* vars)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    return ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB);
}

/*
函数名:GetObj(重要函数)
功能描述:
  根据待求解变量值输出目标函数的罚函数(sigma(max(0, leq)^2))
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetObj(uint64_t* vars)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    double AICORE_VEC = ((((((0.0199999995529652 / (((256 * pow(2, s2t_size_base)) + -64.0))) + 13.1300001144409) * 0.844226280123324 * pow(2, s2t_size_base) * s1tt_size_div_align) + 54.0) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((((1.04999995231628 / (((256 * pow(2, s2t_size_base)) + -47.2700004577637))) + 0.0599999986588955) * 431.157894736842 * pow(2, s2t_size_base) * s1tt_size_div_align) + 101.879997253418) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((((4.09999990463257 / ((174.380004882812 + D))) + 0.0199999995529652) * 1.24999998137355 * D * s1tt2_size) + 55.060001373291) * 128 * bngs1Tb_size * s1t_size_div_align / (s1tt2_size)) + (((((6.1399998664856 / (((256 * pow(2, s2t_size_base)) + -5.59999990463257))) + 0.0599999986588955) * 2301.12363249195 * pow(2, s2t_size_base) * s1tt_size_div_align) + 44.3899993896484) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((0.0130198094690089 * D * s1tt2_size) + 37.9300003051758) * 128 * bngs1Tb_size * s1t_size_div_align / (s1tt2_size)) + (((0.013028792443954 * D * s1tt2_size) + 34.939998626709) * 128 * bngs1Tb_size * s1t_size_div_align / (s1tt2_size)) + (((26.6829669252177 * pow(2, s2t_size_base) * s1tt_size_div_align) + 34.939998626709) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((40.5098911176392 * pow(2, s2t_size_base) * s1tt_size_div_align) + 74.5199966430664) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + ((double)(9)/(double)(4) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (2304 * bngs1Tb_size * s1t_size_div_align / (s1tt2_size)));
    double AIC_MTE2 = ((((((((0.0593999996781349 * D) + 20.0944004058838) * 2.05493742056925 * pow(2, s2t_size_base) * s1t_size_div_align) + (27.1837227059422 * pow(2, s2t_size_base) * s1t_size_div_align)) * ((19.3104000091553 * pow(2, s2t_size_base)) + 1.40000000481422e-09)) + (((((7.19920015335083 * s1t_size_div_align) + 0.398499995470047) * 0.0160541985981972 * D * pow(2, s2t_size_base)) + (0.212372833640173 * D * pow(2, s2t_size_base))) * ((0.0576062500476837 * D) + 2.59999999308036e-09))) * (double)(1)/(double)(256) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) / (((0.000216000000364147 * D) + (0.0462719984352589 * s1t_size_div_align) + (0.147376000881195 * pow(2, s2t_size_base)) + (1.1371240523059e-11 * D * pow(2, s2t_size_base)) + (2.8919753090807e-12 * D * s1t_size_div_align) + (3.43009042157403e-09 * pow(2, s2t_size_base) * s1t_size_div_align)))) + (((((((15.2063999176025 * pow(2, s2t_size_base)) + 20.0944004058838) * 0.00802709929909862 * D * s1t_size_div_align) + (0.106186416820087 * D * s1t_size_div_align)) * ((0.0754312500357628 * D) + 1.40000000481422e-09)) + (((((7.19920015335083 * s1t_size_div_align) + 0.398499995470047) * 0.0160541985981972 * D * pow(2, s2t_size_base)) + (0.212372833640173 * D * pow(2, s2t_size_base))) * ((14.747200012207 * pow(2, s2t_size_base)) + 2.59999999308036e-09))) * (double)(1)/(double)(256) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) / (((0.000575687503442168 * D) + (0.0462719984352589 * s1t_size_div_align) + (0.0552960000932217 * pow(2, s2t_size_base)) + (1.1371240523059e-11 * D * pow(2, s2t_size_base)) + (1.33987907092735e-11 * D * s1t_size_div_align) + (7.40345679124659e-10 * pow(2, s2t_size_base) * s1t_size_div_align)))));
    double AIV_MTE2 = ((((((0.00999999977648258 / (((256 * pow(2, s2t_size_base)) + -64.0))) + 377.209991455078) * 0.0475476961557402 * pow(2, s2t_size_base) * s1tt_size_div_align) + 7.78999996185303) * (double)(1)/(double)(8) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((((1.12000000476837 / (((256 * pow(2, s2t_size_base)) + 41.4000015258789))) + 0.889999985694885) * 40.0234526731509 * pow(2, s2t_size_base) * s1tt_size_div_align) + 11.5) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((((1.35000002384186 / (((256 * pow(2, s2t_size_base)) + 1377.43994140625))) + 0.28999999165535) * 102.862883761709 * pow(2, s2t_size_base) * s1tt_size_div_align) + 4.84999990463257) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)) + (((((1.35000002384186 / ((1377.43994140625 + D))) + 0.28999999165535) * 0.0502260174617721 * D * s1tt2_size) + 4.84999990463257) * 256 * bngs1Tb_size * s1t_size_div_align / (s1tt2_size)));
    double AIV_MTE3 = ((((0.0174154702434844 * D * s1tt2_size) + 0.219999998807907) * 128 * bngs1Tb_size * s1t_size_div_align / (s1tt2_size)) + (((0.277323998062773 * BL * s1tt_size_div_align) + 1.03999996185303) * 16 * bngs1Tb_size * s1t_size_div_align / (s1tt_size_div_align)) + (((35.666883058656 * pow(2, s2t_size_base) * s1tt_size_div_align) + 0.219999998807907) * (double)(1)/(double)(16) * S2 * bngs1Tb_size * pow(2, (-1 * s2t_size_base)) * s1t_size_div_align / (s1tt_size_div_align)));
    return Max(Max(Max(AICORE_VEC, AIC_MTE2), AIV_MTE2), AIV_MTE3);
}
/*
函数名:GetBuffCost(重要函数)
功能描述:
  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetBuffCost(uint64_t* vars)
{
    double CORENUM_cost = GetCORENUMCost(vars);
    double GM_cost = GetGMCost(vars);
    double UB_cost = GetUBCost(vars);
    return ((Min(0, CORENUM_cost) * Min(0, CORENUM_cost)) + (Min(0, GM_cost) * Min(0, GM_cost)) + (Min(0, UB_cost) * Min(0, UB_cost)));
}
/*
函数名:GetLeqCost(重要函数)
功能描述:
  根据待求解变量值输出不等式约束的罚函数(sigma(max(0, leq)^2))
输入参数:
  vars:一个长度为num_var的数组,对应了待求解变量
*/
double GeneralSolvercase0::GetLeqCost(uint64_t* vars)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    double CORENUM_cost = GetCORENUMCost(vars);
    double GM_cost = GetGMCost(vars);
    double UB_cost = GetUBCost(vars);
    double leq1_cost = ((128 * s1t_size_div_align) - S1);
    double leq2_cost = (bngs1Tb_size - ((double)(1)/(double)(128) * B * G * N * S1 / (s1t_size_div_align)));
    double leq3_cost = ((8 * s1tt_size_div_align) - (128 * s1t_size_div_align));
    double leq4_cost = (s1tt2_size - (128 * s1t_size_div_align));
    return ((Max(0, CORENUM_cost) * Max(0, CORENUM_cost)) + (Max(0, GM_cost) * Max(0, GM_cost)) + (Max(0, UB_cost) * Max(0, UB_cost)) + (Max(0, leq1_cost) * Max(0, leq1_cost)) + (Max(0, leq2_cost) * Max(0, leq2_cost)) + (Max(0, leq3_cost) * Max(0, leq3_cost)) + (Max(0, leq4_cost) * Max(0, leq4_cost)));
}
bool GeneralSolvercase0::CheckLocalValid(double* leqs, int32_t idx)
{
    if (idx == bngs1Tb_size_idx) {
        return leqs[2] <= 0 && leqs[4] <= 0;
    } else if (idx == s1t_size_div_align_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0 && leqs[2] <= 0 && leqs[3] <= 0 && leqs[4] <= 0 && leqs[5] <= 0 && leqs[6] <= 0;
    } else if (idx == s1tt2_size_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0 && leqs[6] <= 0;
    } else if (idx == s1tt_size_div_align_idx) {
        return leqs[1] <= 0 && leqs[5] <= 0;
    } else if (idx == s2t_size_base_idx) {
        return leqs[0] <= 0 && leqs[1] <= 0;
    }
    return true;
}

void GeneralSolvercase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    if (idx == bngs1Tb_size_idx) {
        leqs[2] = (Max(0, ((double)(1)/(double)(128) * B * G * N * S1 / (bngs1Tb_size * s1t_size_div_align))) - CORENUM);
        leqs[4] = (bngs1Tb_size - ((double)(1)/(double)(128) * B * G * N * S1 / (s1t_size_div_align)));
    } else if (idx == s1t_size_div_align_idx) {
        leqs[0] = ((1024 * D * s1t_size_div_align) + (393216 * pow(2, s2t_size_base) * s1t_size_div_align) + (8 * D * s1tt2_size) - GM);
        leqs[1] = ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB);
        leqs[2] = (Max(0, ((double)(1)/(double)(128) * B * G * N * S1 / (bngs1Tb_size * s1t_size_div_align))) - CORENUM);
        leqs[3] = ((128 * s1t_size_div_align) - S1);
        leqs[4] = (bngs1Tb_size - ((double)(1)/(double)(128) * B * G * N * S1 / (s1t_size_div_align)));
        leqs[5] = ((8 * s1tt_size_div_align) - (128 * s1t_size_div_align));
        leqs[6] = (s1tt2_size - (128 * s1t_size_div_align));
    } else if (idx == s1tt2_size_idx) {
        leqs[0] = ((1024 * D * s1t_size_div_align) + (393216 * pow(2, s2t_size_base) * s1t_size_div_align) + (8 * D * s1tt2_size) - GM);
        leqs[1] = ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB);
        leqs[6] = (s1tt2_size - (128 * s1t_size_div_align));
    } else if (idx == s1tt_size_div_align_idx) {
        leqs[1] = ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB);
        leqs[5] = ((8 * s1tt_size_div_align) - (128 * s1t_size_div_align));
    } else if (idx == s2t_size_base_idx) {
        leqs[0] = ((1024 * D * s1t_size_div_align) + (393216 * pow(2, s2t_size_base) * s1t_size_div_align) + (8 * D * s1tt2_size) - GM);
        leqs[1] = ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB);
    } else if (idx == -1) {
        leqs[0] = ((1024 * D * s1t_size_div_align) + (393216 * pow(2, s2t_size_base) * s1t_size_div_align) + (8 * D * s1tt2_size) - GM);
        leqs[1] = ((24576 * pow(2, s2t_size_base) * s1tt_size_div_align) + (2560 * BL * s1t_size_div_align) + (4 * D * s1tt2_size) + Max((4 * D * s1tt2_size), (8192 * pow(2, s2t_size_base) * s1tt_size_div_align)) - UB);
        leqs[2] = (Max(0, ((double)(1)/(double)(128) * B * G * N * S1 / (bngs1Tb_size * s1t_size_div_align))) - CORENUM);
        leqs[3] = ((128 * s1t_size_div_align) - S1);
        leqs[4] = (bngs1Tb_size - ((double)(1)/(double)(128) * B * G * N * S1 / (s1t_size_div_align)));
        leqs[5] = ((8 * s1tt_size_div_align) - (128 * s1t_size_div_align));
        leqs[6] = (s1tt2_size - (128 * s1t_size_div_align));
    }
}

void GeneralSolvercase0::DisplayVarVal(uint64_t* vars)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    std::cout << "s1t_size = " << (128 * s1t_size_div_align) << std::endl;
    std::cout << "bngs1Tb_size = " << bngs1Tb_size << std::endl;
    std::cout << "s2t_size = " << (256 * pow(2, s2t_size_base)) << std::endl;
    std::cout << "s1tt_size = " << (8 * s1tt_size_div_align) << std::endl;
    std::cout << "s1tt2_size = " << s1tt2_size << std::endl;
}

void GeneralSolvercase0::MapVarVal(uint64_t* vars, TilingData& tiling_data)
{
    double bngs1Tb_size = static_cast<double>(vars[bngs1Tb_size_idx]);
    double s1t_size_div_align = static_cast<double>(vars[s1t_size_div_align_idx]);
    double s1tt2_size = static_cast<double>(vars[s1tt2_size_idx]);
    double s1tt_size_div_align = static_cast<double>(vars[s1tt_size_div_align_idx]);
    double s2t_size_base = static_cast<double>(vars[s2t_size_base_idx]);
    tiling_data.set_s1t_size((128 * s1t_size_div_align));
    tiling_data.set_bngs1Tb_size(bngs1Tb_size);
    tiling_data.set_s2t_size((256 * pow(2, s2t_size_base)));
    tiling_data.set_s1tt_size((8 * s1tt_size_div_align));
    tiling_data.set_s1tt2_size(s1tt2_size);
}

void GeneralSolvercase0::GetResult(int32_t solution_num, uint64_t* solution, double& obj, TilingData& tiling_data)
{
    if (solution_num > 0) {
        MapVarVal(solution, tiling_data);
        obj = GetObj(solution);
    }
}

bool Gencase0GeneralSolver(double& obj, TilingData& tiling_data)
{
    SolverConfig cfg;
    cfg.top_num = cfg_top_num;
    cfg.search_length = cfg_search_length;
    cfg.iterations = cfg_iterations;
    cfg.get_log = cfg_get_log;
    cfg.simple_ver = cfg_simple_ver;
    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);

    // 以下参数若未注明是可修改参数,则不建议修改
    uint64_t B = tiling_data.get_B();
    uint64_t G = tiling_data.get_G();
    uint64_t N = tiling_data.get_N();
    uint64_t S1 = tiling_data.get_S1();
    S1 = ((S1 + 128 - 1) / 128) * 128;
    uint64_t S2 = tiling_data.get_S2();
    S2 = ((S2 + 256 - 1) / 256) * 256;
    // 由modelinfo传入的待求解变量个数
    int32_t num_var = 5;
    // 由modelinfo传入的不等式约束个数
    int32_t num_leq = 7;
    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解
    uint64_t init_vars[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>((log(((double)(1)/(double)(256) * S2)) / (log(2))))};
    // (可修改参数) 待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解
    uint64_t upper_bound[num_var] = {static_cast<uint64_t>((B * G * N * S1)), static_cast<uint64_t>(((double)(1)/(double)(128) * S1)), static_cast<uint64_t>(S1), static_cast<uint64_t>(((double)(1)/(double)(8) * S1)), static_cast<uint64_t>((log(((double)(1)/(double)(256) * S2)) / (log(2))))};
    // (可修改参数) 待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解
    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(1), static_cast<uint64_t>(0)};
    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值
    bool update_last[num_var] = {false, false, false, false, true};
    // 初始化解的个数为0
    int32_t solution_num = 0;
    // 为求解器的输出分配内存
    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];
    if (solution == nullptr)
    {
        ATT_LOG("Create solution failed!");
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

    GeneralSolvercase0* solver = new(std::nothrow) GeneralSolvercase0(cfg, tiling_data);
    if (solver != nullptr) {
        // 导入通用求解器的输入参数并完成初始化
        if (solver -> Init(input)) {
            // 运行通用求解器并获取算法的解
            if (solver -> Run(solution_num, solution)) {
                solver -> GetResult(solution_num, solution, obj, tiling_data);
                delete solver;
                delete[] solution;
                return true;
            }
        }
    }
    if (solver != nullptr) {
        delete solver;
    }
    if (solution != nullptr) {
        delete[] solution;
    }
    return false;
}


class TilingCase0Impl : public TilingCaseImpl {
 public:
#ifdef OPEN_TILING_CTX
  bool DoTiling(double& obj, TilingData &tiling_data, gert::TilingContext *context) override {
#else
  bool DoTiling(double& obj, TilingData &tiling_data) override {
#endif
    if (!Gencase0GeneralSolver(obj, tiling_data)) {
        return false;
    }

    ComputeOptionParam(tiling_data);
#ifdef OPEN_TILING_CTX
	Getbmm1Tiling(tiling_data, context);
	Getbmm2Tiling(tiling_data, context);
	GetflashSoftmaxTiling(tiling_data);
#endif
		Gencase0AxesTilingData(tiling_data);
		Gencase0GeneralTilingData(tiling_data);

    return true;
  }
 private:
  void SetBUF0(TilingData &tiling_data) {
    const auto s1tt_size = tiling_data.get_s1tt_size();
    const auto s2t_size = tiling_data.get_s2t_size();
    tiling_data.set_BUF0((2 * s1tt_size * s2t_size));
  }
  void SetBUF1(TilingData &tiling_data) {
    const auto s1tt_size = tiling_data.get_s1tt_size();
    const auto s2t_size = tiling_data.get_s2t_size();
    const auto D = tiling_data.get_D();
    const auto s1tt2_size = tiling_data.get_s1tt2_size();
    tiling_data.set_BUF1(Max((4 * s1tt_size * s2t_size), (4 * D * s1tt2_size)));
  }
  void SetBUF2(TilingData &tiling_data) {
    const auto s1tt_size = tiling_data.get_s1tt_size();
    const auto s2t_size = tiling_data.get_s2t_size();
    tiling_data.set_BUF2((s1tt_size * s2t_size));
  }
  void SetBUF3(TilingData &tiling_data) {
    const auto s1tt_size = tiling_data.get_s1tt_size();
    const auto s2t_size = tiling_data.get_s2t_size();
    tiling_data.set_BUF3((s1tt_size * s2t_size));
  }
  void SetBUF4(TilingData &tiling_data) {
    const auto BL = tiling_data.get_BL();
    const auto s1t_size = tiling_data.get_s1t_size();
    tiling_data.set_BUF4((4 * BL * s1t_size));
  }
  void SetBUF5(TilingData &tiling_data) {
    const auto D = tiling_data.get_D();
    const auto s1tt2_size = tiling_data.get_s1tt2_size();
    tiling_data.set_BUF5((4 * D * s1tt2_size));
  }
  void SetQ0(TilingData &tiling_data) {
    const auto s2t_size = tiling_data.get_s2t_size();
    const auto s1t_size = tiling_data.get_s1t_size();
    tiling_data.set_Q0((8 * s1t_size * s2t_size));
  }
  void SetQ1(TilingData &tiling_data) {
    const auto s1tt_size = tiling_data.get_s1tt_size();
    const auto s2t_size = tiling_data.get_s2t_size();
    tiling_data.set_Q1((8 * s1tt_size * s2t_size));
  }
  void SetQ2(TilingData &tiling_data) {
    const auto BL = tiling_data.get_BL();
    const auto s1t_size = tiling_data.get_s1t_size();
    tiling_data.set_Q2((8 * BL * s1t_size));
  }
  void SetQ3(TilingData &tiling_data) {
    const auto BL = tiling_data.get_BL();
    const auto s1t_size = tiling_data.get_s1t_size();
    tiling_data.set_Q3((8 * BL * s1t_size));
  }
  void SetQ4(TilingData &tiling_data) {
    const auto s2t_size = tiling_data.get_s2t_size();
    const auto s1t_size = tiling_data.get_s1t_size();
    tiling_data.set_Q4((4 * s1t_size * s2t_size));
  }
  void SetQ5(TilingData &tiling_data) {
    const auto D = tiling_data.get_D();
    const auto s1t_size = tiling_data.get_s1t_size();
    tiling_data.set_Q5((8 * D * s1t_size));
  }
  void SetQ6(TilingData &tiling_data) {
    const auto D = tiling_data.get_D();
    const auto s1tt2_size = tiling_data.get_s1tt2_size();
    tiling_data.set_Q6((8 * D * s1tt2_size));
  }
  void Setgm_size(TilingData &tiling_data) {
    const auto D = tiling_data.get_D();
    const auto s1tt2_size = tiling_data.get_s1tt2_size();
    const auto s1t_size = tiling_data.get_s1t_size();
    const auto s2t_size = tiling_data.get_s2t_size();
    tiling_data.set_gm_size(((12 * s1t_size * s2t_size) + (8 * D * s1t_size) + (8 * D * s1tt2_size)));
  }

  void ComputeOptionParam(TilingData &tiling_data) {
    SetBUF0(tiling_data);
    SetBUF1(tiling_data);
    SetBUF2(tiling_data);
    SetBUF3(tiling_data);
    SetBUF4(tiling_data);
    SetBUF5(tiling_data);
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    Setgm_size(tiling_data);

  }
};
#ifdef OPEN_TILING_CTX
void AssignAttAndOutputSize0(TilingData &tiling_data, gert::TilingContext *context)
{
  auto attrs = context->GetAttrs();
  auto head_num_ptr = attrs->GetAttrPointer<int32_t>(1U);
  int32_t head_num = *head_num_ptr;
  tiling_data.set_head_num(head_num);
  tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
  tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / tiling_data.CORENUM);
  tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
  tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / tiling_data.CORENUM);
}
#endif

#ifdef OPEN_TILING_CTX
bool TilingVarsValidCheck0(TilingData &tiling_data, gert::TilingContext *context)
#else
bool TilingVarsValidCheck0(TilingData &tiling_data)
#endif
{
#ifdef OPEN_TILING_CTX
  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    ATT_LOG("platformInfoPtr is nullptr!");
    return false;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  auto ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto l1Size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
  auto l0cSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
  auto attrs = context->GetAttrs();
  auto head_num_ptr = attrs->GetAttrPointer<int32_t>(1U);
  if (head_num_ptr == nullptr) {
    ATT_LOG("head_num_ptr is nullptr!");
    return false;
  }
  int32_t head_num = *head_num_ptr;
  if ((head_num < 1) || (head_num > 10)) {
    ATT_LOG("(head_num < 1) || (head_num > 10), invalid optional att!");
    return false;
  }
#endif
  if ((tiling_data.get_B() < 1) || (tiling_data.get_B() > 100000)) {
    ATT_LOG("(tiling_data.get_B() < 1) || (tiling_data.get_B() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_D() < 1) || (tiling_data.get_D() > 100000)) {
    ATT_LOG("(tiling_data.get_D() < 1) || (tiling_data.get_D() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_G() < 1) || (tiling_data.get_G() > 100000)) {
    ATT_LOG("(tiling_data.get_G() < 1) || (tiling_data.get_G() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_N() < 1) || (tiling_data.get_N() > 100000)) {
    ATT_LOG("(tiling_data.get_N() < 1) || (tiling_data.get_N() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_S1() < 1) || (tiling_data.get_S1() > 100000)) {
    ATT_LOG("(tiling_data.get_S1() < 1) || (tiling_data.get_S1() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_S2() < 1) || (tiling_data.get_S2() > 100000)) {
    ATT_LOG("(tiling_data.get_S2() < 1) || (tiling_data.get_S2() > 100000), invalid input var!");
    return false;
  }
  return true;
}

#ifdef OPEN_TILING_CTX
bool CalTiling(double &obj, TilingData &tiling_data, gert::TilingContext *context, int32_t tilingCaseId) {
  TilingCaseImplPtr tilingCaseImplPtr;
  switch (tilingCaseId) {
    case 0u:
      if (TilingVarsValidCheck0(tiling_data, context) != true) {
        return false;
      }
      tilingCaseImplPtr = std::make_shared<TilingCase0Impl>();
      break;
    default:
      return false;
  }
  if (tilingCaseImplPtr == nullptr) {
    return false;
  }
  return tilingCaseImplPtr->DoTiling(obj, tiling_data, context);
}
#endif


bool CalTiling(double &obj, TilingData &tiling_data, int32_t tilingCaseId) {
  TilingCaseImplPtr tilingCaseImplPtr;
  switch (tilingCaseId) {
    case 0u:
      if (TilingVarsValidCheck0(tiling_data) != true) {
        return false;
      }
      tilingCaseImplPtr = std::make_shared<TilingCase0Impl>();
      break;
    default:
      return false;
  }
  if (tilingCaseImplPtr == nullptr) {
    return false;
  }
  return tilingCaseImplPtr->DoTiling(obj, tiling_data);
}


#ifdef OPEN_TILING_CTX
bool TryTiling(double &obj, TilingData &tiling_data, gert::TilingContext *context, uint32_t tilingCaseId)
#else
bool TryTiling(double &obj, TilingData &tiling_data, uint32_t tilingCaseId)
#endif
{
  bool ret{false};
  double cur_obj;
  TilingData tmp_tiling = tiling_data;
#ifdef OPEN_TILING_CTX
  ret = CalTiling(cur_obj, tmp_tiling, context, tilingCaseId);
#else
  ret = CalTiling(cur_obj, tmp_tiling, tilingCaseId);
#endif
  if (ret == true && (obj < 0 || cur_obj < obj)) {
    tiling_data = tmp_tiling;
    obj = cur_obj;
    tiling_data.set_tiling_key(tilingCaseId);
  }
  return ret;
}


#ifdef OPEN_TILING_CTX
bool GetTiling(TilingData &tiling_data, gert::TilingContext *context, int32_t tilingCaseId)
#else
bool GetTiling(TilingData &tiling_data, int32_t tilingCaseId)
#endif
{
  bool ret;
  double obj = -1;
  switch(tilingCaseId) {
    case 0:
#ifdef OPEN_TILING_CTX
      ret = CalTiling(obj, tiling_data, context, tilingCaseId);
#else
      ret = CalTiling(obj, tiling_data, tilingCaseId);
#endif
      tiling_data.set_tiling_key(tilingCaseId);
      break;
    default:
    ret = true;
#ifdef OPEN_TILING_CTX
      ret = TryTiling(obj, tiling_data, context, 0) || ret;
#else
      ret = TryTiling(obj, tiling_data, 0) || ret;
#endif
  }
  if (ret != true) {
    return ret;
  }
#ifdef OPEN_TILING_CTX
  switch(tilingCaseId) {
    case 0u:
      AssignAttAndOutputSize0(tiling_data, context);
      break;
    default:
      return false;
  }
#endif
  return true;
}
} // namespace optiling

