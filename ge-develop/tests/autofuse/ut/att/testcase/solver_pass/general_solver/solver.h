/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SOLVER_H
#define SOLVER_H
#include <vector>
#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"

#define private public
#define protected public

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
        
        if (top_num <= 0)
        {
            throw "top_num <= 0!";
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
        if (input.var_num <= 0)
        {
            throw "input.var_num <= 0";
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
        if (num_leq <= 0)
        {
            throw "num_leq <= 0!";
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
        if (var_num <= 0)
        {
            throw "var_num <= 0!";
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
    virtual double GetLeqCost(uint64_t *vars) = 0;
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

template<typename T>
inline T ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (a > value) ? (value + 1) : value;
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

#endif