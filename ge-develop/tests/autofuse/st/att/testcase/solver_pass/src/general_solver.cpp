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
#include <cstdint>

enum class Locality
{
    GLOBALVALID,
    LOCALVALID,
    CROSSREGION,
    INVALID,
    ALTERNATIVE,
    REJECT,
};

enum class TunePriority
{
    HARMLESS,
    DILATED,
    NORMAL,
    OTHER,
    TABU,
    REFUSE,
};

enum class FuncInfo
{
    LEQ,
    BUFFER,
};

enum class UpdateDirection
{
    POSITIVE,
    NONE,
    NEGATIVE,
};

struct UpdateInfo
{
    int32_t idx{0};
    uint32_t thres{0};
    UpdateDirection update_direction{UpdateDirection::NONE};
    double init_obj{0};
    double init_cons{0};
    UpdateInfo(int32_t idx, uint32_t thres, UpdateDirection direction, double obj = 0, double cons = 0) : idx(idx), thres(thres), update_direction(direction), init_obj(obj), init_cons(cons) {}
};

struct Node
{
    uint32_t value{0};
    bool searched{false};
    Node *next_val{nullptr};
    Node *next_var{nullptr};
    Node *next_node{nullptr};
    explicit Node(uint32_t val) : value(val) {}
};

class VisitedNode
{
public:
    explicit VisitedNode(int32_t var_num) : depth(var_num)
    {
        head = new Node(0);
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
    Node *GetVarVal(uint32_t *vars);

private:
    int32_t depth{0};
    Node *head{nullptr};
    Node *tail{nullptr};
};

struct SolverInput
{
    int32_t var_num{0};
    int32_t leq_num{0};
    uint32_t *upper_bound{nullptr};
    uint32_t *cur_vars{nullptr};
    bool *update_last{nullptr};
};

struct SolverConfig
{
    int32_t top_num{5};
    int32_t search_length{1};
    int32_t iterations{500};
    bool simple_ver{false};
    bool get_log{true};
    double momentum_factor{0.9};
};

class VarVal
{
public:
    ~VarVal()
    {
        delete[] vars_;
    }
    void GetVarInfo(double &obj, double &cons) const;
    void GetVars(uint32_t *vars);
    bool SetVarVal(int32_t var_num, double obj, double cons, uint32_t *varval);

private:
    int32_t var_num_{0};
    double obj_{0};
    double cons_{0};
    uint32_t *vars_{nullptr};
};

class Result
{
public:
    ~Result()
    {
        for (int32_t i = 0; i < solution_num_; i++)
        {
            delete solution_[i];
        }
        delete[] solution_;
    }
    bool SetResult(int32_t top_num, int32_t var_num);
    bool AddVarVal(uint32_t *vars, double obj, double cons);
    bool GetResult(int32_t &solution_num, uint32_t *solution);

private:
    int32_t top_n_{0};
    int32_t var_num_{0};
    int32_t solution_num_{0};
    VarVal **solution_{nullptr};
};

struct VarInfo
{
    int32_t var_num{0};
    int32_t chosen_var_idx{-1};
    uint32_t *upper_bound{nullptr};
    uint32_t *history_vars{nullptr};
    uint32_t *rec_vars{nullptr};
    uint32_t *cur_vars{nullptr};
    uint32_t *target_val{nullptr};
    bool *update_last{nullptr};
    ~VarInfo()
    {
        delete[] upper_bound;
        delete[] history_vars;
        delete[] rec_vars;
        delete[] cur_vars;
        delete[] target_val;
        delete[] update_last;
    }
};

struct ConsInfo
{
    int32_t leq_num{0};
    double *leqs{nullptr};
    ~ConsInfo()
    {
        delete[] leqs;
    }
};

struct Momentum
{
    double *momentum{nullptr};
    double *cur_value{nullptr};
    bool *is_valid{nullptr};
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
    ~GeneralSolver()
    {
        delete var_info_;
        delete cons_info_;
        delete momentum_info_;
        delete visited_node_;
        delete result_;
    }

    bool Init(const SolverInput &input);
    bool Run(int32_t &solution_num, uint32_t *solutions);

    int32_t GetVarNum() const;
    bool SetCurVars(uint32_t *vars);
    bool SetUpperBound(uint32_t *upper_bound);

    void PrintCurVars(int32_t iter);

    double GetFuncVal(uint32_t *vars, FuncInfo func_info);
    UpdateDirection GetDescent(uint32_t *vars, int32_t idx, FuncInfo func_info);

    virtual void MapVarVal(uint32_t *vars) = 0;
    virtual double GetObj(uint32_t *vars) = 0;
    virtual double GetBuffCost(uint32_t *vars) = 0;
    virtual double GetLeqCost(uint32_t *vars) = 0;
    virtual bool CheckLocalValid(double *leqs, int32_t idx) = 0;
    virtual void UpdateLeqs(uint32_t *vars, int32_t idx, double *leqs) = 0;

private:
    bool SetVarInfo(const SolverInput &input);
    bool SetConsInfo(int32_t leq_num);
    bool SetMomentum(int32_t var_num);
    bool SetResult(int32_t top_num, int32_t var_num);

    bool SetSolverInput(const SolverInput &input);
    bool SearchVars(uint32_t *vars) const;
    bool UpdateCurVarVal(uint32_t value, int32_t idx);

    Locality GetLocality(int32_t idx, UpdateDirection update_direction);
    bool GetCoarseLoc(const UpdateInfo *update_info, uint32_t &step, Locality &cur_locality);
    bool GetFineLoc(const UpdateInfo *update_info, uint32_t &step, Locality &cur_locality);
    bool GetPeerLoc(const UpdateInfo *update_info, Locality &cur_locality);
    bool LocateLoc(const UpdateInfo *update_info, uint32_t &step, Locality &cur_locality, Locality &best_locality);
    bool TryLocate(int32_t idx, double init_obj, Locality &best_locality);

    TunePriority GetTunePriority(int32_t idx, double rec_obj, double &cur_obj);
    bool SearchLoc(const UpdateInfo *update_info, uint32_t &step, double &cur_obj, TunePriority &cur_priority);
    bool GetHarmlessLoc(const UpdateInfo *update_info, uint32_t &step, double &cur_obj);
    bool GetDilatedLoc(const UpdateInfo *update_info, uint32_t &step);
    bool TuneLoc(const UpdateInfo *update_info, double cur_obj, uint32_t &step, TunePriority &cur_priority, TunePriority &best_priority);
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
    bool is_feasible_;
    bool has_feasible_;

    SolverConfig solver_config_;
    Result *result_{nullptr};
    VarInfo *var_info_{nullptr};
    ConsInfo *cons_info_{nullptr};
    Momentum *momentum_info_{nullptr};
    VisitedNode *visited_node_{nullptr};
};

inline int32_t SMAX(int32_t a, int32_t b)
{
    return (a > b) ? a : b;
}

inline int32_t SMIN(int32_t a, int32_t b)
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
    return (update_direction == UpdateDirection::POSITIVE) ? positive : (update_direction == UpdateDirection::NEGATIVE ? negative : none);
}

inline uint32_t Bound(uint32_t upper_bound, uint32_t lower_bound, uint32_t val, uint32_t step, UpdateDirection direction)
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

void VarVal::GetVars(uint32_t *vars)
{
    for (int32_t i = 0; i < var_num_; i++)
    {
        vars[i] = vars_[i];
    }
}

bool VarVal::SetVarVal(int32_t var_num, double obj, double cons, uint32_t *varval)
{
    if (var_num > 0)
    {
        var_num_ = var_num;
        obj_ = obj;
        cons_ = cons;
        vars_ = new uint32_t[var_num];
        for (int32_t i = 0; i < var_num; i++)
        {
            vars_[i] = varval[i];
        }
        return true;
    }
    return false;
}

// 是否需要增强局部搜索的质量
Node *VisitedNode::GetVarVal(uint32_t *vars)
{
    Node *cur = head;
    Node *new_node;
    for (int32_t i = 0; i < depth; i++)
    {
        if (!cur->next_var)
        {
            new_node = new Node(vars[i]);
            cur->next_var = new_node;
            tail->next_node = new_node;
            tail = tail->next_node;
        }
        cur = cur->next_var;
        while (cur->next_val != nullptr)
        {
            if (cur->value == vars[i])
            {
                break;
            }
            cur = cur->next_val;
        }
        if (cur->value != vars[i])
        {
            new_node = new Node(vars[i]);
            cur->next_val = new_node;
            tail->next_node = new_node;
            tail = tail->next_node;
            cur = new_node;
        }
    }
    return cur;
}

bool Result::SetResult(int32_t top_num, int32_t var_num)
{
    if (top_num > 0)
    {
        solution_num_ = 0;
        top_n_ = top_num;
        var_num_ = var_num;
        solution_ = new VarVal *[top_num];
        return true;
    }
    return false;
}

bool Result::AddVarVal(uint32_t *vars, double obj, double cons)
{
    int32_t rec_num = solution_num_;
    VarVal *new_vars = new VarVal();
    VarVal **temp = new VarVal *[rec_num];

    solution_num_ = SMIN(solution_num_ + 1, top_n_);

    for (int32_t i = 0; i < rec_num; i++)
    {
        temp[i] = solution_[i];
    }

    if (!new_vars->SetVarVal(var_num_, obj, cons, vars))
    {
        return false;
    }

    int32_t cnt_num = 0;
    int32_t temp_idx = 0;
    double cur_obj;
    double cur_cons;
    bool has_add = false;

    /*
     temp: 最大容量为top_n的备选可行解集
     先将solution_复制到temp中
     然后比较new_vars的目标值与temp中元素的目标值
     自小到大地将可行解填入solution_
    */
    while (cnt_num < solution_num_ && temp_idx < rec_num)
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

    if (!has_add && cnt_num < solution_num_)
    {
        solution_[cnt_num++] = new_vars;
    }

    for (int32_t i = 0; i < rec_num; i++)
    {
        temp[i] = nullptr;
    }
    delete[] temp;

    return cnt_num == solution_num_;
}

bool Result::GetResult(int32_t &solution_num, uint32_t *solution)
{
    for (int32_t i = 0; i < solution_num_; i++)
    {
        solution_[i]->GetVars(solution + i * var_num_);
    }
    solution_num = solution_num_;
    return true;
}

double GeneralSolver::GetFuncVal(uint32_t *vars, FuncInfo func_info)
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

UpdateDirection GeneralSolver::GetDescent(uint32_t *vars, int32_t idx, FuncInfo func_info)
{
    uint32_t rec = vars[idx];
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
    if (vars[idx] - 1 >= 1)
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

bool GeneralSolver::SetVarInfo(const SolverInput &input)
{
    if (input.var_num > 0)
    {
        var_info_ = new VarInfo();
        var_info_->var_num = input.var_num;
        var_info_->upper_bound = new uint32_t[input.var_num];
        var_info_->history_vars = new uint32_t[input.var_num];
        var_info_->rec_vars = new uint32_t[input.var_num];
        var_info_->cur_vars = new uint32_t[input.var_num];
        var_info_->target_val = new uint32_t[input.var_num];
        var_info_->update_last = new bool[input.var_num];
        SetCurVars(input.cur_vars);
        SetUpperBound(input.upper_bound);
        return true;
    }
    return false;
}

bool GeneralSolver::SetConsInfo(int32_t leq_num)
{
    if (leq_num > 0)
    {
        cons_info_ = new ConsInfo();
        cons_info_->leq_num = leq_num;
        cons_info_->leqs = new double[leq_num];
        return true;
    }
    return false;
}

bool GeneralSolver::SetMomentum(int32_t var_num)
{
    if (var_num > 0)
    {
        momentum_info_ = new Momentum();
        momentum_info_->momentum = new double[var_num];
        momentum_info_->cur_value = new double[var_num];
        momentum_info_->is_valid = new bool[var_num];
        return true;
    }
    return false;
}

bool GeneralSolver::SetSolverInput(const SolverInput &input)
{
    if (input.var_num <= 0)
    {
        return false;
    }
    visited_node_ = new VisitedNode(input.var_num);
    if (SetVarInfo(input) && SetConsInfo(input.leq_num) && SetMomentum(input.var_num))
    {
        for (int32_t i = 0; i < var_info_->var_num; i++)
        {
            var_info_->update_last[i] = input.update_last[i];
        }
        return true;
    }
    return false;
}

bool GeneralSolver::Init(const SolverInput &input)
{
    result_ = new Result();
    if (!SetSolverInput(input))
    {
        return false;
    }
    if (!result_->SetResult(solver_config_.top_num, input.var_num))
    {
        return false;
    }
    return true;
}

bool GeneralSolver::SetCurVars(uint32_t *vars)
{
    for (int32_t i = 0; i < var_info_->var_num; i++)
    {
        if (vars[i] <= 0)
        {
            return false;
        }
        var_info_->cur_vars[i] = vars[i];
    }
    return true;
}

bool GeneralSolver::SetUpperBound(uint32_t *upper_bound)
{
    for (int32_t i = 0; i < var_info_->var_num; i++)
    {
        if (upper_bound[i] <= 0)
        {
            return false;
        }
        var_info_->upper_bound[i] = upper_bound[i];
    }
    return true;
}

void GeneralSolver::PrintCurVars(int32_t iter)
{
    std::cout << "iter = " << iter << std::endl;
    MapVarVal(var_info_->cur_vars);
}

// 仅用于更新特定变量
bool GeneralSolver::UpdateCurVarVal(uint32_t value, int32_t idx)
{
    if (value <= 0)
    {
        return false;
    }
    var_info_->cur_vars[idx] = value;
    UpdateLeqs(var_info_->cur_vars, idx, cons_info_->leqs);
    return true;
}

bool GeneralSolver::SearchVars(uint32_t *vars) const
{
    Node *cur_node = visited_node_->GetVarVal(vars);
    return cur_node->searched;
}

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

bool GeneralSolver::GetCoarseLoc(const UpdateInfo *update_info, uint32_t &step, Locality &cur_locality)
{
    uint32_t update_value;

    int32_t idx = update_info->idx;
    uint32_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    do
    {
        step = (step == 0) ? 1 : (step << 1);
        update_value = Bound(var_info_->upper_bound[idx], 1, var_info_->rec_vars[idx], step, update_direction);
        UpdateCurVarVal(update_value, idx);
        cur_locality = GetLocality(idx, update_direction);
        var_info_->cur_vars[idx] = var_info_->rec_vars[idx];
        if (cur_locality <= Locality::CROSSREGION)
        {
            step = ((cur_locality == Locality::CROSSREGION) && (step != 1)) ? (step >> 1) : step;
            break;
        }
    } while (step < thres);
    update_value = Bound(var_info_->upper_bound[idx], 1, var_info_->rec_vars[idx], step, update_direction);
    UpdateCurVarVal(update_value, idx);
    return thres != 0;
}

bool GeneralSolver::GetFineLoc(const UpdateInfo *update_info, uint32_t &step, Locality &cur_locality)
{
    uint32_t update_value;
    Locality rec_locality;

    int32_t idx = update_info->idx;
    UpdateDirection update_direction = update_info->update_direction;
    if (GetLocality(idx, update_direction) <= Locality::LOCALVALID)
    {
        while (step > 1)
        {
            step >>= 1;
            update_value = var_info_->cur_vars[idx] - GetValue(update_direction) * step;
            UpdateCurVarVal(update_value, idx);
            rec_locality = GetLocality(idx, update_direction);
            update_value = var_info_->cur_vars[idx] + ((rec_locality > Locality::CROSSREGION) ? (GetValue(update_direction) * step) : 0);
            UpdateCurVarVal(update_value, idx);
        }
        cur_locality = GetLocality(idx, update_direction);
    }
    return true;
}

bool GeneralSolver::GetPeerLoc(const UpdateInfo *update_info, Locality &cur_locality)
{
    uint32_t left_value;
    uint32_t right_value;
    uint32_t mid_value;
    Locality rec_locality;

    int32_t idx = update_info->idx;
    uint32_t rec_value = var_info_->cur_vars[idx];
    UpdateDirection update_direction = update_info->update_direction;
    UpdateCurVarVal((update_direction == UpdateDirection::POSITIVE) ? 1 : var_info_->upper_bound[idx], idx);
    rec_locality = GetLocality(idx, update_direction);
    if (rec_locality <= Locality::LOCALVALID)
    {
        var_info_->cur_vars[idx] = rec_value;
    }
    else
    {
        left_value = (update_direction == UpdateDirection::POSITIVE) ? (rec_value + 1) : 1;
        right_value = (update_direction == UpdateDirection::POSITIVE) ? (var_info_->upper_bound[idx]) : (rec_value - 1);
        while (left_value < right_value)
        {
            mid_value = (left_value + right_value) >> 1;
            UpdateCurVarVal(mid_value, idx);
            rec_locality = GetLocality(idx, update_direction);
            if (rec_locality <= Locality::LOCALVALID)
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

void GeneralSolver::UpdateMomentum(int32_t idx, double update_value, Locality cur_locality, Locality &best_locality)
{
    double cur_obj;
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

bool GeneralSolver::GetBestChoice()
{
    bool better_choice;
    bool make_sense;
    double cur_value = 0;
    bool has_chosen = false;
    double cur_momentum;
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

bool GeneralSolver::LocateLoc(const UpdateInfo *update_info, uint32_t &step, Locality &cur_locality, Locality &best_locality)
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

bool GeneralSolver::TryLocate(int32_t idx, double init_obj, Locality &best_locality)
{
    Locality cur_locality;
    uint32_t step = 0;
    UpdateDirection update_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);
    if (update_direction != UpdateDirection::NONE)
    {
        uint32_t neg_thres = var_info_->cur_vars[idx] - 1;
        uint32_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];
        uint32_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;
        UpdateInfo *update_info = new UpdateInfo(idx, thres, update_direction, init_obj);
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

TunePriority GeneralSolver::GetTunePriority(int32_t idx, double rec_obj, double &cur_obj)
{
    cur_obj = GetObj(var_info_->cur_vars);
    int32_t last_update = var_info_->rec_vars[idx] - var_info_->history_vars[idx];
    int32_t next_update = var_info_->cur_vars[idx] - var_info_->rec_vars[idx];
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

bool GeneralSolver::SearchLoc(const UpdateInfo *update_info, uint32_t &step, double &cur_obj, TunePriority &cur_priority)
{
    TunePriority rec_priority;
    int32_t idx = update_info->idx;
    uint32_t thres = update_info->thres;
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

bool GeneralSolver::GetHarmlessLoc(const UpdateInfo *update_info, uint32_t &step, double &cur_obj)
{
    double rec_obj;
    int32_t update_value;
    TunePriority rec_priority;
    int32_t idx = update_info->idx;
    uint32_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    var_info_->cur_vars[idx] = var_info_->rec_vars[idx];
    while (step < thres)
    {
        step = solver_config_.simple_ver ? (step << 1) : (step + 1);
        update_value = Bound(var_info_->upper_bound[idx], 1, var_info_->rec_vars[idx], step, update_direction);
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

bool GeneralSolver::GetDilatedLoc(const UpdateInfo *update_info, uint32_t &step)
{
    int32_t idx = update_info->idx;
    uint32_t update_value;
    uint32_t thres = update_info->thres;
    UpdateDirection update_direction = update_info->update_direction;
    double cur_obj;
    double cur_cons;
    double init_obj = update_info->init_obj;
    double init_cons = update_info->init_cons;
    double pre_cons = init_cons;
    while (step < thres)
    {
        step = solver_config_.simple_ver ? (step << 1) : (step + 1);
        update_value = Bound(var_info_->upper_bound[idx], 1, var_info_->rec_vars[idx], step, update_direction);
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

bool GeneralSolver::TuneLoc(const UpdateInfo *update_info, double cur_obj, uint32_t &step, TunePriority &cur_priority, TunePriority &best_priority)
{
    if (cur_priority <= best_priority)
    {
        uint32_t update_value;
        int32_t idx = update_info->idx;
        UpdateDirection update_direction = update_info->update_direction;
        double init_obj = update_info->init_obj;
        double init_cons = update_info->init_cons;
        if (cur_priority == TunePriority::HARMLESS)
        {
            GetHarmlessLoc(update_info, step, cur_obj);
        }
        else if (cur_priority == TunePriority::DILATED)
        {
            if (GetDescent(var_info_->cur_vars, idx, FuncInfo::BUFFER) != update_direction)
            {
                GetDilatedLoc(update_info, step);
            }
            else
            {
                cur_priority = solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;
            }
        }
        update_value = Bound(var_info_->upper_bound[idx], 1, var_info_->rec_vars[idx], step, update_direction);
        UpdateCurVarVal(update_value, idx);
        UpdateMomentum(idx, (init_obj - cur_obj), cur_priority, best_priority);
        return true;
    }
    return false;
}

bool GeneralSolver::TryTune(int32_t idx, UpdateDirection update_direction, double init_obj, double init_cons, TunePriority &best_priority)
{
    uint32_t step = 0;
    uint32_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];
    uint32_t neg_thres = var_info_->cur_vars[idx] - 1;
    uint32_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;
    double cur_obj;
    TunePriority cur_priority = (thres > 0) ? best_priority : TunePriority::REFUSE;
    if (thres > 0)
    {
        UpdateInfo *update_info = new UpdateInfo(idx, thres, update_direction, init_obj, init_cons);
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

bool GeneralSolver::FineTune()
{
    double init_obj = GetObj(var_info_->cur_vars);
    double init_cons = GetBuffCost(var_info_->cur_vars);
    RecordBestVarVal(init_obj);
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
        result_->AddVarVal(var_info_->cur_vars, obj, cons);
        return true;
    }
    return false;
}

bool GeneralSolver::Run(int32_t &solution_num, uint32_t *solutions)
{
    int32_t iter = 1;
    has_feasible_ = false;
    while (iter <= solver_config_.iterations)
    {
        Initialize(iter);
        if (solver_config_.get_log)
        {
            PrintCurVars(iter);
        }
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
            visited_node_->GetVarVal(var_info_->cur_vars)->searched = true;
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