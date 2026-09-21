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
#include "gtest/gtest.h"
#include "solver.h"

class TilingCaseUT2Solver : public GeneralSolver
{
public:
    TilingCaseUT2Solver(SolverConfig &config) : GeneralSolver(config) {}
    void DisplayVarVal(uint64_t vars[]) override;
    double GetObj(uint64_t vars[]) override;
    double GetSmoothObj(uint64_t vars[]) override;
    double GetBuffCost(uint64_t vars[]) override;
    double GetLeqCost(uint64_t vars[]) override;
    bool CheckLocalValid(double leqs[], int32_t idx) override;
    void UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) override;
    double GetBuffDiff(uint64_t* vars, double* weight) override;
    double GetLeqDiff(uint64_t* vars, double* weight) override;
    int32_t thres{10};
};

double TilingCaseUT2Solver::GetObj(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[0]);
    double d_y = static_cast<double>(vars[1]);
    return - d_x;
}

double TilingCaseUT2Solver::GetSmoothObj(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[0]);
    double d_y = static_cast<double>(vars[1]);
    return - d_x;
}

double TilingCaseUT2Solver::GetBuffCost(uint64_t vars[])
{
    double d_y = static_cast<double>(vars[0]);
    return 20 - d_y;
}

double TilingCaseUT2Solver::GetBuffDiff(uint64_t vars[], double* weight)
{
    double d_y = static_cast<double>(vars[0]);
    double leq_cost = 20 - d_y;
    leq_cost *= weight[0] < 0 ? weight[0] : 0;
    return leq_cost;
}

double TilingCaseUT2Solver::GetLeqCost(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[0]);
    double d_y = static_cast<double>(vars[1]);
    return SMAX(d_x - thres, 0.0) * SMAX(d_x - thres, 0.0) + SMAX(d_y - thres, 0.0) * SMAX(d_y - thres, 0.0);
}

double TilingCaseUT2Solver::GetLeqDiff(uint64_t vars[], double* weight)
{
    double d_x = static_cast<double>(vars[0]);
    double d_y = static_cast<double>(vars[1]);
    double leq_cost1 = d_x - thres;
    leq_cost1 *= weight[0] > 0 ? weight[0] : 0;
    double leq_cost2 = d_y - thres;
    leq_cost2 *= weight[1] > 0 ? weight[1] : 0;
    return leq_cost1 + leq_cost2;
}

bool TilingCaseUT2Solver::CheckLocalValid(double leqs[], int32_t idx)
{
    if (idx == 0)
    {
        return leqs[0] <= 0;
    }
    else if (idx == 1)
    {
        return leqs[1] <= 0;
    }
    return false;
}

void TilingCaseUT2Solver::UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[])
{
    double d_x = static_cast<double>(vars[0]);
    double d_y = static_cast<double>(vars[1]);
    if (idx == -1)
    {
        leqs[0] = d_x - thres;
        leqs[1] = d_y - thres;
    }
    else if (idx == 0)
    {
        leqs[0] = d_x - thres;
    }
    else if (idx == 1)
    {
        leqs[1] = d_y - thres;
    }
}

void TilingCaseUT2Solver::DisplayVarVal(uint64_t vars[])
{
    return;
}

class UTTEST_GENERAL_SOLVER_02 : public ::testing::Test
{
public:
    static void TearDownTestCase()
    {
        std::cout << "Test end." << std::endl;
    }
    static void SetUpTestCase()
    {
        std::cout << "Test begin." << std::endl;
    }
    void SetUp() override
    {
        uint64_t top_num = 1;
        uint64_t search_length = 1;
        uint64_t iterations = 500;
        bool simple_ver = false;
        bool get_log = false;
        double momentum_factor = 0.9;

        SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};
        
        solver_ = new TilingCaseUT2Solver(cfg);
        
        uint64_t init_vars[2] = {20, 20};
        uint64_t upper_bound[2] = {20, 20};
        uint64_t lower_bound[2] = {1, 1};
        bool update_last[2] = {false, false};
        SolverInput input = {2, 2, upper_bound, lower_bound, init_vars, update_last};
        solver_->Init(input);
        solver_->Initialize(1);
    }

    void TearDown() override
    {
    }
    bool TestLocateRegion(std::vector<uint64_t> vars, std::vector<bool> update_last, int32_t thres, std::vector<uint64_t>& res);
    bool TestFineTune(bool simple_ver, std::vector<uint64_t> vars, std::vector<uint64_t>& res);
    bool TestInit(int32_t var_num, int32_t leq_num, int32_t top_num);
    bool TestRun(int32_t thres, int32_t& solution_num);

    TilingCaseUT2Solver *solver_;
};

bool UTTEST_GENERAL_SOLVER_02::TestLocateRegion(std::vector<uint64_t> vars, std::vector<bool> update_last, int32_t thres, std::vector<uint64_t>& res)
{
    bool ret;
    solver_ -> var_info_ -> cur_vars[0] = vars[0];
    solver_ -> var_info_ -> cur_vars[1] = vars[1];
    solver_ -> var_info_ -> update_last[0] = update_last[0];
    solver_ -> var_info_ -> update_last[1] = update_last[1];
    solver_ -> thres = thres;
    solver_ -> Initialize(2);
    
    ret = solver_ -> LocateRegion();
    res.clear();
    res.push_back(solver_ -> var_info_ -> cur_vars[0]);
    res.push_back(solver_ -> var_info_ -> cur_vars[1]);
    return ret;
}

bool UTTEST_GENERAL_SOLVER_02::TestFineTune(bool simple_ver, std::vector<uint64_t> vars, std::vector<uint64_t>& res)
{
    bool ret;
    solver_ -> var_info_ -> cur_vars[0] = vars[0];
    solver_ -> var_info_ -> cur_vars[1] = vars[1];
    solver_ -> thres = 10;
    solver_ -> Initialize(1);
    ret = solver_ -> FineTune();
    solver_ -> solver_config_.simple_ver = simple_ver;
    res.clear();
    res.push_back(solver_ -> var_info_ -> cur_vars[0]);
    res.push_back(solver_ -> var_info_ -> cur_vars[1]);
    return ret;
}

bool UTTEST_GENERAL_SOLVER_02::TestInit(int32_t var_num, int32_t leq_num, int32_t top_num) {
    uint64_t upper_bound[1] = {20};
    uint64_t lower_bound[1] = {1};
    uint64_t init_vars[1] = {20};
    bool update_last[1] = {false};
    solver_ -> solver_config_.top_num = top_num;
    SolverInput input = {var_num, leq_num, upper_bound, lower_bound, init_vars, update_last};
    return solver_ -> Init(input);    
}

bool UTTEST_GENERAL_SOLVER_02::TestRun(int32_t thres, int32_t& solution_num) {
    uint64_t init_vars[2] = {20, 20};
    uint64_t upper_bound[2] = {20, 20};
    uint64_t lower_bound[2] = {1, 1};
    bool update_last[2] = {false, false};
    SolverInput input = {2, 2, upper_bound, lower_bound, init_vars, update_last};
    solver_->Init(input);
    solver_ -> thres = thres;
    solver_ -> Initialize(1);
    uint64_t* solution = new uint64_t[2 * 1];
    bool ret = solver_ -> Run(solution_num, solution);
    return ret;    
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_locate_region)
{
    bool ret;
    std::vector<bool> update_last;
    std::vector<uint64_t> vars;
    std::vector<uint64_t> res;
    std::vector<uint64_t> expect_res;
    
    vars = {20, 20};
    expect_res = {10, 20};
    update_last = {false, true};
    ret = TestLocateRegion(vars, update_last, 10, res);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(res, expect_res);

    vars = {20, 20};
    expect_res = {20, 10};
    update_last = {true, false};
    ret = TestLocateRegion(vars, update_last, 10, res);
    EXPECT_EQ(res, expect_res);

    solver_ -> has_feasible_ = true;
    vars = {20, 20};
    expect_res = {20, 10};
    update_last = {false, true};
    ret = TestLocateRegion(vars, update_last, 10, res);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(res, expect_res);

    expect_res = {20, 10};
    vars = {1, 1};
    ret = TestLocateRegion(vars, update_last, -1, res);
    EXPECT_EQ(ret, false);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_fine_tune)
{
    bool ret;
    std::vector<uint64_t> vars;
    std::vector<uint64_t> res;
    std::vector<uint64_t> expect_res;
    
    vars = {2, 2};
    expect_res = {10, 2};
    ret = TestFineTune(false, vars, res);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(res, expect_res);
    
    vars = {10, 2};
    expect_res = {10, 10};
    ret = TestFineTune(false, vars, res);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(res, expect_res);

    vars = {1, 1};
    solver_ -> var_info_ -> upper_bound[0] = 1;
    solver_ -> var_info_ -> upper_bound[1] = 1;
    ret = TestFineTune(false, vars, res);
    EXPECT_EQ(ret, false);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_init)
{
    bool ret;
    ret = TestInit(2, 2, 1);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(solver_ -> var_info_ -> var_num, 2);
    EXPECT_EQ(solver_ -> cons_info_ -> leq_num, 2);
    EXPECT_EQ(solver_ -> result_ -> top_n_, 1);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_run)
{
    bool ret;
    int solution_num;
    
    ret = TestRun(10, solution_num);
    EXPECT_EQ(solution_num, 1);
    EXPECT_EQ(ret, true);
    
    ret = TestRun(1, solution_num);
    EXPECT_EQ(solution_num, 1);
    EXPECT_EQ(ret, true);
    
    ret = TestRun(-1, solution_num);
    EXPECT_EQ(solution_num, 0);
    EXPECT_EQ(ret, false);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_get_varnum)
{
    EXPECT_EQ(solver_ -> GetVarNum(), 2);
    
    solver_ -> var_info_ -> var_num = 3;
    EXPECT_EQ(solver_ -> GetVarNum(), 3);
    
    solver_ -> var_info_ -> var_num = 5;
    EXPECT_EQ(solver_ -> GetVarNum(), 5);

    solver_ -> var_info_ -> var_num = 11;
    EXPECT_EQ(solver_ -> GetVarNum(), 11);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_check_valid)
{
    solver_ -> cons_info_ -> leqs[0] = -1;
    solver_ -> cons_info_ -> leqs[1] = -2;
    EXPECT_EQ(solver_ -> CheckValid(), true);
    
    solver_ -> cons_info_ -> leqs[0] = 1;
    solver_ -> cons_info_ -> leqs[1] = -2;
    EXPECT_EQ(solver_ -> CheckValid(), false);
    
    solver_ -> cons_info_ -> leqs[0] = -1;
    solver_ -> cons_info_ -> leqs[1] = 2;
    EXPECT_EQ(solver_ -> CheckValid(), false);
    
    solver_ -> cons_info_ -> leqs[0] = 1;
    solver_ -> cons_info_ -> leqs[1] = 2;
    EXPECT_EQ(solver_ -> CheckValid(), false);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_reset_momentum)
{
    solver_ -> momentum_info_->is_valid[0] = true;
    solver_ -> momentum_info_->is_valid[1] = true;
    solver_ -> ResetMomentum();
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], false);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], false);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_update_momentum)
{
    Locality best_locality;

    best_locality = Locality::REJECT;
    solver_ -> UpdateMomentum(1, 10, Locality::GLOBALVALID, best_locality);
    EXPECT_EQ(best_locality, Locality::GLOBALVALID);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], false);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], true);
    EXPECT_EQ(solver_ -> momentum_info_->cur_value[1], 10);

    solver_ -> UpdateMomentum(1, 10, Locality::LOCALVALID, best_locality);
    EXPECT_EQ(best_locality, Locality::GLOBALVALID);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], false);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], true);
    EXPECT_EQ(solver_ -> momentum_info_->cur_value[1], 10);

    solver_ -> UpdateMomentum(0, 12, Locality::GLOBALVALID, best_locality);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], true);
    EXPECT_EQ(solver_ -> momentum_info_->cur_value[0], 12);

    TunePriority best_priority;
    best_priority = TunePriority::REFUSE;
    solver_ -> UpdateMomentum(0, 23, TunePriority::DILATED, best_priority);
    EXPECT_EQ(best_priority, TunePriority::DILATED);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], true);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], false);
    EXPECT_EQ(solver_ -> momentum_info_->cur_value[0], 23);

    solver_ -> UpdateMomentum(1, -10, TunePriority::DILATED, best_priority);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], true);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], true);
    EXPECT_EQ(solver_ -> momentum_info_->cur_value[1], -10);

    solver_ -> UpdateMomentum(1, 12, TunePriority::HARMLESS, best_priority);
    EXPECT_EQ(best_priority, TunePriority::HARMLESS);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], false);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], true);
    EXPECT_EQ(solver_ -> momentum_info_->cur_value[1], 12);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_get_best_choice)
{
    bool ret;

    solver_ -> momentum_info_->is_valid[0] = true;
    solver_ -> momentum_info_->is_valid[1] = true;
    solver_ -> momentum_info_->momentum[0] = 0;
    solver_ -> momentum_info_->cur_value[0] = 10;
    solver_ -> momentum_info_->momentum[1] = 0;
    solver_ -> momentum_info_->cur_value[1] = 11;
    ret = solver_ -> GetBestChoice();
    EXPECT_EQ(ret, true);
    EXPECT_EQ(solver_ -> var_info_ -> chosen_var_idx, 1);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[0], 1), true);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[1], 1.1), true);

    solver_ -> momentum_info_->is_valid[0] = true;
    solver_ -> momentum_info_->is_valid[1] = false;
    solver_ -> momentum_info_->momentum[0] = 0;
    solver_ -> momentum_info_->cur_value[0] = 10;
    solver_ -> momentum_info_->momentum[1] = 0;
    solver_ -> momentum_info_->cur_value[1] = 11;
    ret = solver_ -> GetBestChoice();
    EXPECT_EQ(ret, true);
    EXPECT_EQ(solver_ -> var_info_ -> chosen_var_idx, 0);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[0], 1), true);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[1], 0), true);


    solver_ -> momentum_info_->is_valid[0] = true;
    solver_ -> momentum_info_->is_valid[1] = true;
    solver_ -> momentum_info_->momentum[0] = 20;
    solver_ -> momentum_info_->cur_value[0] = 10;
    solver_ -> momentum_info_->momentum[1] = 0;
    solver_ -> momentum_info_->cur_value[1] = 11;
    ret = solver_ -> GetBestChoice();
    EXPECT_EQ(ret, true);
    EXPECT_EQ(solver_ -> var_info_ -> chosen_var_idx, 0);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[0], 19), true);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[1], 1.1), true);

    solver_ -> var_info_ -> chosen_var_idx = -1;
    solver_ -> momentum_info_->is_valid[0] = false;
    solver_ -> momentum_info_->is_valid[1] = false;
    solver_ -> momentum_info_->momentum[0] = 20;
    solver_ -> momentum_info_->cur_value[0] = 10;
    solver_ -> momentum_info_->momentum[1] = 0;
    solver_ -> momentum_info_->cur_value[1] = 11;
    ret = solver_ -> GetBestChoice();
    EXPECT_EQ(ret, false);
}

TEST_F(UTTEST_GENERAL_SOLVER_02, test_update_best_var)
{
    solver_ -> thres = 10;
    solver_ -> var_info_ -> cur_vars[0] = 1;
    solver_ -> var_info_ -> cur_vars[1] = 1;
    solver_ -> var_info_ -> target_val[0] = 3;
    solver_ -> var_info_ -> target_val[1] = 4;
    solver_ -> momentum_info_->is_valid[0] = true;
    solver_ -> momentum_info_->is_valid[1] = true;
    solver_ -> momentum_info_->momentum[0] = 1;
    solver_ -> momentum_info_->momentum[1] = 1.1;
    solver_ -> var_info_ -> chosen_var_idx = 1;
    solver_ -> UpdateBestVar();
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[0], false);
    EXPECT_EQ(solver_ -> momentum_info_->is_valid[1], false);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[0], 0), true);
    EXPECT_EQ(IsEqual(solver_ -> momentum_info_->momentum[1], 1.1), true);
    EXPECT_EQ(solver_ -> var_info_ -> cur_vars[0], 1);
    EXPECT_EQ(solver_ -> var_info_ -> cur_vars[1], 4);
    EXPECT_EQ(solver_ -> cons_info_ -> leqs[0], -9);
    EXPECT_EQ(solver_ -> cons_info_ -> leqs[1], -6);
}