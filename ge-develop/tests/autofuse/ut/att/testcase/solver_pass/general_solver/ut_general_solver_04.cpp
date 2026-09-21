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

class TilingCaseUT4Solver : public GeneralSolver
{
public:
    TilingCaseUT4Solver(SolverConfig &config) : GeneralSolver(config) {}
    void DisplayVarVal(uint64_t vars[]) override{
        return;
    }
    double GetObj(uint64_t vars[]) override{
        return 0;
    }
    double GetSmoothObj(uint64_t vars[]) override{
        return 0;
    }
    double GetBuffCost(uint64_t vars[]) override{
        return 0;
    }
    double GetBuffDiff(uint64_t vars[], double* weight) override{
        return 0;
    }
    double GetLeqCost(uint64_t vars[]) override{
        return 0;
    }
    double GetLeqDiff(uint64_t vars[], double* weight) override{
        return 0;
    }
    bool CheckLocalValid(double leqs[], int32_t idx) override{
        return true;
    }
    void UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) override {
    double d_x = static_cast<double>(vars[0]);
    if (idx == -1 || idx == 0)
    {
        if (d_x >= left_edge && d_x <= right_edge) {
            leqs[0] = -1;
        } else {
            leqs[0] = 1;
        }
    }
    }
    int32_t left_edge = 1;
    int32_t right_edge = 5;
};

class UTTEST_GENERAL_SOLVER_04 : public ::testing::Test{
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
        solver_ = new TilingCaseUT4Solver(cfg);
        uint64_t init_vars[1] = {20};
        uint64_t upper_bound[1] = {20};
        uint64_t lower_bound[1] = {20};
        bool update_last[1] = {false};
        SolverInput input = {1, 1, upper_bound, lower_bound, init_vars, update_last};
        solver_->Init(input);
        solver_->Initialize(0);
    }
    void TearDown() override
    {
    }
    TilingCaseUT4Solver *solver_;
};

TEST_F(UTTEST_GENERAL_SOLVER_04, test_SetVarInfo){
    SolverInput input1;
    input1.var_num = 10;
    input1.cur_vars = new uint64_t(10);
    input1.upper_bound = new uint64_t(10);
    input1.lower_bound = new uint64_t(10);
    solver_ -> var_info_ = new VarInfo(input1);
    EXPECT_EQ(solver_ -> var_info_ -> var_num, 10);
}

TEST_F(UTTEST_GENERAL_SOLVER_04, test_SetConsInfo){
    bool res1, res2;
    solver_ -> cons_info_ = new ConsInfo(10);
    EXPECT_EQ(solver_ -> cons_info_ -> leq_num, 10);
}

TEST_F(UTTEST_GENERAL_SOLVER_04, test_SetResult){
    bool res1, res2;
    Result result = Result(10, 5);
    EXPECT_EQ(result.top_n_, 10);
    EXPECT_EQ(result.var_num_, 5);
}

TEST_F(UTTEST_GENERAL_SOLVER_04, test_SetSolverInput){
    SolverInput input;
    input.var_num = -10;
    EXPECT_EQ(solver_ -> SetSolverInput(input), false);
}

TEST_F(UTTEST_GENERAL_SOLVER_04, test_SearchVars){
    VisitedNode visited(5);
    solver_ -> visited_node_ = &visited;
    uint64_t vars[5] = {0,1,2,2,4};
    bool res = solver_ -> SearchVars(vars);
    EXPECT_EQ(res, false);
}

TEST_F(UTTEST_GENERAL_SOLVER_04, test_UpdateCurVarVal){
    bool res;
    res = solver_ -> UpdateCurVarVal(0, 3);
    EXPECT_EQ(res, false);
    res = solver_ -> UpdateCurVarVal(2, 0);
    EXPECT_EQ(res, true);
    EXPECT_EQ(solver_ -> var_info_ ->cur_vars[0], 2);
    // EXPECT_EQ(solver_ -> cons_info_ -> leqs[0], -1);
}