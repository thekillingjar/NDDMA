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
#include "solver.h"
#include "test_common_utils.h"

const int32_t all = -1;
const int32_t x = 0;
const int32_t leq0 = 0;
const int32_t two = 2;
const int32_t four = 4;
const int32_t ten = 10;

struct Params
{
    uint64_t a;
};

class TilingCase7Solver : public GeneralSolver
{
public:
    TilingCase7Solver(SolverConfig &config, Params &params) : GeneralSolver(config), a(params.a) {}

    void DisplayVarVal(uint64_t vars[]) override;
    double GetObj(uint64_t vars[]) override;
    double GetSmoothObj(uint64_t vars[]) override;
    double GetBuffCost(uint64_t vars[]) override;
    double GetLeqCost(uint64_t vars[]) override;
    bool CheckLocalValid(double leqs[], int32_t idx) override;
    void UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) override;
    double GetBuffDiff(uint64_t* vars, double* weight) override;
    double GetLeqDiff(uint64_t* vars, double* weight) override;

    std::vector<std::vector<uint64_t>> GetResult(int32_t solution_num, uint64_t *solution)
    {
        int32_t idx;
        std::vector<uint64_t> res;
        std::vector<std::vector<uint64_t>> result;
        for (int32_t i = 0; i < solution_num; i++)
        {
            res.clear();
            for (int32_t j = 0; j < GetVarNum(); j++)
            {
                res.emplace_back(*(solution + i * GetVarNum() + j));
            }
            result.emplace_back(res);
        }
        return result;
    }

private:
    uint64_t a;
};

double TilingCase7Solver::GetObj(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    if (vars[x] <= a + two)
    {
        return d_x;
    }
    else if (vars[x] <= two * a + two)
    {
        return -d_x + two * a + four;
    }
    else
    {
        return d_x - two * a;
    }
}

double TilingCase7Solver::GetSmoothObj(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    if (vars[x] <= a + two)
    {
        return d_x;
    }
    else if (vars[x] <= two * a + two)
    {
        return -d_x + two * a + four;
    }
    else
    {
        return d_x - two * a;
    }
}

double TilingCase7Solver::GetBuffCost(uint64_t vars[])
{
    return 0;
}

double TilingCase7Solver::GetBuffDiff(uint64_t vars[], double* weight)
{
    return 0;
}

double TilingCase7Solver::GetLeqCost(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    return SMAX(d_x - ten, 0.0) * SMAX(d_x - ten, 0.0);
}

double TilingCase7Solver::GetLeqDiff(uint64_t vars[], double* weight)
{
    double d_x = static_cast<double>(vars[x]);
    double leq_cost = d_x - ten;
    leq_cost *= weight[0] > 0 ? weight[0] : 0;
    return leq_cost;
}

bool TilingCase7Solver::CheckLocalValid(double leqs[], int32_t idx)
{
    if (idx == x)
    {
        return leqs[leq0] <= 0;
    }
    return false;
}

void TilingCase7Solver::UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[])
{
    double d_x = static_cast<double>(vars[x]);
    if (idx == all)
    {
        leqs[leq0] = d_x - ten;
    }
    else if (idx == 0)
    {
        leqs[leq0] = d_x - ten;
    }
}

void TilingCase7Solver::DisplayVarVal(uint64_t vars[])
{
    std::cout << "a0 = " << vars[0] << std::endl;
}

class ATT_TEST_GENERAL_SOLVER_07 : public ::testing::Test
{
public:
    void TearDown() override
    {
        // 清理测试生成的临时文件
        autofuse::test::CleanupTestArtifacts();
        // before the destructor).
    }
};

std::vector<std::vector<uint64_t>> TestSearchLength(uint64_t a, uint64_t search_length)
{
    uint64_t top_num = 1;
    uint64_t iterations = 500;
    bool simple_ver = true;
    bool get_log = false;
    double momentum_factor = 0.9;

    int32_t num_var = 1;
    int32_t num_leq = 1;
    uint64_t init_vars[num_var] = {15};
    uint64_t upper_bound[num_var] = {15};
    uint64_t lower_bound[num_var] = {1};
    bool update_last[num_var] = {false};

    std::vector<std::vector<uint64_t>> result;
    int32_t solution_num = 0;
    uint64_t *solution = new uint64_t[num_var * top_num];

    SolverInput input = {num_var, num_leq, upper_bound, lower_bound, init_vars, update_last};
    SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};

    Params params = {a};

    TilingCase7Solver *solver = new TilingCase7Solver(cfg, params);

    if (solver->Init(input))
    {
        if (solver->Run(solution_num, solution))
        {
            result = solver->GetResult(solution_num, solution);
        }
    }

    delete solver;
    delete[] solution;
    return result;
}

TEST_F(ATT_TEST_GENERAL_SOLVER_07, test_search_length)
{
    std::vector<std::vector<uint64_t>> expect_result1;
    std::vector<std::vector<uint64_t>> expect_result2;

    expect_result1 = {{1}};
    expect_result2 = {{6}};
    EXPECT_EQ(TestSearchLength(2, 5), expect_result1);
    EXPECT_EQ(TestSearchLength(2, 1), expect_result2);
}