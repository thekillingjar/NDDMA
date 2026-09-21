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
const int32_t cycle = 2;
const int32_t one = 1;
const int32_t three = 3;
const int32_t ten = 10;

struct Params
{
    uint64_t a;
};

class TilingCase8Solver : public GeneralSolver
{
public:
    TilingCase8Solver(SolverConfig &config, Params &params) : GeneralSolver(config), a(params.a) {}

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

double TilingCase8Solver::GetObj(uint64_t vars[])
{
    if (vars[x] % cycle == 0)
    {
        return three;
    }
    else
    {
        return one;
    }
}

double TilingCase8Solver::GetSmoothObj(uint64_t vars[])
{
    if (vars[x] % cycle == 0)
    {
        return three;
    }
    else
    {
        return one;
    }
}

double TilingCase8Solver::GetBuffCost(uint64_t vars[])
{
    return 0;
}

double TilingCase8Solver::GetBuffDiff(uint64_t vars[], double* weight)
{
    return 0;
}

double TilingCase8Solver::GetLeqCost(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    return SMAX(d_x - ten, 0.0) * SMAX(d_x - ten, 0.0);
}

double TilingCase8Solver::GetLeqDiff(uint64_t vars[], double* weight)
{
    double d_x = static_cast<double>(vars[x]);
    double leq_cost = d_x - ten;
    leq_cost *= weight[0] > 0 ? weight[0] : 0;
    return leq_cost;
}

bool TilingCase8Solver::CheckLocalValid(double leqs[], int32_t idx)
{
    if (idx == x)
    {
        return leqs[leq0] <= 0;
    }
    return false;
}

void TilingCase8Solver::UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[])
{
    double d_x = static_cast<double>(vars[x]);
    if (idx == all)
    {
        leqs[leq0] = d_x - ten;
    }
    else if (idx == x)
    {
        leqs[leq0] = d_x - ten;
    }
}

void TilingCase8Solver::DisplayVarVal(uint64_t vars[])
{
    std::cout << "a0 = " << vars[0] << std::endl;
}

class ATT_TEST_GENERAL_SOLVER_08 : public ::testing::Test
{
public:
    void TearDown() override
    {
        // 清理测试生成的临时文件
        autofuse::test::CleanupTestArtifacts();
        // before the destructor).
    }
};

std::vector<std::vector<uint64_t>> TestTopNum(uint64_t top_num, int32_t &solution_num)
{
    uint64_t a = 2;
    uint64_t search_length = 5;
    uint64_t iterations = 500;
    bool simple_ver = false;
    bool get_log = false;
    double momentum_factor = 0.9;

    int32_t num_var = 1;
    int32_t num_leq = 1;
    uint64_t init_vars[num_var] = {15};
    uint64_t upper_bound[num_var] = {15};
    uint64_t lower_bound[num_var] = {1};
    bool update_last[num_var] = {false};

    std::vector<std::vector<uint64_t>> result;
    uint64_t *solution = new uint64_t[num_var * top_num];

    SolverInput input = {num_var, num_leq, upper_bound, lower_bound, init_vars, update_last};
    SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};
    Params params = {a};

    TilingCase8Solver *solver = new TilingCase8Solver(cfg, params);

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

TEST_F(ATT_TEST_GENERAL_SOLVER_08, test_topnum)
{
    int solution_num;

    TestTopNum(1, solution_num);
    EXPECT_EQ(solution_num, 1);

    TestTopNum(2, solution_num);
    EXPECT_EQ(solution_num, 2);

    TestTopNum(3, solution_num);
    EXPECT_EQ(solution_num, 3);

    TestTopNum(4, solution_num);
    EXPECT_EQ(solution_num, 4);

    TestTopNum(5, solution_num);
    EXPECT_EQ(solution_num, 5);
}

TEST_F(ATT_TEST_GENERAL_SOLVER_08, test_topnum_solution)
{
    int solution_num;
    std::vector<std::vector<uint64_t>> expect_result;

    expect_result = {{9},{7},{5},{3},{1}};
    EXPECT_EQ(TestTopNum(5, solution_num), expect_result);
}