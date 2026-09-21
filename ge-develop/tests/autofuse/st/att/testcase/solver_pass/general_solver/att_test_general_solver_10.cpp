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
#include <chrono>
#include "solver.h"
#include "test_common_utils.h"

const int32_t all = -1;
const int32_t x = 0;
const int32_t y = 1;
const int32_t leq0 = 0;
const int32_t one = 1;

struct Params {
    uint64_t a;
};

class TilingCase10Solver : public GeneralSolver
{
public:
    TilingCase10Solver(SolverConfig& config, Params& params) : GeneralSolver(config), a(params.a) {}

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

double TilingCase10Solver::GetObj(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    double d_y = static_cast<double>(vars[y]);
    return d_x + d_y;
}

double TilingCase10Solver::GetSmoothObj(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    double d_y = static_cast<double>(vars[y]);
    return d_x + d_y;
}

double TilingCase10Solver::GetBuffCost(uint64_t vars[])
{
    return 0;
}

double TilingCase10Solver::GetBuffDiff(uint64_t vars[], double* weight)
{
    return 0;
}

double TilingCase10Solver::GetLeqCost(uint64_t vars[])
{
    double d_x = static_cast<double>(vars[x]);
    double d_y = static_cast<double>(vars[y]);
    return SMAX(d_x + d_y - one - a, 0.0) * SMAX(d_x + d_y - one - a, 0.0);
}

double TilingCase10Solver::GetLeqDiff(uint64_t vars[], double* weight)
{
    double d_x = static_cast<double>(vars[x]);
    double d_y = static_cast<double>(vars[y]);
    double leq_cost = d_x + d_y - one - a;
    leq_cost *= weight[0] > 0 ? weight[0] : 0;
    return leq_cost;
}

bool TilingCase10Solver::CheckLocalValid(double leqs[], int32_t idx)
{
    if (idx == x) {
        return leqs[leq0] <= 0;
    } else if (idx == y) {
        return leqs[leq0] <= 0;
    }
    return false;
}

void TilingCase10Solver::UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[])
{
    double d_x = static_cast<double>(vars[x]);
    double d_y = static_cast<double>(vars[y]);
    if (idx == all) {
        leqs[leq0] = d_x + d_y - one - a;
    } else if (idx == x) {
        leqs[leq0] = d_x + d_y - one - a;
    } else if (idx == y) {
        leqs[leq0] = d_x + d_y - one - a;
    }
}

void TilingCase10Solver::DisplayVarVal(uint64_t vars[])
{
    std::cout << "a0 = " << vars[0] << std::endl;
    std::cout << "a1 = " << vars[1] << std::endl;
}

class ATT_TEST_GENERAL_SOLVER_10 : public ::testing::Test
{
public:
    void TearDown() override
    {
        // 清理测试生成的临时文件
        autofuse::test::CleanupTestArtifacts();
        // before the destructor).
    }
};

void TestPerfTime(uint64_t a, bool simple_ver, double& used_time, double& pref)
{
    uint64_t top_num = 1;
    uint64_t search_length = 1;
    uint64_t iterations = 500;
    bool get_log = false;
    double momentum_factor = 0.9;
    
    int32_t num_var = 2;
    int32_t num_leq = 1;
    uint64_t init_vars[num_var] = {2 * a, 2 * a};
    uint64_t upper_bound[num_var] = {5 * a, 5 * a};
    uint64_t lower_bound[num_var] = {1, 1};
    bool update_last[num_var] = {false, false};

    int32_t solution_num = 0;
    uint64_t* solution = new uint64_t[num_var * top_num];

    timespec start, end;

    SolverInput input = {num_var, num_leq, upper_bound, lower_bound, init_vars, update_last};
    
    SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};
    
    Params params = {a};
    
    TilingCase10Solver* solver = new TilingCase10Solver(cfg, params);
    
    clock_gettime(CLOCK_REALTIME, &start);
    if (solver -> Init(input)) {
        if (solver -> Run(solution_num, solution)) {
            pref = solver -> GetObj(solution);
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    used_time = (double)end.tv_nsec - (double)start.tv_nsec;
    
    delete solver;
    delete[] solution;
}

/*
TEST_F(ATT_TEST_GENERAL_SOLVER_10, test_simple_time)
{
    double used_time_simple;
    double used_time_complex;
    double pref_simple;
    double pref_complex;

    TestPerfTime(200, true, used_time_simple, pref_simple);
    TestPerfTime(200, false, used_time_complex, pref_complex);
    EXPECT_LE(used_time_simple, used_time_complex);

    TestPerfTime(500, true, used_time_simple, pref_simple);
    TestPerfTime(500, false, used_time_complex, pref_complex);
    EXPECT_LE(used_time_simple, used_time_complex);

    TestPerfTime(750, true, used_time_simple, pref_simple);
    TestPerfTime(750, false, used_time_complex, pref_complex);
    EXPECT_LE(used_time_simple, used_time_complex);

    TestPerfTime(1000, true, used_time_simple, pref_simple);
    TestPerfTime(1000, false, used_time_complex, pref_complex);
    EXPECT_LE(used_time_simple, used_time_complex);
}
*/

TEST_F(ATT_TEST_GENERAL_SOLVER_10, test_simple_pref)
{
    double used_time_simple;
    double used_time_complex;
    double pref_simple;
    double pref_complex;

    TestPerfTime(20, true, used_time_simple, pref_simple);
    TestPerfTime(20, false, used_time_complex, pref_complex);
    EXPECT_GE(pref_simple, pref_complex);

    TestPerfTime(50, true, used_time_simple, pref_simple);
    TestPerfTime(50, false, used_time_complex, pref_complex);
    EXPECT_GE(pref_simple, pref_complex);

    TestPerfTime(75, true, used_time_simple, pref_simple);
    TestPerfTime(75, false, used_time_complex, pref_complex);
    EXPECT_GE(pref_simple, pref_complex);

    TestPerfTime(100, true, used_time_simple, pref_simple);
    TestPerfTime(100, false, used_time_complex, pref_complex);
    EXPECT_GE(pref_simple, pref_complex);
}