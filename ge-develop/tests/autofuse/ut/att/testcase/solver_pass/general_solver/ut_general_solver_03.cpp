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

class UTTEST_GENERAL_SOLVER_03 : public ::testing::Test
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
        uint64_t val[1] = {3};
        varval_ = new VarVal(1, 0.0, 0.0, val);
    }

    void TearDown() override
    {
    }
    VarVal *varval_;
    Result *result_;
};

TEST_F(UTTEST_GENERAL_SOLVER_03, test_smax)
{
    EXPECT_EQ(SMAX(2, 10), 10);
    EXPECT_EQ(SMAX(2, 1), 2);
    EXPECT_EQ(SMAX(10, 2), 10);
    EXPECT_EQ(SMAX(1, 2), 2);
    EXPECT_EQ(SMAX(2, 2), 2);
}

TEST_F(UTTEST_GENERAL_SOLVER_03, test_smin)
{
    EXPECT_EQ(SMIN(2, 10), 2);
    EXPECT_EQ(SMIN(2, 1), 1);
    EXPECT_EQ(SMIN(10, 2), 2);
    EXPECT_EQ(SMIN(1, 2), 1);
    EXPECT_EQ(SMIN(2, 2), 2);
}

TEST_F(UTTEST_GENERAL_SOLVER_03, test_isequal)
{
    double a;
    double b;

    a = 0.15;
    b = 0.12;
    EXPECT_EQ(IsEqual(a, b), false);
    
    a = 15523;
    b = 15524;
    EXPECT_EQ(IsEqual(a, b), false);

    a = 0.401;
    b = 0.401;
    EXPECT_EQ(IsEqual(a, b), true);

    a = 65534;
    b = 65534;
    EXPECT_EQ(IsEqual(a, b), true);
}

TEST_F(UTTEST_GENERAL_SOLVER_03, test_get_value)
{
    EXPECT_EQ(GetValue(UpdateDirection::POSITIVE), 1);
    EXPECT_EQ(GetValue(UpdateDirection::NEGATIVE), -1);
    EXPECT_EQ(GetValue(UpdateDirection::NONE), 0);
}

TEST_F(UTTEST_GENERAL_SOLVER_03, test_bound)
{
    EXPECT_EQ(Bound(100, 1, 2, 3, UpdateDirection::POSITIVE), 5);
    EXPECT_EQ(Bound(100, 1, 2, 100, UpdateDirection::POSITIVE), 100);
    EXPECT_EQ(Bound(100, 1, 5, 2, UpdateDirection::NEGATIVE), 3);
    EXPECT_EQ(Bound(100, 1, 5, 10, UpdateDirection::NEGATIVE), 1);
    EXPECT_EQ(Bound(100, 10, 15, 10, UpdateDirection::NEGATIVE), 10);
}

TEST_F(UTTEST_GENERAL_SOLVER_03, test_get_varinfo)
{
    double obj;
    double cons;

    varval_ -> obj_ = 10;
    varval_ -> cons_ = 15;
    varval_ -> GetVarInfo(obj, cons);
    EXPECT_EQ(obj, 10);
    EXPECT_EQ(cons, 15);
    
    varval_ -> obj_ = 15;
    varval_ -> cons_ = 20;
    varval_ -> GetVarInfo(obj, cons);
    EXPECT_EQ(obj, 15);
    EXPECT_EQ(cons, 20);
}

TEST_F(UTTEST_GENERAL_SOLVER_03, test_set_get_vars)
{
    uint64_t vars[2];

    vars[0] = 5;
    varval_ = new VarVal(1, 10, 15, vars);
    varval_ -> GetVars(vars);
    EXPECT_EQ(vars[0], 5);
    EXPECT_EQ(varval_ -> var_num_, 1);
    EXPECT_EQ(varval_ -> obj_, 10);
    EXPECT_EQ(varval_ -> cons_, 15);

    vars[0] = 10;
    vars[1] = 20;
    varval_ = new VarVal(2, 15, 35, vars);
    varval_ -> GetVars(vars);
    EXPECT_EQ(vars[0], 10);
    EXPECT_EQ(vars[1], 20);
    EXPECT_EQ(varval_ -> var_num_, 2);
    EXPECT_EQ(varval_ -> obj_, 15);
    EXPECT_EQ(varval_ -> cons_, 35);
}