/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "general_solver.h"
#include "generator/solver_pass/general_solver_code.h"

class ATT_TEST_GENERAL_SOLVER_GENERATOR_01 : public ::testing::Test
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
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }
};

TEST_F(ATT_TEST_GENERAL_SOLVER_GENERATOR_01, test_code_generator)
{
    // to check
    EXPECT_FALSE(att::GetGeneralSolver().empty());
}