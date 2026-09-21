/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#ifndef private
#define private public
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_prof.h"
#include "acl/acl_tdt.h"
#include "acl/acl_tdt_queue.h"
#include "acl/acl_mdl.h"
#include "acl/ops/acl_cblas.h"
#include "runtime/base.h"
#undef private
#endif

class UTEST_ACL_compatibility_enum_check : public testing::Test
{
    public:
        UTEST_ACL_compatibility_enum_check() {}
    protected:
        virtual void SetUp() {}
        virtual void TearDown() {}
};

TEST_F(UTEST_ACL_compatibility_enum_check, aclmdlConfigAttr)
{
    aclmdlConfigAttr configAttr;
    configAttr = (aclmdlConfigAttr)0;
    EXPECT_EQ(configAttr, ACL_MDL_PRIORITY_INT32);

    configAttr = (aclmdlConfigAttr)1;
    EXPECT_EQ(configAttr, ACL_MDL_LOAD_TYPE_SIZET);

    configAttr = (aclmdlConfigAttr)2;
    EXPECT_EQ(configAttr, ACL_MDL_PATH_PTR);

    configAttr = (aclmdlConfigAttr)3;
    EXPECT_EQ(configAttr, ACL_MDL_MEM_ADDR_PTR);

    configAttr = (aclmdlConfigAttr)4;
    EXPECT_EQ(configAttr, ACL_MDL_MEM_SIZET);

    configAttr = (aclmdlConfigAttr)5;
    EXPECT_EQ(configAttr, ACL_MDL_WEIGHT_ADDR_PTR);

    configAttr = (aclmdlConfigAttr)6;
    EXPECT_EQ(configAttr, ACL_MDL_WEIGHT_SIZET);

    configAttr = (aclmdlConfigAttr)7;
    EXPECT_EQ(configAttr, ACL_MDL_WORKSPACE_ADDR_PTR);

    configAttr = (aclmdlConfigAttr)8;
    EXPECT_EQ(configAttr, ACL_MDL_WORKSPACE_SIZET);

    configAttr = (aclmdlConfigAttr)9;
    EXPECT_EQ(configAttr, ACL_MDL_INPUTQ_NUM_SIZET);

    configAttr = (aclmdlConfigAttr)10;
    EXPECT_EQ(configAttr, ACL_MDL_INPUTQ_ADDR_PTR);

    configAttr = (aclmdlConfigAttr)11;
    EXPECT_EQ(configAttr, ACL_MDL_OUTPUTQ_NUM_SIZET);

    configAttr = (aclmdlConfigAttr)12;
    EXPECT_EQ(configAttr, ACL_MDL_OUTPUTQ_ADDR_PTR);

    configAttr = (aclmdlConfigAttr)13;
    EXPECT_EQ(configAttr, ACL_MDL_WORKSPACE_MEM_OPTIMIZE);

    configAttr = (aclmdlConfigAttr)14;
    EXPECT_EQ(configAttr, ACL_MDL_WEIGHT_PATH_PTR);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclmdlInputAippType)
{
    aclmdlInputAippType inputAippType;
    inputAippType = (aclmdlInputAippType)0;
    EXPECT_EQ(inputAippType, ACL_DATA_WITHOUT_AIPP);

    inputAippType = (aclmdlInputAippType)1;
    EXPECT_EQ(inputAippType, ACL_DATA_WITH_STATIC_AIPP);

    inputAippType = (aclmdlInputAippType)2;
    EXPECT_EQ(inputAippType, ACL_DATA_WITH_DYNAMIC_AIPP);

    inputAippType = (aclmdlInputAippType)3;
    EXPECT_EQ(inputAippType, ACL_DYNAMIC_AIPP_NODE);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclCompileType)
{
    aclopCompileType compileType;
    compileType = (aclopCompileType)0;
    EXPECT_EQ(compileType, ACL_COMPILE_SYS);

    compileType = (aclopCompileType)1;
    EXPECT_EQ(compileType, ACL_COMPILE_UNREGISTERED);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclCompileOpt)
{
    aclCompileOpt compileOpt;
    compileOpt = (aclCompileOpt)0;
    EXPECT_EQ(compileOpt, ACL_PRECISION_MODE);

    compileOpt = (aclCompileOpt)1;
    EXPECT_EQ(compileOpt, ACL_AICORE_NUM);

    compileOpt = (aclCompileOpt)2;
    EXPECT_EQ(compileOpt, ACL_AUTO_TUNE_MODE);

    compileOpt = (aclCompileOpt)3;
    EXPECT_EQ(compileOpt, ACL_OP_SELECT_IMPL_MODE);

    compileOpt = (aclCompileOpt)4;
    EXPECT_EQ(compileOpt, ACL_OPTYPELIST_FOR_IMPLMODE);

    compileOpt = (aclCompileOpt)5;
    EXPECT_EQ(compileOpt, ACL_OP_DEBUG_LEVEL);

    compileOpt = (aclCompileOpt)6;
    EXPECT_EQ(compileOpt, ACL_DEBUG_DIR);

    compileOpt = (aclCompileOpt)7;
    EXPECT_EQ(compileOpt, ACL_OP_COMPILER_CACHE_MODE);

    compileOpt = (aclCompileOpt)8;
    EXPECT_EQ(compileOpt, ACL_OP_COMPILER_CACHE_DIR);

    compileOpt = (aclCompileOpt)9;
    EXPECT_EQ(compileOpt, ACL_OP_PERFORMANCE_MODE);

    compileOpt = (aclCompileOpt)10;
    EXPECT_EQ(compileOpt, ACL_OP_JIT_COMPILE);

    compileOpt = (aclCompileOpt)11;
    EXPECT_EQ(compileOpt, ACL_OP_DETERMINISTIC);

    compileOpt = (aclCompileOpt)12;
    EXPECT_EQ(compileOpt, ACL_CUSTOMIZE_DTYPES);

    compileOpt = (aclCompileOpt)13;
    EXPECT_EQ(compileOpt, ACL_OP_PRECISION_MODE);

    compileOpt = (aclCompileOpt)14;
    EXPECT_EQ(compileOpt, ACL_ALLOW_HF32);

    compileOpt = (aclCompileOpt)15;
    EXPECT_EQ(compileOpt, ACL_PRECISION_MODE_V2);

    compileOpt = (aclCompileOpt)16;
    EXPECT_EQ(compileOpt, ACL_OP_DEBUG_OPTION);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclCompileFlag)
{
    aclOpCompileFlag compileFlag;
    compileFlag = (aclCompileFlag)0;
    EXPECT_EQ(compileFlag, ACL_OP_COMPILE_DEFAULT);

    compileFlag = (aclCompileFlag)1;
    EXPECT_EQ(compileFlag, ACL_OP_COMPILE_FUZZ);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclopEngineType)
{
    aclopEngineType engineType;
    engineType = (aclopEngineType)0;
    EXPECT_EQ(engineType, ACL_ENGINE_SYS);

    engineType = (aclopEngineType)1;
    EXPECT_EQ(engineType, ACL_ENGINE_AICORE);

    engineType = (aclopEngineType)2;
    EXPECT_EQ(engineType, ACL_ENGINE_VECTOR);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclTransType)
{
    aclTransType transType;
    transType = (aclTransType)0;
    EXPECT_EQ(transType, ACL_TRANS_N);

    transType = (aclTransType)1;
    EXPECT_EQ(transType, ACL_TRANS_T);

    transType = (aclTransType)2;
    EXPECT_EQ(transType, ACL_TRANS_NZ);

    transType = (aclTransType)3;
    EXPECT_EQ(transType, ACL_TRANS_NZ_T);
}

TEST_F(UTEST_ACL_compatibility_enum_check, aclComputeType)
{
    aclComputeType computeType;
    computeType = (aclComputeType)0;
    EXPECT_EQ(computeType, ACL_COMPUTE_HIGH_PRECISION);

    computeType = (aclComputeType)1;
    EXPECT_EQ(computeType, ACL_COMPUTE_LOW_PRECISION);
}
