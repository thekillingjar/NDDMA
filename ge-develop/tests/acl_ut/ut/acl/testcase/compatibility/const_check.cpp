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
#include "acl/error_codes/ge_error_codes.h"
#include "acl/error_codes/rt_error_codes.h"
#include "acl/acl_mdl.h"
#undef private
#endif

class UTEST_ACL_compatibility_const_check : public testing::Test
{
    public:
        UTEST_ACL_compatibility_const_check() {}
    protected:
        virtual void SetUp() {}
        virtual void TearDown() {}
};

TEST_F(UTEST_ACL_compatibility_const_check, acl_mdl)
{
    EXPECT_EQ(ACL_DIM_ENDPOINTS, 2);
    EXPECT_EQ(ACL_MAX_DIM_CNT, 128);
    EXPECT_EQ(ACL_MAX_TENSOR_NAME_LEN, 128);
    EXPECT_EQ(ACL_MAX_BATCH_NUM, 128);
    EXPECT_EQ(ACL_MAX_HW_NUM, 128);
    EXPECT_EQ(ACL_MAX_SHAPE_COUNT, 128);
    EXPECT_EQ(ACL_INVALID_NODE_INDEX,  0xFFFFFFFF);

    EXPECT_EQ(ACL_MDL_LOAD_FROM_FILE, 1);
    EXPECT_EQ(ACL_MDL_LOAD_FROM_FILE_WITH_MEM, 2);
    EXPECT_EQ(ACL_MDL_LOAD_FROM_MEM, 3);
    EXPECT_EQ(ACL_MDL_LOAD_FROM_MEM_WITH_MEM, 4);
    EXPECT_EQ(ACL_MDL_LOAD_FROM_FILE_WITH_Q, 5);
    EXPECT_EQ(ACL_MDL_LOAD_FROM_MEM_WITH_Q, 6);

    EXPECT_EQ(std::string(ACL_DYNAMIC_TENSOR_NAME), std::string("ascend_mbatch_shape_data"));
    EXPECT_EQ(std::string(ACL_DYNAMIC_AIPP_NAME), std::string("ascend_dynamic_aipp_data"));
    EXPECT_EQ(std::string(ACL_ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES), std::string("_datadump_original_op_names"));
}

TEST_F(UTEST_ACL_compatibility_const_check, acl_op)
{
    EXPECT_EQ(ACL_COMPILE_FLAG_BIN_SELECTOR, 1);
}

TEST_F(UTEST_ACL_compatibility_const_check, model_stream_flags)
{
    EXPECT_EQ(ACL_MODEL_STREAM_FLAG_HEAD, 0x00000000U);
    EXPECT_EQ(ACL_MODEL_STREAM_FLAG_DEFAULT, 0x7FFFFFFFU);
}
