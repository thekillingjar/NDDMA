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
#include <nlohmann/json.hpp>
#include <iostream>

#include <list>

#include "framework/common/types.h"
#include "graph/utils/tensor_utils.h"

#define protected public
#define private public
#include "param_calculate/tensor_compute_util.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class tensor_compute_util_st: public testing::Test
{
protected:
    void SetUp()
    {
    }

    void TearDown()
    {

    }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(tensor_compute_util_st, verify_tensor_success1)
{
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    Status status = TensorComputeUtil::VerifyTensor(tensor_desc);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(tensor_compute_util_st, verify_tensor_success2)
{
    vector<int64_t> dims;
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    Status status = TensorComputeUtil::VerifyTensor(tensor_desc);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(tensor_compute_util_st, verify_tensor_fail1)
{
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_RESERVED);
    tensor_desc.SetDataType(DT_FLOAT16);
    Status status = TensorComputeUtil::VerifyTensor(tensor_desc);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(tensor_compute_util_st, verify_tensor_fail2)
{
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_UNDEFINED);
    Status status = TensorComputeUtil::VerifyTensor(tensor_desc);
    EXPECT_NE(status, fe::SUCCESS);
}

TEST_F(tensor_compute_util_st, verify_tensor_fail3)
{
    vector<int64_t> dims = {1,2,-3,4};
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    Status status = TensorComputeUtil::VerifyTensor(tensor_desc);
    EXPECT_EQ(status, INVALID_DIM_VALUE);
}

TEST_F(tensor_compute_util_st, GetElementCountByMultiply_01)
{
    vector<int64_t> dims = {};
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    int64_t element_cnt;
    Status status = TensorComputeUtil::GetElementCountByMultiply(tensor_desc, element_cnt);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(element_cnt, 1);
}

TEST_F(tensor_compute_util_st, GetElementCountByMultiply_02)
{
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    int64_t element_cnt;
    Status status = TensorComputeUtil::GetElementCountByMultiply(tensor_desc, element_cnt);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(element_cnt, 24);
}

TEST_F(tensor_compute_util_st, GetTensorSizeByDataType_01)
{
    int64_t element_cnt = 0;
    ge::DataType data_type = DT_FLOAT16;
    int64_t data_type_size;
    int32_t output_real_calc_flag = 0;
    Status status = TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, data_type_size, output_real_calc_flag);

    EXPECT_EQ(status, fe::FAILED);
}

TEST_F(tensor_compute_util_st, GetTensorSizeByDataType_02)
{
    int64_t element_cnt = 2000000001;
    ge::DataType data_type = DT_FLOAT16;
    int64_t data_type_size;
    int32_t output_real_calc_flag = 0;
    Status status = TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, data_type_size, output_real_calc_flag);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(tensor_compute_util_st, GetTensorSizeByDataType_03)
{
    int64_t element_cnt = 100;
    ge::DataType data_type = DT_FLOAT16;
    int64_t size;
    int32_t output_real_calc_flag = 0;
    Status status = TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, size, output_real_calc_flag);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(size, 256);
}

TEST_F(tensor_compute_util_st, GetTensorSizeByDataType_04)
{
    int64_t element_cnt = 2000000000;
    ge::DataType data_type = DT_UNDEFINED;
    int64_t size;
    int32_t output_real_calc_flag = 0;
    Status status = TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, size, output_real_calc_flag);
    EXPECT_EQ(status, fe::FAILED);
}
