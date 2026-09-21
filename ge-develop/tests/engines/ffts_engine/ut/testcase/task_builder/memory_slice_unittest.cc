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

#define private public
#define protected public
#include "inc/memory_slice.h"
#undef private
#undef protected

using namespace std;
using namespace ffts;
using namespace ge;

class MemorySliceUTest : public testing::Test {
 protected:
	void SetUp() {
	}

	void TearDown() {
	}
};

TEST_F(MemorySliceUTest, GenerateDataCtxParam_failed) {
  ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
  vector<int64_t> dim = {1, 1};
  GeShape shape(dim);
  tensor_desc->SetShape(shape);
  tensor_desc->SetFormat(FORMAT_NC1HWC0);
  tensor_desc->SetDataType(DT_FLOAT);
  std::vector<DimRange> slice;
  ge::DataType dtype;
  int64_t burst_len;
  std::vector<DataContextParam> data_ctx;
  Status ret = MemorySlice::GenerateDataCtxParam(shape.GetDims(), slice, DT_FLOAT, burst_len, data_ctx);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(MemorySliceUTest, GenerateDataCtxParam_failed1) {
  ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
  vector<int64_t> dim = {1, 1};
  GeShape shape(dim);
  tensor_desc->SetShape(shape);
  tensor_desc->SetFormat(FORMAT_NC1HWC0);
  tensor_desc->SetDataType(DT_FLOAT);
  DimRange range1 = {-1, -1};
  DimRange range2 = {-1, -1};
  std::vector<DimRange> slice;
  slice.emplace_back(range1);
  slice.emplace_back(range2);
  ge::DataType dtype;
  int64_t burst_len;
  std::vector<DataContextParam> data_ctx;
  Status ret = MemorySlice::GenerateDataCtxParam(shape.GetDims(), slice, DT_FLOAT, burst_len, data_ctx);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(MemorySliceUTest, GenerateDataCtxTypeErr) {
  ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
  vector<int64_t> dim1 = {4, 4, 4};
  GeShape shape(dim1);
  tensor_desc->SetShape(shape);
  tensor_desc->SetFormat(FORMAT_NC1HWC0);
  tensor_desc->SetDataType(DT_INT4);
  std::vector<DimRange> slice;
  DimRange dim;
  dim.higher = 4;
  dim.lower = 3;
  slice.emplace_back(dim);
  dim.higher = 4;
  dim.lower = 0;
  slice.emplace_back(dim);
  slice.emplace_back(dim);
  ge::DataType dtype;
  int64_t burst_len = 0xfffffff;
  std::vector<DataContextParam> data_ctx;
  Status ret = MemorySlice::GenerateDataCtxParam(shape.GetDims(), slice, DT_INT4, burst_len, data_ctx);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(MemorySliceUTest, GenerateDataCtxParam1) {
  ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
  vector<int64_t> dim1 = {4, 4, 4};
  GeShape shape(dim1);
  tensor_desc->SetShape(shape);
  tensor_desc->SetFormat(FORMAT_NC1HWC0);
  tensor_desc->SetDataType(DT_FLOAT);
  std::vector<DimRange> slice;
  DimRange dim;
  dim.higher = 4;
  dim.lower = 3;
  slice.emplace_back(dim);
  dim.higher = 4;
  dim.lower = 0;
  slice.emplace_back(dim);
  slice.emplace_back(dim);
  ge::DataType dtype;
  int64_t burst_len = 0xfffffff;
  std::vector<DataContextParam> data_ctx;
  Status ret = MemorySlice::GenerateDataCtxParam(shape.GetDims(), slice, DT_FLOAT, burst_len, data_ctx);
  EXPECT_EQ(SUCCESS, ret);
}
