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
#include <string>
#include <vector>

#define protected public
#define private public
#include "common/format/range_axis_util.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;


class RANGE_AXIS_UTIL_UTEST: public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(RANGE_AXIS_UTIL_UTEST, AddTransNode_failed1) {
  vector<std::pair<int64_t, int64_t>> original_range_vec;
  vector<int64_t> original_dim_vec;
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value;

  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByND(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, AddTransNode_failed2) {
  vector<std::pair<int64_t, int64_t>> original_range_vec;
  vector<int64_t> original_dim_vec;
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};

  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByND(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, AddTransNode_suc) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};

  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByND(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

// TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByFz_suc) {
//   vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
//   vector<int64_t> original_dim_vec = {1, 1, 1, 1};
//   uint32_t c0;
//   vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
//                                                      {1, 1}, {1, 1}, {1, 1}, {1, 1}};
//
//   RangeAxisUtil range_axis_util;
//   Status ret = range_axis_util.GetRangeAxisValueByFz(original_range_vec, original_dim_vec, c0, range_value);
//   EXPECT_EQ(ret, fe::SUCCESS);
// }

TEST_F(RANGE_AXIS_UTIL_UTEST, CheckParamValue_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  size_t min_size = DIM_DEFAULT_SIZE;
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.CheckParamValue(original_range_vec, original_dim_vec, c0, range_value, min_size);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByND_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByND(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByNCHW_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByNCHW(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByNHWC_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByNHWC(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByNC1HWC0_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByNC1HWC0(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByHWCN_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByHWCN(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(RANGE_AXIS_UTIL_UTEST, GetRangeAxisValueByCHWN_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByCHWN(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}
