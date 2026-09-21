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

#define protected public
#define private public

#include "common/fe_utils.h"
#include "common/format/axis_name_util.h"
#include "common/format/axis_util.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class axis_name_util_stest: public testing::Test
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

TEST_F(axis_name_util_stest, get_NDHWC_reshape_type)
{
  ge::Format format = ge::FORMAT_NDHWC;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {"DHWC", "NHWC", "NDWC", "NDHC", "NHWC", "NDWC", "NDHC", "NDHW", "HWC", "DWC",
                                       "DHC",  "NWC",  "NHC",  "NDC",  "NWC",  "NHC",  "NHW",  "NDC",  "NDW", "NDH",
                                       "WC",   "HC",   "DC",   "NC",   "NC",   "NW",   "NH",   "ND"};
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

TEST_F(axis_name_util_stest, get_NCDHW_reshape_type)
{
  ge::Format format = ge::FORMAT_NCDHW;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {"CDHW", "NDHW", "NCHW", "NCDW", "NDHW", "NCHW", "NCDW", "NCDH", "DHW", "CHW",
                                       "CDW",  "NHW",  "NDW",  "NCW",  "NHW",  "NDW",  "NDH",  "NCW",  "NCH", "NCD",
                                       "HW",   "DW",   "CW",   "NW",   "NW",   "NH",   "ND",   "NC"};
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

TEST_F(axis_name_util_stest, get_DHWCN_reshape_type)
{
  ge::Format format = ge::FORMAT_DHWCN;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {"HWCN", "DWCN", "DHCN", "DHWN", "DWCN", "DHCN", "DHWN", "DHWC", "WCN", "HCN",
                                       "HWN",  "DCN",  "DWN",  "DHN",  "DCN",  "DWN",  "DWC",  "DHN",  "DHC", "DHW",
                                       "CN",   "WN",   "HN",   "DN",   "DN",   "DC",   "DW",   "DH"};
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

TEST_F(axis_name_util_stest, get_DHWNC_reshape_type)
{
  ge::Format format = ge::FORMAT_DHWNC;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {"HWNC", "DWNC", "DHNC", "DHWC", "DWNC", "DHNC", "DHWC", "DHWN", "WNC", "HNC",
                                       "HWC",  "DNC",  "DWC",  "DHC",  "DNC",  "DWC",  "DWN",  "DHC",  "DHN", "DHW",
                                       "NC",   "WC",   "HC",   "DC",   "DC",   "DN",   "DW",   "DH"};
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

Status GetNDHWCAxisExceptNameStub(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  return fe::SUCCESS;
}
std::map<ge::Format, GetAxisNameByAxisValueInfoPtr> get_axis_name_except_func_map_stub = {
    {ge::FORMAT_NDHWC, std::make_shared<GetAxisNameByAxisValueInfo>(GetNDHWCAxisExceptNameStub)}};

TEST_F(axis_name_util_stest, GetCHWNAxisName_suc) {
  vector<int64_t> axis_values = {0, 1, 2, 3};
  std::vector<string> axis_name;
  Status ret = AxisNameUtil::GetCHWNAxisName(axis_values, axis_name);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::map<ge::Format, GetAxisNameByAxisValueInfoPtr>::iterator iter_get_axis_func =
      get_axis_name_except_func_map_stub.find(ge::FORMAT_NDHWC);
  AxisNameUtil::GetAxisNames(axis_values, iter_get_axis_func);
}
