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

class axis_name_util_utest: public testing::Test
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

TEST_F(axis_name_util_utest, get_nchw_reshape_type)
{
    ge::Format format = ge::FORMAT_NCHW;
    std::vector<vector<int64_t>> axis_values = {
        {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
        {0,1,2},{0,1,3}, {0,2,3}, {1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
    };
    std::vector<string> reshape_types = {
        "CHW", "NHW", "NCW", "NCH", "CHW", "NHW", "NCW", "NCH",
        "HW", "CW", "CH", "NW", "NH", "NC", "HW", "CW", "CH", "NW", "NH", "NC",
        "W", "H", "C", "N", "W", "H", "C", "N"
    };
    for (size_t i = 0; i < axis_values.size(); i++) {
      string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
      EXPECT_EQ(reshape_type, reshape_types[i]);
    }
}

TEST_F(axis_name_util_utest, get_nhwc_reshape_type)
{
  ge::Format format = ge::FORMAT_NHWC;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {
      "HWC", "NWC", "NHC", "NHW", "HWC", "NWC", "NHC", "NHW",
      "WC", "HC", "HW", "NC", "NW", "NH", "WC", "HC", "HW", "NC", "NW", "NH",
      "C", "W", "H", "N", "C", "W", "H", "N"
  };
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

TEST_F(axis_name_util_utest, get_hwcn_reshape_type)
{
  ge::Format format = ge::FORMAT_HWCN;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {
      "WCN", "HCN", "HWN", "HWC", "WCN", "HCN", "HWN", "HWC",
      "CN", "WN", "WC", "HN", "HC", "HW", "CN", "WN", "WC", "HN", "HC", "HW",
      "N", "C", "W", "H", "N", "C", "W", "H"
  };
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

TEST_F(axis_name_util_utest, get_chwn_reshape_type)
{
  ge::Format format = ge::FORMAT_CHWN;
  std::vector<vector<int64_t>> axis_values = {
      {0}, {1}, {2}, {3}, {-4}, {-3}, {-2}, {-1},
      {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {-4,-3}, {-4,-2}, {-4,-1}, {-3,-2}, {-3,-1}, {-2,-1},
      {0,1,2},{0,1,3},{0,2,3},{1,2,3}, {-4,-3,-2},{-4,-3,-1}, {-4,-2,-1}, {-3,-2,-1}
  };
  std::vector<string> reshape_types = {
      "HWN", "CWN", "CHN", "CHW", "HWN", "CWN", "CHN", "CHW",
      "WN", "HN", "HW", "CN", "CW", "CH", "WN", "HN", "HW", "CN", "CW", "CH",
      "N", "W", "H", "C", "N", "W", "H", "C"
  };
  for (size_t i = 0; i < axis_values.size(); i++) {
    string reshape_type = AxisNameUtil::GetReshapeType(format, axis_values[i]);
    EXPECT_EQ(reshape_type, reshape_types[i]);
  }
}

TEST_F(axis_name_util_utest, get_NDHWC_reshape_type)
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

TEST_F(axis_name_util_utest, get_NCDHW_reshape_type)
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

TEST_F(axis_name_util_utest, get_DHWCN_reshape_type)
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

TEST_F(axis_name_util_utest, get_DHWNC_reshape_type)
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

TEST_F(axis_name_util_utest, GetCHWNAxisName_suc) {
  vector<int64_t> axis_values = {0, 1, 2, 3};
  std::vector<string> axis_name;
  Status ret = AxisNameUtil::GetCHWNAxisName(axis_values, axis_name);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::map<ge::Format, GetAxisNameByAxisValueInfoPtr>::iterator iter_get_axis_func =
      get_axis_name_except_func_map_stub.find(ge::FORMAT_NDHWC);
  AxisNameUtil::GetAxisNames(axis_values, iter_get_axis_func);
}

TEST_F(axis_name_util_utest, check_params_test)
{
  vector<int64_t> original_dim_vec = {0, 1, 2};
  uint32_t c0 = 1;
  vector<int64_t> nd_value = {0, 1, 2, 3};
  size_t dim_default_size = DIM_DEFAULT_SIZE;
  Status status = AxisUtil::CheckParams(original_dim_vec, c0, nd_value, dim_default_size);
  EXPECT_EQ(status, fe::FAILED);
}
