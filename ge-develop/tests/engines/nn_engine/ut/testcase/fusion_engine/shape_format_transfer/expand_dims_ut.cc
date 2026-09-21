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
#include "graph/ge_tensor.h"

#define protected public
#define private public
#include "expand_dimension.h"
#include "common/fe_op_info_common.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

class ut_expand_dims : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  Status RunExpandDimsCase(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                           const vector<int64_t> &dims, const vector<int64_t> &expect_dims) {
    std::cout << "RunExpandDimsCase: origin_format=" << origin_format << ", format=" << format
              << ", reahpe type=" << reshape_type << ", dim size=" << dims.size() << std::endl;
    ge::GeShape new_shape(dims);
    ExpandDimension(origin_format, format, reshape_type, new_shape);
    EXPECT_EQ(new_shape.GetDims(), expect_dims);

    ge::GeShape shape_1(dims);
    ge::GeShape shape_2(dims);
    ExpandDimension(origin_format, format, reshape_type, shape_1, shape_2);
    EXPECT_EQ(shape_2.GetDims(), expect_dims);

    if (new_shape.GetDims() != expect_dims || shape_2.GetDims() != expect_dims) {
      return fe::FAILED;
    }
    return fe::SUCCESS;
  }
};

TEST_F(ut_expand_dims, not_expand_cases) {
  Status ret = RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "FORBIDDEN", {8, 9}, {8, 9});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, "HW", {8, 9}, {8, 9});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {6, 7, 8, 9}, {6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "", {4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, "CN", {8, 9}, {8, 9});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_ND, "HW", {8, 9}, {8, 9});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_ND_RNN_BIAS, "HW", {8, 9}, {8, 9});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_ZN_RNN, "HW", {8, 9}, {8, 9});
}

TEST_F(ut_expand_dims, default_reshape_type_cases) {
  Status ret = RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WN", {}, {1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CN", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NH", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "NC", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CN", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WN", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "ND", {}, {1, 1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WN", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CWN", {5}, {1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NH", {5}, {1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "NCHW", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CN", {5}, {1, 1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WNCD", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "ND", {5}, {1, 1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NH", {5, 6}, {1, 1, 5, 6});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "NC", {5, 6}, {1, 1, 5, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {1, 1, 1, 5, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 1, 1, 5, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "ND", {5, 6}, {1, 1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WHN", {5, 6, 7}, {1, 5, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CWN", {5, 6, 7}, {1, 5, 6, 7});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NHW", {5, 6, 7}, {1, 5, 6, 7});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CNW", {5, 6, 7}, {1, 5, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CND", {5, 6, 7}, {1, 1, 5, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WDN", {5, 6, 7}, {1, 1, 5, 6, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "WCND", {5, 6, 7}, {1, 1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CNWD", {5, 6, 7, 8}, {1, 5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "NDHWC", {5, 6, 7, 8}, {1, 5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7, 8}, {1, 5, 6, 7, 8});
}

TEST_F(ut_expand_dims, nchw_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "N", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HCW", {}, {1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "N", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "C", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "H", {5}, {1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "W", {5}, {1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CHW", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {5}, {1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NC", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCH", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCW", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NH", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NHW", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NW", {5, 6}, {5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CH", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CHW", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CW", {5, 6}, {1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCH", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCW", {5, 6, 7}, {5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NHW", {5, 6, 7}, {5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CHW", {5, 6, 7}, {1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NC", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(ut_expand_dims, nhwc_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "N", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NH", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {}, {1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "N", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "H", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "W", {5}, {1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "C", {5}, {1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "WC", {5}, {1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NH", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHW", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHC", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NW", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NWC", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NC", {5, 6}, {5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HC", {5, 6}, {1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "WC", {5, 6}, {1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHW", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHC", {5, 6, 7}, {5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NWC", {5, 6, 7}, {5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {5, 6, 7}, {1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NC", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(ut_expand_dims, hwcn_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "N", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HW", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWN", {}, {1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "H", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "W", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "C", {5}, {1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "N", {5}, {1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WCN", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "CN", {5}, {1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWC", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWN", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HC", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HCN", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HN", {5, 6}, {5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WC", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WCN", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWC", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWN", {5, 6, 7}, {5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HCN", {5, 6, 7}, {5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WCN", {5, 6, 7}, {1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "CN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(ut_expand_dims, chwn_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "C", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HW", {}, {1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {}, {1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "C", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "H", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "W", {5}, {1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "N", {5}, {1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5}, {5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {5}, {1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "WN", {5}, {1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CH", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHW", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHN", {5, 6}, {5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CW", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CWN", {5, 6}, {5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {5, 6}, {1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HN", {5, 6}, {1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHW", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6, 7}, {5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHN", {5, 6, 7}, {5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CWN", {5, 6, 7}, {5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {5, 6, 7}, {1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8}, {5, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(ut_expand_dims, ndhwc_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "", {}, {1, 1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "C", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HW", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHW", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDWC", {}, {1, 1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "N", {5}, {5, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "D", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "H", {5}, {1, 1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "W", {5}, {1, 1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "C", {5}, {1, 1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5}, {5, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHWC", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HWC", {5}, {1, 1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "WC", {5}, {1, 1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "N", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "D", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "W", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "C", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "ND", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDH", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDC", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHC", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NH", {5, 6}, {5, 1, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NW", {5, 6}, {5, 1, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NC", {5, 6}, {5, 1, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DH", {5, 6}, {1, 5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DW", {5, 6}, {1, 5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DC", {5, 6}, {1, 5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HW", {5, 6}, {1, 1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HC", {5, 6}, {1, 1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "WC", {5, 6}, {1, 1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDH", {5, 6, 7}, {5, 6, 7, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDW", {5, 6, 7}, {5, 6, 1, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDC", {5, 6, 7}, {5, 6, 1, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHW", {5, 6, 7}, {5, 1, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHC", {5, 6, 7}, {5, 1, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NWC", {5, 6, 7}, {5, 1, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHW", {5, 6, 7}, {1, 5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHC", {5, 6, 7}, {1, 5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DWC", {5, 6, 7}, {1, 5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HWC", {5, 6, 7}, {1, 1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHW", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHC", {5, 6, 7, 8}, {5, 6, 7, 1, 8});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDWC", {5, 6, 7, 8}, {5, 6, 1, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHWC", {5, 6, 7, 8}, {5, 1, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDWC", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
  RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
}

TEST_F(ut_expand_dims, ncdhw_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "", {}, {1, 1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "C", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "HW", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NHW", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDHW", {}, {1, 1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "N", {5}, {5, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "C", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "D", {5}, {1, 1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "H", {5}, {1, 1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "W", {5}, {1, 1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5}, {5, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDHW", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DHW", {5}, {1, 1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "HW", {5}, {1, 1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "N", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "C", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "D", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "W", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NC", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCD", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCH", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDH", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCHW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "ND", {5, 6}, {5, 1, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NH", {5, 6}, {5, 1, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NW", {5, 6}, {5, 1, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CD", {5, 6}, {1, 5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CH", {5, 6}, {1, 5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CW", {5, 6}, {1, 5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DH", {5, 6}, {1, 1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DW", {5, 6}, {1, 1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "HW", {5, 6}, {1, 1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCD", {5, 6, 7}, {5, 6, 7, 1, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCH", {5, 6, 7}, {5, 6, 1, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCW", {5, 6, 7}, {5, 6, 1, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDH", {5, 6, 7}, {5, 1, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDW", {5, 6, 7}, {5, 1, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NHW", {5, 6, 7}, {5, 1, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDH", {5, 6, 7}, {1, 5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDW", {5, 6, 7}, {1, 5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CHW", {5, 6, 7}, {1, 5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DHW", {5, 6, 7}, {1, 1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDH", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDW", {5, 6, 7, 8}, {5, 6, 7, 1, 8});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCHW", {5, 6, 7, 8}, {5, 6, 1, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDHW", {5, 6, 7, 8}, {5, 1, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDHW", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCHW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDHW", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
  RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
}

TEST_F(ut_expand_dims, dhwcn_reshape_type) {
  Status ret = RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "", {}, {1, 1, 1, 1, 1});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "C", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HW", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWC", {}, {1, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWN", {}, {1, 1, 1, 1, 1});

  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "D", {5}, {5, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "H", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "W", {5}, {1, 1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "C", {5}, {1, 1, 1, 5, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "N", {5}, {1, 1, 1, 1, 5});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5}, {5, 1, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWCN", {5}, {1, 5, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WCN", {5}, {1, 1, 5, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "CN", {5}, {1, 1, 1, 5, 1});

  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "D", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "H", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "W", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "C", {5, 6}, {5, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "N", {5, 6}, {5, 6});

  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DH", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHW", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHC", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHN", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWN", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHCN", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6}, {5, 6, 1, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DW", {5, 6}, {5, 1, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DC", {5, 6}, {5, 1, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DN", {5, 6}, {5, 1, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HW", {5, 6}, {1, 5, 6, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HC", {5, 6}, {1, 5, 1, 6, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HN", {5, 6}, {1, 5, 1, 1, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WC", {5, 6}, {1, 1, 5, 6, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WN", {5, 6}, {1, 1, 5, 1, 6});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "CN", {5, 6}, {1, 1, 1, 5, 6});

  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHW", {5, 6, 7}, {5, 6, 7, 1, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHC", {5, 6, 7}, {5, 6, 1, 7, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHN", {5, 6, 7}, {5, 6, 1, 1, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DWC", {5, 6, 7}, {5, 1, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DWN", {5, 6, 7}, {5, 1, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DCN", {5, 6, 7}, {5, 1, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWC", {5, 6, 7}, {1, 5, 6, 7, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWN", {5, 6, 7}, {1, 5, 6, 1, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HCN", {5, 6, 7}, {1, 5, 1, 6, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WCN", {5, 6, 7}, {1, 1, 5, 6, 7});

  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWN", {5, 6, 7, 8}, {5, 6, 7, 1, 8});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHCN", {5, 6, 7, 8}, {5, 6, 1, 7, 8});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DWCN", {5, 6, 7, 8}, {5, 1, 6, 7, 8});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWCN", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWCN", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
  RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
}
