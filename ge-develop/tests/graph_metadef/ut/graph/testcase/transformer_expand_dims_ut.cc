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
#include <bitset>
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "expand_dimension.h"

namespace transformer {
class TransformerExpandDimsUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  void EXPECT_RunExpandDimsCase(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                                const vector<int64_t> &dims, const vector<int64_t> &expect_dims) {
    std::cout << "EXPECT_RunExpandDimsCase: origin_format=" << origin_format << ", format=" << format
              << ", reahpe type=" << reshape_type << ", dim size=" << dims.size() << std::endl;
    string op_type = "Relu";
    uint32_t tensor_index = 0;
    ge::GeShape shape(dims);
    bool ret = ExpandDimension(op_type, origin_format, format, tensor_index, reshape_type, shape);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(shape.GetDims(), expect_dims);

    ge::GeShape new_shape(dims);
    int64_t int_reshape_type = ExpandDimension::GenerateReshapeType(origin_format, format, new_shape.GetDimNum(),
                                                                    reshape_type);
    if (int_reshape_type != 0) {
      size_t full_size = static_cast<size_t>(int_reshape_type >> 56);
      size_t expect_full_size = 0;
      ExpandDimension::GetFormatFullSize(origin_format, expect_full_size);
      EXPECT_EQ(full_size, expect_full_size);
    }

    ExpandDimension::ExpandDims(int_reshape_type, new_shape);
    EXPECT_EQ(new_shape.GetDims(), expect_dims);

    ge::GeShape shape_1(dims);
    ge::GeShape shape_2(dims);
    ExpandDimension::ExpandDims(int_reshape_type, shape_1, shape_2);
    EXPECT_EQ(shape_2.GetDims(), expect_dims);
  }

  void EXPECT_RunNewExpandDimsCase(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                                   const vector<int64_t> &dims, const vector<int64_t> &expect_dims) {
    std::cout << "origin_format=" << origin_format << ", format=" << format
              << ", reahpe type=" << reshape_type << ", dim size=" << dims.size() << std::endl;
    ge::GeShape new_shape(dims);
    int64_t int_reshape_type = ExpandDimension::GenerateReshapeType(origin_format, format, new_shape.GetDimNum(),
                                                                    reshape_type);
    if (int_reshape_type != 0) {
      size_t full_size = static_cast<size_t>(int_reshape_type >> 56);
      size_t expect_full_size = 0;
      ExpandDimension::GetFormatFullSize(origin_format, expect_full_size);
      EXPECT_EQ(full_size, expect_full_size);
    }
    ExpandDimension::ExpandDims(int_reshape_type, new_shape);
    EXPECT_EQ(new_shape.GetDims(), expect_dims);
  }

  void RunReshapeTypeCase(const ge::Format &format, const size_t &dims_size, const std::string &reshape_type) {
    std::cout << "RunReshapeTypeCase: origin format=" << format << ", dims size=" << dims_size << ", reshape_type=" << reshape_type << std::endl;
    int64_t reshape_mask = transformer::ExpandDimension::GenerateReshapeType(format, ge::FORMAT_NC1HWC0, dims_size, reshape_type);
    std::string ret_shape_type;
    std::string fail_reason;
    bool ret = transformer::ExpandDimension::GenerateReshapeTypeByMask(format, dims_size, reshape_mask, ret_shape_type, fail_reason);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(fail_reason.empty(), true);
    EXPECT_EQ(reshape_type, ret_shape_type);
  }
};

TEST_F(TransformerExpandDimsUT, all_expand_dims_cases_1) {
  int64_t max_reshape_type = 0xff;
  vector<size_t> full_size_vec = {4, 5};
  vector<vector<int64_t>> dim_vecs = {{}, {5}, {5, 6}, {5, 6, 7}, {5, 6, 7, 8}, {5, 6, 7, 8, 9}};
  for (const size_t &full_size : full_size_vec) {
    for (const vector<int64_t> &dims : dim_vecs) {
      for (int64_t i = 0; i <= max_reshape_type; i++) {
        ge::GeShape shape(dims);
        int64_t reshape_type = i | (full_size << 56);
        std::cout << "reshape_type = " << std::bitset<8>(reshape_type) << ", shape = " << shape.ToString();
        EXPECT_NO_THROW(ExpandDimension::ExpandDims(reshape_type, shape));
        std::cout << ", after expand dims shape = " << shape.ToString() << std::endl;
      }
    }
  }
}

TEST_F(TransformerExpandDimsUT, all_expand_dims_cases_2) {
  int64_t max_reshape_type = 0xff;
  vector<size_t> full_size_vec = {4, 5};
  vector<vector<int64_t>> dim_vecs = {{}, {5}, {5, 6}, {5, 6, 7}, {5, 6, 7, 8}, {5, 6, 7, 8, 9}};
  for (const size_t &full_size : full_size_vec) {
    for (const vector<int64_t> &dims : dim_vecs) {
      for (int64_t i = 0; i <= max_reshape_type; i++) {
        ge::GeShape shape_1(dims);
        ge::GeShape shape_2(dims);
        int64_t reshape_type = i | (full_size << 56);
        std::cout << "reshape_type = " << std::bitset<8>(reshape_type) << ", shape = " << shape_1.ToString();
        EXPECT_NO_THROW(ExpandDimension::ExpandDims(reshape_type, shape_1, shape_2));
        std::cout << ", after expand dims shape = " << shape_2.ToString() << std::endl;
      }
    }
  }
}

TEST_F(TransformerExpandDimsUT, not_expand_cases) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "FORBIDDEN", {8, 9}, {8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, "HW", {8, 9}, {8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {6, 7, 8, 9}, {6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "", {4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, "CN", {8, 9}, {8, 9});

  EXPECT_RunNewExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_ND, "HW", {8, 9}, {8, 9});
  EXPECT_RunNewExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_ND_RNN_BIAS, "HW", {8, 9}, {8, 9});
  EXPECT_RunNewExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_ZN_RNN, "HW", {8, 9}, {8, 9});
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_cases) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WN", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CN", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NH", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "NC", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CN", {}, {1, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WN", {}, {1, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "ND", {}, {1, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NC1HWC0, "CD", {}, {1, 1, 1, 1, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "", {5}, {1, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NC1HWC0, "", {5}, {1, 1, 1, 1, 5});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WN", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CWN", {5}, {1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NH", {5}, {1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "NCHW", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CN", {5}, {1, 1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WNCD", {5}, {1, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "ND", {5}, {1, 1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NC1HWC0, "CD", {5}, {1, 1, 1, 1, 5});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NH", {5, 6}, {1, 1, 5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "NC", {5, 6}, {1, 1, 5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {1, 1, 1, 5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 1, 1, 5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "ND", {5, 6}, {1, 1, 1, 5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NC1HWC0, "CD", {5, 6}, {1, 1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "WHN", {5, 6, 7}, {1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "CWN", {5, 6, 7}, {1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "NHW", {5, 6, 7}, {1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CNW", {5, 6, 7}, {1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CND", {5, 6, 7}, {1, 1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "WDN", {5, 6, 7}, {1, 1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "WCND", {5, 6, 7}, {1, 1, 5, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NC1HWC0, "NCD", {5, 6, 7}, {1, 1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NC1HWC0, "CNWD", {5, 6, 7, 8}, {1, 5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, "NDHWC", {5, 6, 7, 8}, {1, 5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7, 8}, {1, 5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NC1HWC0, "NCDH", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_DHWNC, "N", {5}, {5, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, "D", {5}, {1, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, "H", {5}, {1, 1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, "W", {5}, {1, 1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, "ND", {6, 5}, {6, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, "HW", {6, 5}, {1, 1, 6, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, "NC", {6, 5}, {6, 1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_DHWNC, "DC", {6, 5}, {1, 6, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, "NDH", {7, 6, 5}, {7, 6, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, "HWC", {7, 6, 5}, {1, 1, 7, 6, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, "NDHW", {8, 7, 6, 5}, {8, 7, 6, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_DHWNC, "N", {5}, {5, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, "C", {5}, {1, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, "D", {5}, {1, 1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, "H", {5}, {1, 1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, "NC", {6, 5}, {6, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, "DH", {6, 5}, {1, 1, 6, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, "NW", {6, 5}, {6, 1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_DHWNC, "CW", {6, 5}, {1, 6, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, "NCD", {7, 6, 5}, {7, 6, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, "DHW", {7, 6, 5}, {1, 1, 7, 6, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, "NCDH", {8, 7, 6, 5}, {8, 7, 6, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, "D", {5}, {5, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NCDHW, "H", {5}, {1, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDHWC, "W", {5}, {1, 1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, "C", {5}, {1, 1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDHWC, "DH", {6, 5}, {6, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NCDHW, "WC", {6, 5}, {1, 1, 6, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, "DN", {6, 5}, {6, 1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, "HN", {6, 5}, {1, 6, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDHWC, "DHW", {7, 6, 5}, {7, 6, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NCDHW, "WCN", {7, 6, 5}, {1, 1, 7, 6, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDHWC, "DHWC", {8, 7, 6, 5}, {8, 7, 6, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_DHWNC, "D", {5}, {5, 1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NCDHW, "H", {5}, {1, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NDHWC, "W", {5}, {1, 1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_DHWCN, "N", {5}, {1, 1, 1, 5, 1});  
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NDHWC, "DH", {6, 5}, {6, 5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NCDHW, "WN", {6, 5}, {1, 1, 6, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_DHWCN, "DC", {6, 5}, {6, 1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_DHWNC, "HC", {6, 5}, {1, 6, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NDHWC, "DHW", {7, 6, 5}, {7, 6, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NCDHW, "WNC", {7, 6, 5}, {1, 1, 7, 6, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWNC, ge::FORMAT_NDHWC, "DHWN", {8, 7, 6, 5}, {8, 7, 6, 5, 1});
}

TEST_F(TransformerExpandDimsUT, nchw_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "N", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HCW", {}, {1, 1, 1, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "N", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "C", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "H", {5}, {1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "W", {5}, {1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CHW", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {5}, {1, 1, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NC", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCH", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCW", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NH", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NHW", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NW", {5, 6}, {5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CH", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CHW", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CW", {5, 6}, {1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCH", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCW", {5, 6, 7}, {5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NHW", {5, 6, 7}, {5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "CHW", {5, 6, 7}, {1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NC", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "NCHW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(TransformerExpandDimsUT, nhwc_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "N", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NH", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {}, {1, 1, 1, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "N", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "H", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "W", {5}, {1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "C", {5}, {1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "WC", {5}, {1, 1, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NH", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHW", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHC", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NW", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NWC", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NC", {5, 6}, {5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HC", {5, 6}, {1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "WC", {5, 6}, {1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHW", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHC", {5, 6, 7}, {5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NWC", {5, 6, 7}, {5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HWC", {5, 6, 7}, {1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NC", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "NHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(TransformerExpandDimsUT, hwcn_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "N", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HW", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWN", {}, {1, 1, 1, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "H", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "W", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "C", {5}, {1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "N", {5}, {1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WCN", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "CN", {5}, {1, 1, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWC", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWN", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HC", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HCN", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HN", {5, 6}, {5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WC", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WCN", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWC", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWN", {5, 6, 7}, {5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HCN", {5, 6, 7}, {5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "WCN", {5, 6, 7}, {1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "HWCN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, "CN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(TransformerExpandDimsUT, chwn_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "C", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HW", {}, {1, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {}, {1, 1, 1, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "C", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "H", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "W", {5}, {1, 1, 5, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "N", {5}, {1, 1, 1, 5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5}, {5, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {5}, {1, 5, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "WN", {5}, {1, 1, 5, 1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "C", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "W", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "N", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CH", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHW", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHN", {5, 6}, {5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CW", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CWN", {5, 6}, {5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CN", {5, 6}, {5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HW", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {5, 6}, {1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HN", {5, 6}, {1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "WN", {5, 6}, {1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHW", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6, 7}, {5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHN", {5, 6, 7}, {5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CWN", {5, 6, 7}, {5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HWN", {5, 6, 7}, {1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "HW", {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CHWN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, "CN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
}

TEST_F(TransformerExpandDimsUT, ndhwc_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "C", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HW", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHW", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDWC", {}, {1, 1, 1, 1 ,1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "N", {5}, {5, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "D", {5}, {1, 5, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "H", {5}, {1, 1, 5, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "W", {5}, {1, 1, 1, 5 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "C", {5}, {1, 1, 1, 1 ,5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5}, {5, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHWC", {5}, {1, 5, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HWC", {5}, {1, 1, 5, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "WC", {5}, {1, 1, 1, 5 ,1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "N", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "D", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "W", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "C", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "ND", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDH", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDC", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHC", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NH", {5, 6}, {5, 1, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NW", {5, 6}, {5, 1, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NC", {5, 6}, {5, 1, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DH", {5, 6}, {1, 5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DW", {5, 6}, {1, 5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DC", {5, 6}, {1, 5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HW", {5, 6}, {1, 1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HC", {5, 6}, {1, 1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "WC", {5, 6}, {1, 1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDH", {5, 6, 7}, {5, 6, 7, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDW", {5, 6, 7}, {5, 6, 1, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDC", {5, 6, 7}, {5, 6, 1, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHW", {5, 6, 7}, {5, 1, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHC", {5, 6, 7}, {5, 1, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NWC", {5, 6, 7}, {5, 1, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHW", {5, 6, 7}, {1, 5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHC", {5, 6, 7}, {1, 5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DWC", {5, 6, 7}, {1, 5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "HWC", {5, 6, 7}, {1, 1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHW", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHC", {5, 6, 7, 8}, {5, 6, 7, 1, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDWC", {5, 6, 7, 8}, {5, 6, 1, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHWC", {5, 6, 7, 8}, {5, 1, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDWC", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, "NDHWC", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
}

TEST_F(TransformerExpandDimsUT, ncdhw_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "C", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "HW", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NHW", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDHW", {}, {1, 1, 1, 1 ,1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "N", {5}, {5, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "C", {5}, {1, 5, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "D", {5}, {1, 1, 5, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "H", {5}, {1, 1, 1, 5 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "W", {5}, {1, 1, 1, 1 ,5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5}, {5, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDHW", {5}, {1, 5, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DHW", {5}, {1, 1, 5, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "HW", {5}, {1, 1, 1, 5 ,1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "N", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "C", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "D", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "W", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NC", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCD", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCH", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDH", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCHW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "ND", {5, 6}, {5, 1, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NH", {5, 6}, {5, 1, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NW", {5, 6}, {5, 1, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CD", {5, 6}, {1, 5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CH", {5, 6}, {1, 5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CW", {5, 6}, {1, 5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DH", {5, 6}, {1, 1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DW", {5, 6}, {1, 1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "HW", {5, 6}, {1, 1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCD", {5, 6, 7}, {5, 6, 7, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCH", {5, 6, 7}, {5, 6, 1, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCW", {5, 6, 7}, {5, 6, 1, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDH", {5, 6, 7}, {5, 1, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDW", {5, 6, 7}, {5, 1, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NHW", {5, 6, 7}, {5, 1, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDH", {5, 6, 7}, {1, 5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDW", {5, 6, 7}, {1, 5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CHW", {5, 6, 7}, {1, 5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "DHW", {5, 6, 7}, {1, 1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDH", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDW", {5, 6, 7, 8}, {5, 6, 7, 1, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCHW", {5, 6, 7, 8}, {5, 6, 1, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NDHW", {5, 6, 7, 8}, {5, 1, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDHW", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCHW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "CDHW", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, "NCDHW", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type) {
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "C", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HW", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWC", {}, {1, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWN", {}, {1, 1, 1, 1 ,1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "D", {5}, {5, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "H", {5}, {1, 5, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "W", {5}, {1, 1, 5, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "C", {5}, {1, 1, 1, 5 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "N", {5}, {1, 1, 1, 1 ,5});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5}, {5, 1, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWCN", {5}, {1, 5, 1, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WCN", {5}, {1, 1, 5, 1 ,1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "CN", {5}, {1, 1, 1, 5 ,1});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "D", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "H", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "W", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "C", {5, 6}, {5, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "N", {5, 6}, {5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DH", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHW", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHC", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHN", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWN", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHCN", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6}, {5, 6, 1, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DW", {5, 6}, {5, 1, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DC", {5, 6}, {5, 1, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DN", {5, 6}, {5, 1, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HW", {5, 6}, {1, 5, 6, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HC", {5, 6}, {1, 5, 1, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HN", {5, 6}, {1, 5, 1, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WC", {5, 6}, {1, 1, 5, 6, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WN", {5, 6}, {1, 1, 5, 1, 6});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "CN", {5, 6}, {1, 1, 1, 5, 6});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHW", {5, 6, 7}, {5, 6, 7, 1, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHC", {5, 6, 7}, {5, 6, 1, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHN", {5, 6, 7}, {5, 6, 1, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DWC", {5, 6, 7}, {5, 1, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DWN", {5, 6, 7}, {5, 1, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DCN", {5, 6, 7}, {5, 1, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWC", {5, 6, 7}, {1, 5, 6, 7, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWN", {5, 6, 7}, {1, 5, 6, 1, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HCN", {5, 6, 7}, {1, 5, 1, 6, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "WCN", {5, 6, 7}, {1, 1, 5, 6, 7});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6, 7, 8}, {5, 6, 7, 8, 1});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWN", {5, 6, 7, 8}, {5, 6, 7, 1, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHCN", {5, 6, 7, 8}, {5, 6, 1, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DWCN", {5, 6, 7, 8}, {5, 1, 6, 7, 8});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWCN", {5, 6, 7, 8}, {1, 5, 6, 7, 8});

  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWC", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6, 7, 8, 9}, {5, 6, 7, 8, 9});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "HWCN", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
  EXPECT_RunExpandDimsCase(ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, "DHWCN", {5, 6, 7, 8, 9, 7}, {5, 6, 7, 8, 9, 7});
}

TEST_F(TransformerExpandDimsUT, reshape_type_case1) {
  RunReshapeTypeCase(ge::FORMAT_NCHW, 1, "N");
  RunReshapeTypeCase(ge::FORMAT_NCHW, 2, "NC");
  RunReshapeTypeCase(ge::FORMAT_NCHW, 2, "HW");
  RunReshapeTypeCase(ge::FORMAT_NCHW, 3, "NHW");
  RunReshapeTypeCase(ge::FORMAT_NCHW, 3, "CHW");
  RunReshapeTypeCase(ge::FORMAT_NCHW, 4, "NCHW");

  RunReshapeTypeCase(ge::FORMAT_NDHWC, 1, "D");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 2, "NC");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 2, "DC");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 2, "HW");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 3, "NHC");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 3, "HWC");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 3, "NHW");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 4, "DHWC");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 4, "NHWC");
  RunReshapeTypeCase(ge::FORMAT_NDHWC, 5, "NDHWC");
}

TEST_F(TransformerExpandDimsUT, reshape_type_case2) {
  std::string reshape_type;
  std::string failed_reason;
  bool ret = transformer::ExpandDimension::GenerateReshapeTypeByMask(ge::FORMAT_ND, 2, 0, reshape_type, failed_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(reshape_type.empty(), true);

  ret = transformer::ExpandDimension::GenerateReshapeTypeByMask(ge::FORMAT_ND, 2, 1, reshape_type, failed_reason);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(reshape_type.empty(), true);

  ret = transformer::ExpandDimension::GenerateReshapeTypeByMask(ge::FORMAT_NC1HWC0, 2, 1, reshape_type, failed_reason);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(reshape_type.empty(), true);

  int64_t reshape_mask = 3;
  ret = transformer::ExpandDimension::GenerateReshapeTypeByMask(ge::FORMAT_NHWC, 2, reshape_mask, reshape_type, failed_reason);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(reshape_type.empty(), true);

  reshape_mask = 3;
  ret = transformer::ExpandDimension::GenerateReshapeTypeByMask(ge::FORMAT_NHWC, 3, reshape_mask, reshape_type, failed_reason);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(reshape_type.empty(), true);
}
}  // namespace ge
