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
#include "common/range_format_transfer/transfer_range_according_to_format.h"
#include "common/fe_op_info_common.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

class ut_transfer_shape : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  Status RunTransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &dtype,
                          const bool &expect_ret, const vector<int64_t> &dims, const vector<int64_t> &expect_dim,
                          const ge::OpDescPtr op_desc = nullptr) {
    std::cout << std::endl
              << "RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << std::endl;
    fe::Status expect_status = expect_ret ? fe::SUCCESS : fe::FAILED;
    CalcShapeExtraAttr extra_attr = {1, 1, -1};
    if (op_desc != nullptr) {
      (void)ge::AttrUtils::GetInt(op_desc, "input_size", extra_attr.input_size);
      (void)ge::AttrUtils::GetInt(op_desc, "hidden_size", extra_attr.hidden_size);
      (void)ge::AttrUtils::GetInt(op_desc, "state_size", extra_attr.state_size);
    }

    ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
    int64_t group = static_cast<int64_t>(ge::GetSubFormat(format));
    ge::GeShape old_shape(dims);
    ge::GeShape new_shape;
    fe::ShapeAndFormat shape_and_format_info2(old_shape, new_shape, origin_format, primary_format, dtype, group,
                                              extra_attr);

    fe::Status status = fe::GetShapeAccordingToFormat(shape_and_format_info2);
    EXPECT_EQ(status, expect_status);
    if (status == fe::SUCCESS) {
      EXPECT_EQ(new_shape.GetDims(), expect_dim);
      return fe::SUCCESS;
    }
    return fe::FAILED;
  }
};

TEST_F(ut_transfer_shape, transfer_shape_verify_param) {
  // RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT16, true, {}, {});
  Status ret = RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND, DT_INT8, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_EQ(ret, fe::SUCCESS);
  // RunTransferShape(ge::FORMAT_ND, ge::FORMAT_NHWC, DT_FLOAT, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  ret = RunTransferShape(ge::FORMAT_ND, ge::FORMAT_NCDHW, DT_INT32, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_UNDEFINED, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_NE(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_MAX, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_NE(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_RESERVED, ge::FORMAT_NHWC, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_NE(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_END, ge::FORMAT_NHWC, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_NE(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_RESERVED, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_NE(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_END, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_NE(ret, fe::SUCCESS);
}

TEST_F(ut_transfer_shape, transfer_shape_from_nchw) {
  // RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_FLOAT16, true, {}, {});
  // RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT16, true, {}, {});
  Status ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT, true, {5}, {5});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_INT64, true, {5, 6}, {5, 6});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});
  EXPECT_EQ(ret, fe::SUCCESS);

  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_INT8, true, {5, 6, 7, 8}, {5, 7, 8, 6});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_HWCN, DT_UINT16, true, {5, 6, 7, 8}, {7, 8, 6, 5});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_CHWN, DT_INT16, true, {5, 6, 7, 8}, {6, 7, 8, 5});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT32, true, {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_INT32, true, {5, 6, 7, 8}, {5, 7, 8, 6});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_HWCN, DT_FLOAT, true, {5, 6, 7, 8}, {7, 8, 6, 5});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_CHWN, DT_FLOAT16, true, {5, 6, 7, 8}, {6, 7, 8, 5});
  EXPECT_EQ(ret, fe::SUCCESS);

  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT8, true, {8, 512, 5, 5}, {8, 16, 5, 5, 32});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT8, true, {8, 512, 5, 5}, {8, 16, 5, 5, 32});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_FLOAT, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_FLOAT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT1, true, {8, 512, 5, 5}, {8, 2, 5, 5, 256});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT2, true, {8, 512, 5, 5}, {8, 4, 5, 5, 128});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT2, true, {8, 512, 5, 5}, {8, 4, 5, 5, 128});
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT4, true, {8, 512, 5, 5}, {8, 8, 5, 5, 64});
  EXPECT_EQ(ret, fe::SUCCESS);

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT8, true, {18, 512, 5, 5}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT8, true, {18, 512, 5, 5}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT32, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT32, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_FLOAT, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_FLOAT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT1, true, {18, 512, 5, 5}, {2, 5, 5, 2, 256, 256});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT2, true, {18, 512, 5, 5}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT2, true, {18, 512, 5, 5}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT4, true, {18, 512, 5, 5}, {8, 5, 5, 2, 64, 64});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT8, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT8, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT32, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT32, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_FLOAT, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_FLOAT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT1, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT2, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT2, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT4, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {48, 512, 5, 5}, {50, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {48, 512, 5, 5}, {200, 3, 16, 64});

  int32_t group = 16;
  ge::Format target_format =
      static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT8, true, {48, 512, 5, 5}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT8, true, {48, 512, 5, 5}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT32, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT32, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT1, true, {48, 512, 5, 5}, {800, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT2, true, {48, 512, 5, 5}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT2, true, {48, 512, 5, 5}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT4, true, {48, 512, 5, 5}, {3200, 3, 16, 64});
}

TEST_F(ut_transfer_shape, transfer_shape_from_nchw_unknow_shape) {
  Status ret =
      RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {-1, 512, 5, 5}, {-1, 32, 5, 5, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {-1, 512, -1, 5}, {-1, 32, -1, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {8, -1, 5, 5}, {8, -1, 5, 5, 16});
  // RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {-1, 512, 5, 5}, {800, -1, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, 5, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, 5, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, -1, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, -1, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {-1, -1, -1, -1}, {-1, -1, 16, 16});
}

TEST_F(ut_transfer_shape, transfer_shape_from_hwcn) {
  // RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_FLOAT16, true, {}, {});
  // RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_FLOAT16, true, {}, {});
  Status ret = RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_FLOAT, true, {5}, {5});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_INT64, true, {5, 6}, {5, 6});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT8, true, {7, 8, 6, 5}, {5, 6, 7, 8});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_INT8, true, {7, 8, 6, 5}, {5, 7, 8, 6});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_UINT16, true, {7, 8, 6, 5}, {7, 8, 6, 5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_CHWN, DT_INT16, true, {7, 8, 6, 5}, {6, 7, 8, 5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT32, true, {7, 8, 6, 5}, {5, 6, 7, 8});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_INT32, true, {7, 8, 6, 5}, {5, 7, 8, 6});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_FLOAT, true, {7, 8, 6, 5}, {7, 8, 6, 5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_CHWN, DT_FLOAT16, true, {7, 8, 6, 5}, {6, 7, 8, 5});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT8, true, {5, 5, 512, 8}, {8, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT8, true, {5, 5, 512, 8}, {8, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT32, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT32, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_FLOAT, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_FLOAT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT1, true, {5, 5, 512, 8}, {8, 2, 5, 5, 256});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT2, true, {5, 5, 512, 8}, {8, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT2, true, {5, 5, 512, 8}, {8, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT4, true, {5, 5, 512, 8}, {8, 8, 5, 5, 64});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT8, true, {5, 5, 512, 18}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT8, true, {5, 5, 512, 18}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT32, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT32, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_FLOAT, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_FLOAT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT1, true, {5, 5, 512, 18}, {2, 5, 5, 2, 256, 256});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT2, true, {5, 5, 512, 18}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT2, true, {5, 5, 512, 18}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT4, true, {5, 5, 512, 18}, {8, 5, 5, 2, 64, 64});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT8, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT8, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT32, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT32, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_FLOAT, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_FLOAT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT1, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT2, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT2, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT4, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {5, 5, 512, 48}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {5, 5, 512, 48}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {5, 5, 512, 48}, {50, 3, 16, 256});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {5, 5, 512, 48}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {5, 5, 512, 48}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {5, 5, 512, 48}, {200, 3, 16, 64});

  int32_t group = 16;
  ge::Format target_format =
      static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT8, true, {5, 5, 512, 48}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT8, true, {5, 5, 512, 48}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT32, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT32, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_FLOAT, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_FLOAT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT1, true, {5, 5, 512, 48}, {800, 3, 16, 256});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT2, true, {5, 5, 512, 48}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT2, true, {5, 5, 512, 48}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT4, true, {5, 5, 512, 48}, {3200, 3, 16, 64});
}

TEST_F(ut_transfer_shape, transfer_shape_from_ncdhw) {
  // RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {}, {});
  // RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {}, {});
  Status ret = RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT, true, {5}, {5});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT64, true, {5, 6}, {5, 6});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT8, true, {8, 512, 9, 5, 5}, {8, 9, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT8, true, {8, 512, 9, 5, 5}, {8, 9, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT32, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT32, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT1, true, {8, 512, 9, 5, 5}, {8, 9, 2, 5, 5, 256});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT2, true, {8, 512, 9, 5, 5}, {8, 9, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT2, true, {8, 512, 9, 5, 5}, {8, 9, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT4, true, {8, 512, 9, 5, 5}, {8, 9, 8, 5, 5, 64});

  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT8, true, {48, 512, 3, 5, 5}, {1200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT8, true, {48, 512, 3, 5, 5}, {1200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT32, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT32, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_FLOAT, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_FLOAT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT1, true, {48, 512, 3, 5, 5}, {150, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT2, true, {48, 512, 3, 5, 5}, {300, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT2, true, {48, 512, 3, 5, 5}, {300, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT4, true, {48, 512, 3, 5, 5}, {600, 3, 16, 64});

  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT8, true, {90, 512, 3, 5, 5},
                   {450, 16, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT8, true, {90, 512, 3, 5, 5},
                   {450, 16, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT16, true, {90, 512, 3, 5, 5},
                   {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT16, true, {90, 512, 3, 5, 5},
                   {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT32, true, {90, 512, 3, 5, 5},
                   {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT32, true, {90, 512, 3, 5, 5},
                   {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_FLOAT, true, {90, 512, 3, 5, 5},
                   {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_FLOAT16, true, {90, 512, 3, 5, 5},
                   {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT1, true, {90, 512, 3, 5, 5},
                   {450, 2, 16, 256});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT2, true, {90, 512, 3, 5, 5},
                   {450, 4, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT2, true, {90, 512, 3, 5, 5},
                   {450, 4, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT4, true, {90, 512, 3, 5, 5},
                   {450, 8, 16, 64});

  int32_t group = 16;
  ge::Format target_format =
      static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z_3D), group));
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT8, true, {48, 512, 3, 5, 5}, {19200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT8, true, {48, 512, 3, 5, 5}, {19200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT32, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT32, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_FLOAT, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_FLOAT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT1, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT2, true, {48, 512, 3, 5, 5}, {4800, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT2, true, {48, 512, 3, 5, 5}, {4800, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT4, true, {48, 512, 3, 5, 5}, {9600, 3, 16, 64});
}

TEST_F(ut_transfer_shape, transfer_shape_from_nd_to_nz) {
  Status ret = RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34}, {1, 3, 16, 16});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {18, 34}, {3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT1, true, {1, 18, 134}, {1, 1, 2, 16, 256});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT2, true, {1, 18, 134}, {1, 2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT2, true, {1, 18, 134}, {1, 2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT4, true, {1, 18, 134}, {1, 3, 2, 16, 64});
}

TEST_F(ut_transfer_shape, transfer_shape_from_nd_to_fz) {
  Status ret = RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {18, 34}, {1, 3, 16, 32});
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {18, 34}, {1, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {188, 23}, {1, 2, 16, 256});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {188, 23}, {2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {188, 23}, {2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {188, 23}, {3, 2, 16, 64});

  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {48, 512, 5, 5}, {50, 3, 16, 256});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {48, 512, 5, 5}, {200, 3, 16, 64});

  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT8, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT8, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT32, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT32, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_FLOAT, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_FLOAT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT1, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT2, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT2, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT4, true, {48, 80, 5, 5}, {6, 4, 16, 16});
}

TEST_F(ut_transfer_shape, transfer_shape_from_nd_to_zn_rnn) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  Status ret = RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128}, op_desc);
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {65, 128}, {}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {65, 128}, {}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {65, 128}, {}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {65, 128}, {}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {65, 128}, {}, op_desc);

  (void)ge::AttrUtils::SetInt(op_desc, "input_size", 30);
  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 40);
  (void)ge::AttrUtils::SetInt(op_desc, "state_size", -1);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {70, 128}, {5, 9, 16, 16}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {70, 128}, {5, 3, 16, 256}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {70, 128}, {5, 3, 16, 128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {70, 128}, {5, 3, 16, 64}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {70, 128}, {5, 6, 16, 32}, op_desc);

  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {9, 40, 128}, {9, 3, 9, 16, 16},
                   op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {9, 40, 128}, {9, 3, 3, 16, 256}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {9, 40, 128}, {9, 3, 3, 16, 128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {9, 40, 128}, {9, 3, 3, 16, 64}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {9, 40, 128}, {9, 3, 6, 16, 32}, op_desc);

  (void)ge::AttrUtils::SetInt(op_desc, "state_size", 70);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {70, 128}, {5, 9, 16, 16}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {70, 128}, {5, 3, 16, 256}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {70, 128}, {5, 3, 16, 128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {70, 128}, {5, 3, 16, 64}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {70, 128}, {5, 6, 16, 32}, op_desc);

  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {9, 100, 128}, {9, 7, 9, 16, 16},
                   op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {9, 100, 128}, {9, 7, 3, 16, 256},
                   op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {9, 100, 128}, {9, 7, 3, 16, 128},
                   op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {9, 100, 128}, {9, 7, 3, 16, 64}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {9, 100, 128}, {9, 7, 6, 16, 32}, op_desc);
}

TEST_F(ut_transfer_shape, transfer_shape_from_nd_to_nd_rnn_bias) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  Status ret = RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {}, {}, op_desc);
  EXPECT_EQ(ret, fe::SUCCESS);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {2400}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {18, 80}, {18, 1280}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT1, true, {18, 80}, {18, 20480}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT2, true, {18, 80}, {18, 10240}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT4, true, {18, 80}, {18, 5120}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 2560}, op_desc);

  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 64);
  (void)ge::AttrUtils::SetInt(op_desc, "input_size", 1);
  (void)ge::AttrUtils::SetInt(op_desc, "state_size", 1);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {}, {}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {18, 80}, {18, 64}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT1, true, {18, 80}, {18, 256}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT2, true, {18, 80}, {18, 128}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT4, true, {18, 80}, {18, 64}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 64}, op_desc);

  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 0);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {150}, op_desc);
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 80}, op_desc);
}