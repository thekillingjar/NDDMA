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
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "expand_dimension.h"
#include "transfer_shape_according_to_format.h"
#include "transfer_shape_according_to_format_ext.h"
#include "transfer_range_according_to_format.h"
#include "platform/platform_info.h"
#include "transfer_def.h"
#include "transfer_shape_utils.h"

using namespace ge;

namespace transformer {
class TransformerTransferShapeUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  void EXPECT_RunTransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &dtype,
                               const bool &expect_ret, const vector<int64_t> &dims, const vector<int64_t> &expect_dim,
                               bool only_test_first_interface = false) {
    std::cout << "EXPECT_RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << std::endl;
    ge::GeShape shape(dims);
    ShapeAndFormat shape_and_format_info {shape, origin_format, format, dtype};
    ShapeTransferAccordingToFormat shape_transfer;
    bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(shape.GetDims(), expect_dim);
    }
    if (only_test_first_interface) {
      return;
    }
    gert::Shape current_shape;
    for (const int64_t &d : dims) {
      current_shape.AppendDim(d);
    }
    gert::Shape ret_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape, ret_shape);
    if (ret && dims != expect_dim) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < ret_shape.GetDimNum(); ++i) {
        new_dim.push_back(ret_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < current_shape.GetDimNum(); ++i) {
        new_dim.push_back(current_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }
    ExtAxisValue ext_axis;
    shape_transfer.InitExtAxisValue(nullptr, ext_axis);
    ge::GeShape src_shape(dims);
    ge::GeShape dst_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape, dst_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret && dims != expect_dim) {
      EXPECT_EQ(dst_shape.GetDims(), expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(src_shape.GetDims(), expect_dim);
    }
  }

  void EXPECT_RunTransferShape(const ge::OpDescPtr &op_desc, const ge::Format &origin_format, const ge::Format &format,
                               const ge::DataType &dtype, const bool &expect_ret, const vector<int64_t> &dims,
                               const vector<int64_t> &expect_dim, bool only_test_first_interface = false,
                               const int64_t &m0_val = 16) {
    std::cout << "EXPECT_RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << std::endl;
    ge::GeShape shape(dims);
    ShapeAndFormat shape_and_format_info {shape, origin_format, format, dtype};
    ShapeTransferAccordingToFormat shape_transfer;
    bool ret = shape_transfer.GetShapeAccordingToFormat(op_desc, shape_and_format_info);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(shape.GetDims(), expect_dim);
    }
    if (only_test_first_interface) {
      return;
    }

    gert::Shape current_shape;
    for (const int64_t &d : dims) {
      current_shape.AppendDim(d);
    }

    gert::Shape ret_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape, ret_shape, op_desc);
    if (ret && dims != expect_dim) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < ret_shape.GetDimNum(); ++i) {
        new_dim.push_back(ret_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape, op_desc);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < current_shape.GetDimNum(); ++i) {
        new_dim.push_back(current_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }
    ExtAxisValue ext_axis;
    shape_transfer.InitExtAxisValue(op_desc, ext_axis);
    ext_axis[3] = m0_val;
    ge::GeShape src_shape(dims);
    ge::GeShape dst_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape, dst_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret && dims != expect_dim) {
      EXPECT_EQ(dst_shape.GetDims(), expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(src_shape.GetDims(), expect_dim);
    }
  }

  void RunTransferShapeWithExtAxis(const ge::OpDescPtr &op_desc, const ge::Format &origin_format, const ge::Format &format,
                        const ge::DataType &dtype, const bool &expect_ret, const vector<int64_t> &dims,
                        const vector<int64_t> &expect_dim, const int64_t &m0_val = 16) {
    std::cout << "EXPECT_RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << ", m0 value=" << m0_val << std::endl;

    ShapeTransferAccordingToFormat shape_transfer;
    ExtAxisValue ext_axis;
    shape_transfer.InitExtAxisValue(op_desc, ext_axis);
    ext_axis[3] = m0_val;
    ge::GeShape src_shape(dims);
    ge::GeShape dst_shape;
    bool ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape, dst_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret && dims != expect_dim) {
      EXPECT_EQ(dst_shape.GetDims(), expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(src_shape.GetDims(), expect_dim);
    }
  }
};

TEST_F(TransformerTransferShapeUT, transfer_shape_verify_param) {
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND, DT_INT8, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_NHWC, DT_FLOAT, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_NCDHW, DT_INT32, true, {3, 4, 5, 6}, {3, 4, 5, 6});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_UNDEFINED, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_MAX, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_RESERVED, ge::FORMAT_NHWC, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_END, ge::FORMAT_NHWC, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_RESERVED, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_END, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nchw) {
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT, true, {5}, {5});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_INT64, true, {5, 6}, {5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_INT8, true, {5, 6, 7, 8}, {5, 7, 8, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_HWCN, DT_UINT16, true, {5, 6, 7, 8}, {7, 8, 6, 5});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_CHWN, DT_INT16, true, {5, 6, 7, 8}, {6, 7, 8, 5});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT32, true, {5, 6, 7, 8}, {5, 6, 7, 8});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_INT32, true, {5, 6, 7, 8}, {5, 7, 8, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_HWCN, DT_FLOAT, true, {5, 6, 7, 8}, {7, 8, 6, 5});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_CHWN, DT_FLOAT16, true, {5, 6, 7, 8}, {6, 7, 8, 5});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT8, true, {8, 512, 5, 5}, {8, 16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT8, true, {8, 512, 5, 5}, {8, 16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_FLOAT, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_FLOAT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT1, true, {8, 512, 5, 5}, {8, 2, 5, 5, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT2, true, {8, 512, 5, 5}, {8, 4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT2, true, {8, 512, 5, 5}, {8, 4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT4, true, {8, 512, 5, 5}, {8, 8, 5, 5, 64});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_UINT8, true, {512, 1, 5, 5}, {16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_INT8, true, {512, 1, 5, 5}, {16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_UINT16, true, {512, 1, 5, 5}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_INT16, true, {512, 1, 5, 5}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_UINT32, true, {512, 1, 5, 5}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_INT32, true, {512, 1, 5, 5}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_FLOAT, true, {512, 1, 5, 5}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_FLOAT16, true, {512, 1, 5, 5}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_UINT1, true, {512, 1, 5, 5}, {2, 5, 5, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_UINT2, true, {512, 1, 5, 5}, {4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_INT2, true, {512, 1, 5, 5}, {4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, DT_INT4, true, {512, 1, 5, 5}, {8, 5, 5, 64});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT8, true, {18, 512, 5, 5}, {16, 5, 5, 2, 32, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT8, true, {18, 512, 5, 5}, {16, 5, 5, 2, 32, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT32, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT32, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_FLOAT, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_FLOAT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT1, true, {18, 512, 5, 5}, {2, 5, 5, 2, 256, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT2, true, {18, 512, 5, 5}, {4, 5, 5, 2, 128, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT2, true, {18, 512, 5, 5}, {4, 5, 5, 2, 128, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT4, true, {18, 512, 5, 5}, {8, 5, 5, 2, 64, 64});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT8, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT8, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT32, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT32, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_FLOAT, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_FLOAT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT1, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT2, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT2, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT4, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {48, 512, 5, 5}, {50, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {48, 512, 5, 5}, {200, 3, 16, 64});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT8, true, {48, 3, 5, 5}, {4, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT8, true, {48, 3, 5, 5}, {4, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT16, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT16, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT32, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT32, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT1, true, {48, 3, 5, 5}, {1, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT2, true, {48, 3, 5, 5}, {1, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT2, true, {48, 3, 5, 5}, {1, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT4, true, {48, 3, 5, 5}, {2, 3, 16, 64});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT8, true, {48, 512, 5, 5}, {6400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT8, true, {48, 512, 5, 5}, {6400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT32, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT32, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT1, true, {48, 512, 5, 5}, {800, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT2, true, {48, 512, 5, 5}, {1600, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT2, true, {48, 512, 5, 5}, {1600, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT4, true, {48, 512, 5, 5}, {3200, 3, 16, 64});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nchw_unknow_shape) {
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {-1, 512, 5, 5}, {-1, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {-1, 512, -1, 5}, {-1, 32, -1, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {8, -1, 5, 5}, {8, -1, 5, 5, 16});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {-1, 33, 5, 5}, {-1, 9, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {-1, 33, -1, 5}, {-1, 9, -1, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {8, -1, 5, 5}, {8, -1, 5, 5, 4});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {-1, 33, 5, 5}, {75, -1, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, 5, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, -1, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, 5, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, -1, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, -1, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, -1, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {-1, -1, -1, -1}, {-1, -1, 16, 16});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, 512, -1, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {-1, 512, 5, 5}, {800, -1, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, -1, 5, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {-1, -1, 5, 5}, {-1, -1, 16, 16});

  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {-1, 3, 5, 5}, {7, -1, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, 5, 5}, {7, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, -1, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, 5, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, -1, 5}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, 5, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, -1, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, -1, -1}, {-1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {-1, -1, -1, -1}, {-1, -1, 16, 16});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_hwcn) {
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_FLOAT, true, {5}, {5});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_INT64, true, {5, 6}, {5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT8, true, {7, 8, 6, 5}, {5, 6, 7, 8});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_INT8, true, {7, 8, 6, 5}, {5, 7, 8, 6});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_UINT16, true, {7, 8, 6, 5}, {7, 8, 6, 5});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_CHWN, DT_INT16, true, {7, 8, 6, 5}, {6, 7, 8, 5});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT32, true, {7, 8, 6, 5}, {5, 6, 7, 8});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_INT32, true, {7, 8, 6, 5}, {5, 7, 8, 6});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_FLOAT, true, {7, 8, 6, 5}, {7, 8, 6, 5});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_CHWN, DT_FLOAT16, true, {7, 8, 6, 5}, {6, 7, 8, 5});

  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT8, true, {5, 5, 512, 8}, {8, 16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT8, true, {5, 5, 512, 8}, {8, 16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT32, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT32, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_FLOAT, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_FLOAT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT1, true, {5, 5, 512, 8}, {8, 2, 5, 5, 256});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT2, true, {5, 5, 512, 8}, {8, 4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT2, true, {5, 5, 512, 8}, {8, 4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT4, true, {5, 5, 512, 8}, {8, 8, 5, 5, 64});

  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_UINT8, true, {5, 5, 1, 512}, {16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_INT8, true, {5, 5, 1, 512}, {16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_UINT16, true, {5, 5, 1, 512}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_INT16, true, {5, 5, 1, 512}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_UINT32, true, {5, 5, 1, 512}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_INT32, true, {5, 5, 1, 512}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_FLOAT, true, {5, 5, 1, 512}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_FLOAT16, true, {5, 5, 1, 512}, {32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_UINT1, true, {5, 5, 1, 512}, {2, 5, 5, 256});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_UINT2, true, {5, 5, 1, 512}, {4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_INT2, true, {5, 5, 1, 512}, {4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, DT_INT4, true, {5, 5, 1, 512}, {8, 5, 5, 64});

  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT8, true, {5, 5, 512, 18}, {16, 5, 5, 2, 32, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT8, true, {5, 5, 512, 18}, {16, 5, 5, 2, 32, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT32, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT32, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_FLOAT, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_FLOAT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT1, true, {5, 5, 512, 18}, {2, 5, 5, 2, 256, 256});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT2, true, {5, 5, 512, 18}, {4, 5, 5, 2, 128, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT2, true, {5, 5, 512, 18}, {4, 5, 5, 2, 128, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT4, true, {5, 5, 512, 18}, {8, 5, 5, 2, 64, 64});

  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT8, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT8, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT32, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT32, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_FLOAT, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_FLOAT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT1, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT2, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT2, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT4, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});

  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {5, 5, 512, 48}, {400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {5, 5, 512, 48}, {400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {5, 5, 512, 48}, {50, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {5, 5, 512, 48}, {100, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {5, 5, 512, 48}, {100, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {5, 5, 512, 48}, {200, 3, 16, 64});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT8, true, {5, 5, 512, 48}, {6400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT8, true, {5, 5, 512, 48}, {6400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT32, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT32, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_FLOAT, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_FLOAT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT1, true, {5, 5, 512, 48}, {800, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT2, true, {5, 5, 512, 48}, {1600, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT2, true, {5, 5, 512, 48}, {1600, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT4, true, {5, 5, 512, 48}, {3200, 3, 16, 64});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_ncdhw) {
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT, true, {5}, {5});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT64, true, {5, 6}, {5, 6});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT8, true, {8, 512, 9, 5, 5}, {8, 9, 16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT8, true, {8, 512, 9, 5, 5}, {8, 9, 16, 5, 5, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT32, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT32, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT1, true, {8, 512, 9, 5, 5}, {8, 9, 2, 5, 5, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT2, true, {8, 512, 9, 5, 5}, {8, 9, 4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT2, true, {8, 512, 9, 5, 5}, {8, 9, 4, 5, 5, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT4, true, {8, 512, 9, 5, 5}, {8, 9, 8, 5, 5, 64});

  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {8, 9, 5, 15, 512});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {9, 5, 15, 512, 8});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_DHWNC, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {9, 5, 15, 8, 512});

  EXPECT_RunTransferShape(ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {8, 15, 512, 9, 5});
  EXPECT_RunTransferShape(ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {512, 9, 5, 15, 8});
  EXPECT_RunTransferShape(ge::FORMAT_NDHWC, ge::FORMAT_DHWNC, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {512, 9, 5, 8, 15});

  EXPECT_RunTransferShape(ge::FORMAT_DHWCN, ge::FORMAT_NCDHW, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {15, 5, 8, 512, 9});
  EXPECT_RunTransferShape(ge::FORMAT_DHWCN, ge::FORMAT_NDHWC, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {15, 8, 512, 9, 5});
  EXPECT_RunTransferShape(ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {8, 512, 9, 15, 5});

  EXPECT_RunTransferShape(ge::FORMAT_DHWNC, ge::FORMAT_NCDHW, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {5, 15, 8, 512, 9});
  EXPECT_RunTransferShape(ge::FORMAT_DHWNC, ge::FORMAT_NDHWC, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {5, 8, 512, 9, 15});
  EXPECT_RunTransferShape(ge::FORMAT_DHWNC, ge::FORMAT_DHWCN, DT_FLOAT16, true, {8, 512, 9, 5, 15}, {8, 512, 9, 15, 5});

  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT8, true, {48, 512, 3, 5, 5}, {1200, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT8, true, {48, 512, 3, 5, 5}, {1200, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT32, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT32, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_FLOAT, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_FLOAT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT1, true, {48, 512, 3, 5, 5}, {150, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT2, true, {48, 512, 3, 5, 5}, {300, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT2, true, {48, 512, 3, 5, 5}, {300, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT4, true, {48, 512, 3, 5, 5}, {600, 3, 16, 64});

  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT8, true, {90, 512, 3, 5, 5}, {450, 16, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT8, true, {90, 512, 3, 5, 5}, {450, 16, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT16, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT16, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT32, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT32, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_FLOAT, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_FLOAT16, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT1, true, {90, 512, 3, 5, 5}, {450, 2, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT2, true, {90, 512, 3, 5, 5}, {450, 4, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT2, true, {90, 512, 3, 5, 5}, {450, 4, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT4, true, {90, 512, 3, 5, 5}, {450, 8, 16, 64});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z_3D), group));
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT8, true, {48, 512, 3, 5, 5}, {19200, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT8, true, {48, 512, 3, 5, 5}, {19200, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT32, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT32, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_FLOAT, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_FLOAT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT1, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT2, true, {48, 512, 3, 5, 5}, {4800, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT2, true, {48, 512, 3, 5, 5}, {4800, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT4, true, {48, 512, 3, 5, 5}, {9600, 3, 16, 64});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_4d_to_6hd) {
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {4, 33, 7, 7}, {4, 1, 3, 7, 7, 16});
  EXPECT_RunTransferShape(ge::FORMAT_NHWC, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {4, 7, 7, 33}, {4, 1, 3, 7, 7, 16});
  EXPECT_RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {7, 7, 33, 4}, {4, 1, 3, 7, 7, 16});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_nz) {
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34}, {1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34, 1}, {1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {18, 34}, {3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT1, true, {1, 18, 134}, {1, 1, 2, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT2, true, {1, 18, 134}, {1, 2, 2, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT2, true, {1, 18, 134}, {1, 2, 2, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT4, true, {1, 18, 134}, {1, 3, 2, 16, 64});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {-2}, {-2}, true);
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {8, 1000}, {63, 1, 16, 16});

  transformer::TransferShapeUtils::m0_list_.fill(1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34}, {1, 34, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34, 1}, {1, 34, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {18, 34}, {3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT8, true, {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT8, true, {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT16, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT16, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT32, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT32, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT1, true, {1, 18, 134}, {1, 1, 18, 1, 256}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT2, true, {1, 18, 134}, {1, 2, 18, 1, 128}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT2, true, {1, 18, 134}, {1, 2, 18, 1, 128}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT4, true, {1, 18, 134}, {1, 3, 18, 1, 64}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {8, 1000}, {63, 8, 1, 16}, 1);
  transformer::TransferShapeUtils::m0_list_.fill(16);
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_nz_C0_16) {
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true, {34}, {1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true, {34, 1}, {1, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT, true, {18, 34}, {3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT8, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT8, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT16, true,
                          {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT32, true,
                          {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true,
                          {1, 18, 34}, {1, 3, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT1, true,
                          {1, 18, 134}, {1, 9, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT2, true,
                          {1, 18, 134}, {1, 9, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT2, true,
                          {1, 18, 134}, {1, 9, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT4, true, {1, 18, 134}, {1, 9, 2, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true, {-2}, {-2}, true);
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true, {8, 1000}, {63, 1, 16, 16});

  transformer::TransferShapeUtils::m0_list_.fill(1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true,
                              {34}, {1, 34, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true,
                              {34, 1}, {1, 34, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT, true,
                              {18, 34}, {3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT8, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT8, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT16, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT16, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT32, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT32, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true,
                              {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT1, true,
                              {1, 18, 134}, {1, 9, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_UINT2, true,
                              {1, 18, 134}, {1, 9, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT2, true,
                              {1, 18, 134}, {1, 9, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_16, DT_INT4, true,
                              {1, 18, 134}, {1, 9, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ_C0_16, DT_FLOAT16, true,
                              {8, 1000}, {63, 8, 1, 16}, 1);
  transformer::TransferShapeUtils::m0_list_.fill(16);
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_nz_C0_32) {
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true, {34}, {1, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true, {34, 1}, {1, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT, true, {18, 34}, {2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT16, true,
                          {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT16, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT32, true,
                          {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT32, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true,
                          {1, 18, 34}, {1, 2, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT1, true,
                          {1, 18, 134}, {1, 5, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT2, true,
                          {1, 18, 134}, {1, 5, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT2, true,
                          {1, 18, 134}, {1, 5, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT4, true, {1, 18, 134}, {1, 5, 2, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true, {-2}, {-2}, true);
  EXPECT_RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true, {8, 1000}, {32, 1, 16, 32});

  transformer::TransferShapeUtils::m0_list_.fill(1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true,
                              {34}, {1, 34, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true,
                              {34, 1}, {1, 34, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT, true,
                              {18, 34}, {2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT8, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT8, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT16, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT16, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT32, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT32, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true,
                              {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT1, true,
                              {1, 18, 134}, {1, 5, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_UINT2, true,
                              {1, 18, 134}, {1, 5, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT2, true,
                              {1, 18, 134}, {1, 5, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32, DT_INT4, true,
                              {1, 18, 134}, {1, 5, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ_C0_32, DT_FLOAT16, true,
                              {8, 1000}, {32, 8, 1, 32}, 1);
  transformer::TransferShapeUtils::m0_list_.fill(16);
  transformer::TransferShapeUtils::GetC0Value(DT_FLOAT, ge::FORMAT_FRACTAL_NZ_C0_2);
  transformer::TransferShapeUtils::GetC0Value(DT_FLOAT, ge::FORMAT_FRACTAL_NZ_C0_8);
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_fz) {
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {18, 34}, {1, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {18, 34}, {1, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {18, 34}, {2, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {18, 34}, {2, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {18, 34}, {2, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {18, 34}, {2, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {18, 34}, {2, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {18, 34}, {2, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {188, 23}, {1, 2, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {188, 23}, {2, 2, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {188, 23}, {2, 2, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {188, 23}, {3, 2, 16, 64});

  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {48, 512, 5, 5}, {50, 3, 16, 256});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {48, 512, 5, 5}, {200, 3, 16, 64});

  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT8, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT8, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT32, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT32, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_FLOAT, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_FLOAT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT1, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT2, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT2, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  EXPECT_RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT4, true, {48, 80, 5, 5}, {6, 4, 16, 16});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_zn_rnn) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {65, 128}, {65, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {65, 128}, {65, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {65, 128}, {65, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {65, 128}, {65, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {65, 128}, {65, 128});

  (void)ge::AttrUtils::SetInt(op_desc, "input_size", 30);
  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 40);
  (void)ge::AttrUtils::SetInt(op_desc, "state_size", -1);
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {70, 128}, {5, 9, 16, 16});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {70, 128}, {5, 3, 16, 256});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {70, 128}, {5, 3, 16, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {70, 128}, {5, 3, 16, 64});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {70, 128}, {5, 6, 16, 32});

  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {9, 40, 128},
                   {9, 3, 9, 16, 16});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {9, 40, 128}, {9, 3, 3, 16, 256});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {9, 40, 128}, {9, 3, 3, 16, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {9, 40, 128}, {9, 3, 3, 16, 64});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {9, 40, 128}, {9, 3, 6, 16, 32});

  (void)ge::AttrUtils::SetInt(op_desc, "state_size", 70);
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {70, 128}, {5, 9, 16, 16});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {70, 128}, {5, 3, 16, 256});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {70, 128}, {5, 3, 16, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {70, 128}, {5, 3, 16, 64});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {70, 128}, {5, 6, 16, 32});

  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {9, 100, 128},
                   {9, 7, 9, 16, 16});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {9, 100, 128},
                   {9, 7, 3, 16, 256});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {9, 100, 128},
                   {9, 7, 3, 16, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {9, 100, 128}, {9, 7, 3, 16, 64});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {9, 100, 128}, {9, 7, 6, 16, 32});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_nd_rnn_bias) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {2400});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {18, 80}, {18, 1280});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT1, true, {18, 80}, {18, 20480});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT2, true, {18, 80}, {18, 10240});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT4, true, {18, 80}, {18, 5120});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 2560});

  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 64);
  (void)ge::AttrUtils::SetInt(op_desc, "input_size", 1);
  (void)ge::AttrUtils::SetInt(op_desc, "state_size", 1);
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {}, {});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {18, 80}, {18, 64});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT1, true, {18, 80}, {18, 256});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT2, true, {18, 80}, {18, 128});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT4, true, {18, 80}, {18, 64});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 64});

  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 0);
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {150});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 80});
  EXPECT_RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {-2}, {-2}, true);
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nyuva) {
    ShapeTransferAccordingToFormat shape_transfer;
    gert::Shape current_shape;
    vector<int64_t> dims = {42, 63, 3};
    vector<int64_t> expect_dim = {48, 64, 3};
    for (const int64_t &d : dims) {
      current_shape.AppendDim(d);
    }
    bool ret = shape_transfer.TransferShape(ge::FORMAT_NYUV, ge::FORMAT_NYUV_A, DT_INT8, current_shape);
    EXPECT_EQ(ret, true);
    if (ret) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < current_shape.GetDimNum(); ++i) {
        new_dim.push_back(current_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_1) {
  ge::Format src_format = ge::FORMAT_ND;
  gert::Shape src_shape({1});
  ge::Format dst_format = ge::FORMAT_ND;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp1);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_2) {
  ge::Format src_format = ge::FORMAT_ND;
  gert::Shape src_shape({1, 16});
  ge::Format dst_format = ge::FORMAT_FRACTAL_NZ;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({16, 16}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{1}, {0}};
  std::vector<std::vector<int32_t>> tmp2 = {{1}, {0}, {-1}, {-1}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_3) {
  ge::Format src_format = ge::FORMAT_ND;
  gert::Shape src_shape({1, 16, 64});
  ge::Format dst_format = ge::FORMAT_FRACTAL_NZ;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 16, 16}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0}, {2}, {1}};
  std::vector<std::vector<int32_t>> tmp2 = {{0}, {2}, {1}, {-1}, {-1}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_4) {
  ge::Format src_format = ge::FORMAT_ND;
  gert::Shape src_shape({1, 16, 64});
  ge::Format dst_format = ge::FORMAT_FRACTAL_Z;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, false);
  EXPECT_EQ(aligned_shape, gert::Shape({}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, false);
  std::vector<std::vector<int32_t>> tmp1 = {};
  std::vector<std::vector<int32_t>> tmp2 = {};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_5) {
  ge::Format src_format = ge::FORMAT_NCHW;
  gert::Shape src_shape({1, 16, 64});
  ge::Format dst_format = ge::FORMAT_NCHW;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 1, 1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0}, {1}, {2}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp1);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_6) {
  ge::Format src_format = ge::FORMAT_NCHW;
  gert::Shape src_shape({1, 16, 64, 128});
  ge::Format dst_format = ge::FORMAT_HWCN;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 1, 1, 1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{3}, {2}, {0}, {1}};
  std::vector<std::vector<int32_t>> tmp2 = {{2}, {3}, {1}, {0}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_7) {
  ge::Format src_format = ge::FORMAT_NCHW;
  gert::Shape src_shape({1, 16, 64, 128});
  ge::Format dst_format = ge::FORMAT_NC1HWC0;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 16, 1, 1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0}, {1, 4}, {2}, {3}};
  std::vector<std::vector<int32_t>> tmp2 = {{0}, {1}, {2}, {3}, {1}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_8) {
  ge::Format src_format = ge::FORMAT_NCHW;
  gert::Shape src_shape({1, 16});
  ge::Format dst_format = ge::FORMAT_NC1HWC0;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 9;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({16, 1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{1, 4}, {2}};
  std::vector<std::vector<int32_t>> tmp2 = {{-1}, {0}, {1}, {-1}, {0}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_9) {
  ge::Format src_format = ge::FORMAT_NCHW;
  gert::Shape src_shape({1, 16});
  ge::Format dst_format = ge::FORMAT_NC1HWC0;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 3;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{2}, {3}};
  std::vector<std::vector<int32_t>> tmp2 = {{-1}, {-1}, {0}, {1}, {-1}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}
TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_10) {
  ge::Format src_format = ge::FORMAT_NDHWC;
  gert::Shape src_shape({1, 16, 32, 64, 128});
  ge::Format dst_format = ge::FORMAT_NCDHW;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 1, 1, 1, 1}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0}, {2}, {3}, {4}, {1}};
  std::vector<std::vector<int32_t>> tmp2 = {{0}, {4}, {1}, {2}, {3}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_11) {
  ge::Format src_format = ge::FORMAT_NDHWC;
  gert::Shape src_shape({1, 16});
  ge::Format dst_format = ge::FORMAT_NDC1HWC0;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 7;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 16}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{4}, {2, 5}};
  std::vector<std::vector<int32_t>> tmp2 = {{-1}, {-1}, {1}, {-1}, {0}, {1}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_12) {
  ge::Format src_format = ge::FORMAT_HWCN;
  gert::Shape src_shape({1, 2, 3, 16});
  ge::Format dst_format = ge::FORMAT_FRACTAL_Z;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 0;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 1, 16, 16}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0}, {0}, {0, 3}, {1, 2}};
  std::vector<std::vector<int32_t>> tmp2 = {{2, 0, 1}, {3}, {3}, {2}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

TEST_F(TransformerTransferShapeUT, get_aligned_shape_and_transfer_dim_13) {
  ge::Format src_format = ge::FORMAT_HWCN;
  gert::Shape src_shape({1, 16});
  ge::Format dst_format = ge::FORMAT_FRACTAL_Z;
  ge::DataType data_type = ge::DT_FLOAT16;
  int64_t reshape_type_mask = 6;
  gert::Shape aligned_shape;
  transformer::AlignShapeInfo align_shape_info = {src_format, dst_format, src_shape, data_type, reshape_type_mask};
  bool ret = transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(aligned_shape, gert::Shape({1, 16}));
  transformer::TransferDimsInfo transfor_dims_info = {src_format, dst_format, src_shape, reshape_type_mask};
  transformer::AxisIndexMapping axis_index_mapping;
  ret = transformer::TransferShapeUtils::TransferDims(transfor_dims_info, axis_index_mapping);
  EXPECT_EQ(ret, true);
  std::vector<std::vector<int32_t>> tmp1 = {{0},{1, 2}};
  std::vector<std::vector<int32_t>> tmp2 = {{-1, 0, -1}, {1}, {1}, {-1}};
  EXPECT_EQ(axis_index_mapping.src_to_dst_transfer_dims, tmp1);
  EXPECT_EQ(axis_index_mapping.dst_to_src_transfer_dims, tmp2);
}

// TEST_F(TransformerTransferShapeUT, ext_transfershape_case1) {
//   ShapeTransferAccordingToFormatExt shape_transfer_ext;
//   ExtAxisOpValue op_value = {1, 1, 1};
//   gert::Shape shape_1 = {3, 4, 5, 6};
//   fe::PlatFormInfos platform_infos;
//   shape_transfer_ext.TransferShape(ge::FORMAT_ND, ge::FORMAT_ND, DT_INT8, shape_1, op_value, &platform_infos);
// }

// TEST_F(TransformerTransferShapeUT, ext_transfershape_case2) {
//   ShapeTransferAccordingToFormatExt shape_transfer_ext;
//   ExtAxisOpValue op_value = {1, 1, 1};
//   gert::Shape shape_1 = {3, 4, 5, 6};
//   gert::Shape shape_2 = {3, 4, 5, 6};
//   shape_transfer_ext.TransferShape(ge::FORMAT_ND, ge::FORMAT_ND, DT_INT8, shape_1, shape_2, op_value);
// }
}  // namespace ge
