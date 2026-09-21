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

#include "formats/format_transfers/format_transfer_dhwnc_fractal_z_3D_transpose.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace formats {

class UtestFormatTransferDhwncFractalZ3DTranspose : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFormatTransferDhwncFractalZ3DTranspose, dhwcn_invalid_data_type_string) {
  uint8_t data[1 * 4 * 4 * 1 * 16 * 16] = {1};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_DHWNC, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED, 5));
  TransArgs args{data, src_format, dst_format, FORMAT_DHWNC, FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED,
                 FORMAT_RESERVED, 16, 16, {1, 4, 4, 1, 16, 16}, {4, 4, 3, 1}, DT_STRING};
  TransResult result;

  FormatTransferDhwncFractalZ3DTranspose transfer;
  Status ret = transfer.TransFormat(args, result);
  EXPECT_EQ(ret, ACL_ERROR_GE_DATATYPE_INVALID);
}

TEST_F(UtestFormatTransferDhwncFractalZ3DTranspose, dhwcn_data_type_uint8_invalid_shape) {
  uint8_t data[1 * 4 * 4 * 1 * 16 * 16] = {1};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 6 indicates that cube size is 32
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_DHWNC, FORMAT_RESERVED, 6));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED, 6));
  TransArgs args{data, src_format, dst_format, FORMAT_DHWNC, FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED,
                 FORMAT_RESERVED,  32, 32, {1, 4, 4, 1, 16, 16}, {4, 4, 3, 1}, DT_UINT8};
  TransResult result;

  FormatTransferDhwncFractalZ3DTranspose transfer;
  Status ret = transfer.TransFormat(args, result);
  EXPECT_EQ(ret, ACL_ERROR_GE_SHAPE_INVALID);
}

TEST_F(UtestFormatTransferDhwncFractalZ3DTranspose, dhwcn_data_type_uint8_invalid_format) {
  uint8_t data[1 * 4 * 4 * 1 * 16 * 16] = {1};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 6 indicates that cube size is 32
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_DHWNC, FORMAT_RESERVED, 6));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_DHWCN, FORMAT_RESERVED, 6));
  TransArgs args{data, src_format, dst_format, FORMAT_DHWCN, FORMAT_DHWCN, FORMAT_RESERVED, FORMAT_RESERVED, 32, 32,
                 {1, 4, 4, 1, 16, 16}, {4, 4, 3, 1}, DT_UINT8};
  TransResult result;

  FormatTransferDhwncFractalZ3DTranspose transfer;
  Status ret = transfer.TransFormat(args, result);
  EXPECT_EQ(ret, ACL_ERROR_GE_FORMAT_INVALID);
}

TEST_F(UtestFormatTransferDhwncFractalZ3DTranspose, dhwcn_data_type_uint8_dst_invalid_shape) {
  uint8_t data[1 * 4 * 4 * 16 * 16] = {1};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 6 indicates that cube size is 32
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_DHWNC, FORMAT_RESERVED, 6));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED, 6));
  TransArgs args{data, src_format, dst_format, FORMAT_DHWNC, FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED,
                 FORMAT_RESERVED, 32, 32, {1, 4, 4, 16, 16}, {4, 4, 3, 1}, DT_UINT8};
  TransResult result;

  FormatTransferDhwncFractalZ3DTranspose transfer;
  Status ret = transfer.TransFormat(args, result);
  EXPECT_EQ(ret, ACL_ERROR_GE_SHAPE_INVALID);
}

TEST_F(UtestFormatTransferDhwncFractalZ3DTranspose, dhwcn_data_type_uint8_success) {
  uint8_t data[1 * 4 * 4 * 16 * 16] = {1};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 6 indicates that cube size is 32
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_DHWNC, FORMAT_RESERVED, 6));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED, 6));
  TransArgs args{data, src_format, dst_format, FORMAT_DHWNC, FORMAT_FRACTAL_Z_3D_TRANSPOSE, FORMAT_RESERVED,
                 FORMAT_RESERVED, 32, 32, {1, 4, 4, 16, 16}, {16, 1, 16, 32}, DT_UINT8};
  TransResult result;

  FormatTransferDhwncFractalZ3DTranspose transfer;
  Status ret = transfer.TransFormat(args, result);
  EXPECT_EQ(ret, SUCCESS);
}

}  // namespace formats
}  // namespace ge
