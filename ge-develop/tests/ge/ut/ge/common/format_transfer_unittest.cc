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

#include "formats/format_transfers/format_transfer_nchw_nc1hwc0.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace formats {

class UtestFormatTransfer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFormatTransfer, build_transfer_success) {
  uint8_t data[1 * 3 * 224 * 224 * 2];
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NCHW, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  TransArgs args{data, src_format, dst_format, FORMAT_NCHW, FORMAT_NC1HWC0, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                 {1, 3, 224, 224}, {1, 1, 224, 224, 16}, DT_FLOAT16};
  auto transfer = BuildFormatTransfer(args);
  EXPECT_NE(transfer, nullptr);
}

TEST_F(UtestFormatTransfer, build_unsupported_transfer) {
  uint8_t data[1 * 3 * 224 * 224 * 2];
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_RESERVED, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NCHW, FORMAT_RESERVED, 5));
  TransArgs args1{data, src_format, dst_format, FORMAT_RESERVED, FORMAT_NCHW, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                  {1, 1, 224, 224, 16}, {1, 3, 224, 224}, DT_FLOAT16};
  auto transfer1 = BuildFormatTransfer(args1);
  EXPECT_EQ(transfer1, nullptr);

  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format2 = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NCHW, FORMAT_RESERVED, 5));
  const Format dst_format2 = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_RESERVED, FORMAT_RESERVED, 5));
  TransArgs args2{data, src_format2, dst_format2, FORMAT_NCHW, FORMAT_RESERVED, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                  {1, 3, 224, 224}, {1, 1, 224, 224, 16}, DT_FLOAT16};
  auto transfer2 = BuildFormatTransfer(args2);
  EXPECT_EQ(transfer2, nullptr);
}
}  // namespace formats
}  // namespace ge
