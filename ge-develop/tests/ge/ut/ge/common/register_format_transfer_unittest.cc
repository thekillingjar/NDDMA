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
#include "formats/register_format_transfer.h"

namespace ge {
  class UtestRegisterFormatTransfer : public testing::Test {
  protected:
    void SetUp() {}
    void TearDown() {}
  };

TEST_F(UtestRegisterFormatTransfer, FormatTransferExists_failed) {
  uint16_t data_5d[1 * 1 * 1 * 1 * 16] = { 13425, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NCHW, FORMAT_RESERVED, 5));
  formats::TransArgs args{
    reinterpret_cast<uint8_t *>(data_5d), src_format, dst_format, FORMAT_NC1HWC0, FORMAT_NCHW, FORMAT_RESERVED,
    FORMAT_RESERVED, 16, 16, {1, 1, 1, 1, 16}, {1, 1, 1, 1}, DT_FLOAT16};
  EXPECT_TRUE(formats::FormatTransferExists(args));
}
}
