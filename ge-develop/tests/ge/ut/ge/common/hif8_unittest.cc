/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>

#include "common/math/hif8_t.h"

namespace ge {

class UtestHiF8: public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestHiF8, fp32_to_hif8) {
  for (uint16_t u8 = 0; u8 <= 255; u8++) {
    HiF8 hif8 = HiF8::FromRawBits(u8);
    EXPECT_EQ(hif8, HiF8(static_cast<float>(hif8)));
  }
}

TEST_F(UtestHiF8, fp16_to_hif8) {
  for (uint16_t u8 = 0; u8 <= 255; u8++) {
    HiF8 hif8 = HiF8::FromRawBits(u8);
    EXPECT_EQ(hif8, HiF8(static_cast<fp16_t>(hif8)));
  }
}

TEST_F(UtestHiF8, fp32_to_hif8_golden) {
  std::vector<std::pair<uint32_t, uint8_t>> golden = {
    { 0b00110110001110001100001010010000, 0b00000100 },
    { 0b00110110100110001010111010110110, 0b00000101 },
    { 0b00110110111001010001010111011101, 0b00000110 },
    { 0b00110111000111000010111111111000, 0b00000110 },
    { 0b00110101100010110100111101100111, 0b00000011 },

    { 0b00111111010010001000010010011110, 0b00011101 },
    { 0b00111100111111011100011111001101, 0b01010100 },
    { 0b00111111000100010010101000101100, 0b00011001 },
    { 0b00111111011001010101100101100101, 0b00011110 },
    { 0b00111110001000000100000011101101, 0b00111010 },

    { 0b01000011010000010001001110000011, 0b01001110 },
    { 0b01000011100011100000111000100001, 0b01100000 },
    { 0b01000010001101010110001110001001, 0b01000110 },
    { 0b01000011101100110000100011011011, 0b01100001 },
    { 0b01000010010100110110101100100101, 0b01000111 },

    { 0b01000110100001101100011111100111, 0b01101100 },
    { 0b01000110111000011100101001001010, 0b01101110 },
    { 0b01000101001111010110110100011011, 0b01100111 },
    { 0b01000110001010110000101110111111, 0b01101011 },
    { 0b01000111010000110101000001011110, 0b01101111 },
  };

  for (auto pair : golden) {
    EXPECT_EQ(HiF8::BitsFromFp32(pair.first), pair.second);
  }
}

TEST_F(UtestHiF8, fp16_to_hif8_golden) {
  std::vector<std::pair<uint16_t, uint8_t>> golden = {
    { 0b0000000000000101, 0b00000001 },
    { 0b0000000000001000, 0b00000010 },
    { 0b0000000000101100, 0b00000100 },
    { 0b0000000000110011, 0b00000101 },
    { 0b0000000010001101, 0b00000110 },

    { 0b0010100010011011, 0b01010101 },
    { 0b0011011100010110, 0b00110110 },
    { 0b0010111001111001, 0b01010010 },
    { 0b0011101101101001, 0b00011111 },
    { 0b0011100011111100, 0b00011010 },

    { 0b0110011111011111, 0b01100110 },
    { 0b0110010011110011, 0b01100100 },
    { 0b0110010110001101, 0b01100101 },
    { 0b0110000001110000, 0b01100010 },
    { 0b0110100100011111, 0b01100111 },

    { 0b0111100001110001, 0b01101110 },
    { 0b0111011110101110, 0b01101110 },
    { 0b0111010001101110, 0b01101100 },
    { 0b0111000100011011, 0b01101011 },
    { 0b0111100111101010, 0b01101111 },
  };

  for (auto pair : golden) {
    EXPECT_EQ(HiF8::BitsFromFp16(pair.first), pair.second);
  }
}
}  // namespace ge
