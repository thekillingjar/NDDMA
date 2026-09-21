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
#include <ctime>

#include "formats/format_transfers/format_transfer_fractal_z_tbe.h"
#include "formats/formats.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace formats {
class UtestFormatTransferFractalZTbe : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFormatTransferFractalZTbe, success_trans_format) {
  uint16_t data[2 * 2 * 17 * 4] = {
      12210, 13522, 12430, 13868, 14463, 12261, 12394, 15327, 14988, 14663, 14310, 12308, 14886, 15036, 13655, 12774,
      13715, 13322, 13198, 14931, 14944, 10231, 14824, 14512, 14493, 14936, 14513, 14481, 13061, 14808, 14637, 13011,
      15351, 15277, 13709, 9313,  14684, 14460, 14576, 13978, 14945, 13652, 14162, 12974, 11122, 15207, 14677, 12431,
      14361, 14347, 14675, 12983, 14020, 13429, 13678, 14861, 14016, 13590, 13322, 9523,  10130, 15338, 11862, 15194,
      14545, 14488, 14159, 15192, 13563, 14782, 13852, 7998,  14920, 12686, 14363, 13754, 14350, 13814, 15258, 14156,
      14198, 14849, 13955, 15126, 13663, 14033, 14483, 12880, 14765, 12977, 14017, 14881, 10395, 14950, 13676, 12497,
      11587, 13427, 14507, 12408, 14615, 12010, 14586, 13531, 9126,  14077, 12947, 13723, 15185, 15262, 15288, 14608,
      15211, 13514, 12745, 14905, 14579, 14199, 14990, 15012, 13932, 13096, 13995, 10413, 9657,  13398, 15304, 10993,
      13516, 14415, 11920, 13584, 13772, 15204, 14925, 14462, 12207, 14373, 14882, 10069, 13641, 12941, 13577, 13330,
      14191, 13926, 13325, 13662, 13478, 14251, 13212, 15161, 14471, 14691, 13904, 12831, 14277, 14566, 14577, 14575,
      12646, 15218, 13438, 13827, 15323, 15245, 12022, 13928, 13358, 15286, 14556, 14414, 12664, 11754, 13737, 15360,
      14533, 14148, 15259, 14354, 14253, 15358, 13804, 13513, 14825, 13973, 14492, 14943, 15124, 14221, 13908, 12768,
      14923, 14801, 15134, 13681, 15313, 10562, 8965,  14670, 15028, 13264, 14901, 14973, 14120, 12946, 13663, 13418,
      9930,  15264, 13267, 11311, 14857, 15204, 14787, 14466, 11394, 14305, 14712, 11728, 14401, 13790, 15359, 15108,
      13342, 15088, 14348, 12047, 14544, 13244, 15299, 14790, 14565, 14827, 12551, 12386, 15074, 13453, 10206, 14530,
      14922, 11713, 14811, 12342, 14867, 13452, 12332, 11289, 15105, 14896, 15182, 14087, 11717, 14525, 12705, 15096,
      13561, 15094, 13168, 15007, 14888, 14556, 15156, 14829, 12482, 14449, 14379, 14233, 12640, 15000, 13268, 15342,
  };

  uint16_t ret[2 * 1 * 16 * 16] = {
      0x2fb2, 0x387f, 0x3a8c, 0x3a26, 0x3593, 0x3a60, 0x389d, 0x3305, 0x3bf7, 0x395c, 0x3a61, 0x2b72, 0x3819, 0x36c4, 0x36c0, 0x2792,
      0x34d2, 0x2fe5, 0x3947, 0x3abc, 0x340a, 0x27f7, 0x3a58, 0x39d8, 0x3bad, 0x387c, 0x3554, 0x3b67, 0x380b, 0x3475, 0x3516, 0x3bea,
      0x308e, 0x306a, 0x37e6, 0x3557, 0x338e, 0x39e8, 0x38b1, 0x392d, 0x358d, 0x38f0, 0x3752, 0x3955, 0x3953, 0x356e, 0x340a, 0x2e56,
      0x362c, 0x3bdf, 0x3014, 0x31e6, 0x3a53, 0x38b0, 0x3891, 0x32d3, 0x2461, 0x369a, 0x32ae, 0x308f, 0x32b7, 0x3a0d, 0x2533, 0x3b5a,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0x38d1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0x3898, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0x374f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0x3b58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_ND, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_RESERVED, 5));
  TransArgs args{reinterpret_cast<uint8_t *>(data), src_format, dst_format,
                 FORMAT_ND,
                 FORMAT_FRACTAL_Z, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                 {17, 4},
                 {2, 1, 16, 16},
                 DT_FLOAT16};
  TransResult result;
  ASSERT_EQ(TransDataFormat(args, result), SUCCESS);
  ASSERT_EQ(result.length, sizeof(ret) / sizeof(ret[0]) * 2);
  for (size_t i = 0U; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_EQ((reinterpret_cast<uint16_t *>(result.data.get()))[i], ret[i]);
  }

  FormatTransferFractalZTbe transfer;
  EXPECT_EQ(transfer.TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, args.dst_shape),
            SUCCESS);
}

TEST_F(UtestFormatTransferFractalZTbe, invalid_src_shape) {
  uint16_t data[1 * 4 * 4 * 1] = {0};

  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NHWC, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_RESERVED, 5));
  TransArgs args{reinterpret_cast<uint8_t *>(data), src_format, dst_format,
                 FORMAT_NHWC,
                 FORMAT_FRACTAL_NZ, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                 {1, 4, 4},
                 {1, 1, 1, 16, 16},
                 DT_FLOAT16};
  TransResult result;
  FormatTransferFractalZTbe transfer;
  EXPECT_EQ(transfer.TransFormat(args, result), ACL_ERROR_GE_FORMAT_INVALID);
  EXPECT_EQ(transfer.TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, args.dst_shape),
            ACL_ERROR_GE_FORMAT_INVALID);
}

TEST_F(UtestFormatTransferFractalZTbe, invalid_src_data_type) {
  uint16_t data[1 * 1 * 4 * 4] = {0};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NHWC, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_NZ, FORMAT_RESERVED, 5));
  TransArgs args{reinterpret_cast<uint8_t *>(data), src_format, dst_format,
                 FORMAT_NHWC,
                 FORMAT_FRACTAL_NZ, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                 {1, 1, 4, 4},
                 {1, 1, 1, 16, 16},
                 DT_STRING};
  TransResult result;
  FormatTransferFractalZTbe transfer;
  EXPECT_EQ(transfer.TransFormat(args, result), ACL_ERROR_GE_DATATYPE_INVALID);
}

TEST_F(UtestFormatTransferFractalZTbe, invalid_src_format) {
  uint16_t data[1 * 1 * 4 * 4] = {0};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_HWCN, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_NZ, FORMAT_RESERVED, 5));
  TransArgs args{reinterpret_cast<uint8_t *>(data), src_format, dst_format,
                 FORMAT_HWCN,
                 FORMAT_FRACTAL_NZ, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                 {1, 1, 4, 4},
                 {1, 1, 1, 16, 16},
                 DT_FLOAT16};
  TransResult result;
  FormatTransferFractalZTbe transfer;
  EXPECT_EQ(transfer.TransFormat(args, result), ACL_ERROR_GE_FORMAT_INVALID);
}

TEST_F(UtestFormatTransferFractalZTbe, invalid_dst_shape) {
  uint16_t data[1 * 1 * 4 * 4] = {0};
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_ND, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_NZ, FORMAT_RESERVED, 5));
  TransArgs args{reinterpret_cast<uint8_t *>(data), src_format, dst_format,
                 FORMAT_ND,
                 FORMAT_FRACTAL_NZ, FORMAT_RESERVED, FORMAT_RESERVED, 16, 16,
                 {1, 1, 4, 4},
                 {1, 1, 1, 16, 16},
                 DT_FLOAT16};
  TransResult result;
  FormatTransferFractalZTbe transfer;
  EXPECT_EQ(transfer.TransFormat(args, result), ACL_ERROR_GE_FORMAT_INVALID);
}
}  // namespace formats
}  // namespace ge
