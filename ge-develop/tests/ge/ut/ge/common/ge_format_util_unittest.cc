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

#include "common/ge_format_util.h"

#include "formats/formats.h"

namespace ge {
TEST(UtestGeFormatUtilTest, test_trans_shape_failure) {
  Shape shape({1, 3, 224, 224});
  TensorDesc src_desc(shape, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> dst_shape;
  EXPECT_NE(GeFormatUtil::TransShape(src_desc, FORMAT_RESERVED, dst_shape), SUCCESS);
}

TEST(UtestGeFormatUtilTest, test_trans_shape_success) {
  Shape shape({1, 3, 224, 224});
  TensorDesc src_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> expected_shape{1, 1, 224, 224, 16};
  EXPECT_EQ(GeFormatUtil::TransShape(src_desc, FORMAT_NC1HWC0, dst_shape), SUCCESS);
  EXPECT_EQ(dst_shape, expected_shape);
}
}  // namespace ge
