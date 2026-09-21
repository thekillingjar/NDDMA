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
#include <gmock/gmock.h>
#include <vector>

#include "hybrid/executor/runtime_v2/rt_v2_utils.h"
using namespace ge;
using namespace hybrid;
using namespace gert;
namespace ge {
class UtestRtV2Utils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestRtV2Utils, test_rt_v2_utils_debug_shape) {
  GeShape shape;
  auto res = DebugString(shape);
  EXPECT_EQ(res, "[]");

  shape.AppendDim(1);
  shape.AppendDim(2);
  shape.AppendDim(3);
  res = DebugString(shape);
  EXPECT_EQ(res, "[1,2,3]");

  shape.SetDimNum(0);
  shape.AppendDim(-2);
  auto res2 = DebugString(shape);
  EXPECT_EQ(res2, "[-2]");
}

TEST_F(UtestRtV2Utils, test_rt_v2_utils_geshape_2_rtshape) {
  // rt shape is {-2}
  std::vector<int64_t> dims{-2};
  GeShape shape;
  const gert::Shape rt_shape{-2};
  RtShapeAsGeShape(rt_shape, shape);
  EXPECT_EQ(shape.GetDims(), dims);

  // rt shape is {1, 2, 3, 4}
  const gert::Shape rt_shape1{1, 2, 3, 4};
  RtShapeAsGeShape(rt_shape1, shape);
  std::vector<int64_t> dims1{1, 2, 3, 4};
  EXPECT_EQ(shape.GetDims(), dims1);

  // ge shape is {-2}
  const GeShape ge_shape2(dims);
  gert::Shape rt_shape2;
  GeShapeAsRtShape(ge_shape2, rt_shape2);
  EXPECT_EQ(rt_shape2.GetDimNum(), ge_shape2.GetDims().size());
  for (size_t i = 0U; i < rt_shape2.GetDimNum(); ++i) {
    EXPECT_EQ(rt_shape2.GetDim(i), ge_shape2.GetDim(i));
  }

  // ge shape is {-1, -1, 2, 3}
  const GeShape ge_shape3(dims);
  gert::Shape rt_shape3;
  GeShapeAsRtShape(ge_shape3, rt_shape3);
  EXPECT_EQ(rt_shape3.GetDimNum(), ge_shape3.GetDims().size());
  for (size_t i = 0U; i < rt_shape3.GetDimNum(); ++i) {
    EXPECT_EQ(rt_shape3.GetDim(i), ge_shape3.GetDim(i));
  }
}

TEST_F(UtestRtV2Utils, test_rt_v2_debug_placement) {
  ge::Placement placement = kPlacementHost;
  EXPECT_EQ(DebugString(placement), "kPlacementHost");
  placement = kPlacementHost;
  EXPECT_EQ(DebugString(placement), "kPlacementHost");
  placement = kPlacementDevice;
  EXPECT_EQ(DebugString(placement), "kPlacementDevice");
  placement = static_cast<ge::Placement>(2);
  EXPECT_EQ(DebugString(placement), "invalid");
}
}
