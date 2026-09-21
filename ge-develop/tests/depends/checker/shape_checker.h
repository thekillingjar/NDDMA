/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_SHAPE_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_SHAPE_CHECKER_H_
#include <utility>
#include "exe_graph/runtime/storage_shape.h"

namespace gert {
class ShapeChecker {
 public:
  static void CheckShape(const Shape &shape, const Shape &expect_shape) {
    EXPECT_EQ(shape.GetDimNum(), expect_shape.GetDimNum());
    for (size_t i = 0UL; i < shape.GetDimNum(); i++) {
      EXPECT_EQ(shape.GetDim(i), expect_shape.GetDim(i));
    }
  }
};
}  // namespace gert

#endif