/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_FORMATS_UTILS_FORMATS_DEFINITIONS_H_
#define GE_COMMON_FORMATS_UTILS_FORMATS_DEFINITIONS_H_

#include <cstdint>

namespace ge {
namespace formats {
constexpr int64_t kCubeSize = 16L;
constexpr int32_t kNiSize = 16;
constexpr int64_t kShapeItemNumMAX = 1099511627776; // 1024L * 1024L * 1024L * 1024L;
constexpr int32_t DIM_DEFAULT_VALUE = 1;

enum NchwDimIndex {
  kNchwN,
  kNchwC,
  kNchwH,
  kNchwW,
  kNchwDimsNum
};

enum NhwcDimIndex {
  kNhwcN,
  kNhwcH,
  kNhwcW,
  kNhwcC,
  kNhwcDimsNum
};

enum HwcnDimIndex {
  kHwcnH,
  kHwcnW,
  kHwcnC,
  kHwcnN,
  kHwcnDimsNum
};

enum ChwnDimIndex {
  kChwnC,
  kChwnH,
  kChwnW,
  kChwnN,
  kChwnDimsNum
};

enum Nc1hwc0DimIndex {
  kNc1hwc0N,
  kNc1hwc0C1,
  kNc1hwc0H,
  kNc1hwc0W,
  kNc1hwc0C0,
  kNc1hwc0DimsNum
};

enum C1hwncoc0DimIndex {
  kC1hwncoc0C1,
  kC1hwncoc0H,
  kC1hwncoc0W,
  kC1hwncoc0N,
  kC1hwncoc0Co,
  kC1hwncoc0C0,
  kC1hwncoc0DimsNum
};

enum FracZDimIndex {
  kFracZHWC1,
  kFracZN0,
  kFracZNi,
  kFracZC0,
  kFracZDimsNum
};

enum DhwcnDimIndex {
  kDhwcnD,
  kDhwcnH,
  kDhwcnW,
  kDhwcnC,
  kDhwcnN,
  kDhwcnDimsNum
};

enum DhwncDimIndex {
  kDhwncD,
  kDhwncH,
  kDhwncW,
  kDhwncN,
  kDhwncC,
  kDhwncDimsNum
};
}  // namespace formats
}  // namespace ge
#endif  // GE_COMMON_FORMATS_UTILS_FORMATS_DEFINITIONS_H_
