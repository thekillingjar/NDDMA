/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/estimator/basic_estimator.h"
#include "common/format/axis_util.h"

namespace fe {
const string kAiCoreSpecLbl = "AICoreSpec";
const string kCubeFreqKey = "cube_freq";
const double kUsToNs = 1000.0;
BasicEstimator::BasicEstimator() {}

BasicEstimator::~BasicEstimator() {}

uint64_t BasicEstimator::CalcTimeByCycle(PlatFormInfos &platform_info, uint64_t cycle) {
  string cube_freq_str;
  (void)platform_info.GetPlatformRes(kAiCoreSpecLbl, kCubeFreqKey, cube_freq_str);
  auto cube_freq = static_cast<uint64_t>(std::atoi(cube_freq_str.c_str()));

  FE_LOGD("Cube frequency is %lu mhz", cube_freq);
  if (cube_freq == 0) {
    return 0UL;
  }

  double time = static_cast<double>(cycle) / cube_freq * kUsToNs;
  return static_cast<uint64_t>(time);
}
}
