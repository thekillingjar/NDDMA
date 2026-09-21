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
#include <memory>

#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "common/global_variables/diagnose_switch.h"
#include "subscriber/subscriber_utils.h"
#include "runtime/subscriber/global_profiler.h"
namespace gert {
class SubscriberUtilsUT : public testing::Test {};
TEST_F(SubscriberUtilsUT, GetKernelType_GetInvalidKernelTypeHash_WhenForAddInfo) {
  const char *kernel_type = "Test1";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, true);
  EXPECT_EQ(type_hash, kInvalidProfKernelType);
}

TEST_F(SubscriberUtilsUT, GetKernelType_GetValidKernelTypeHash_WhenForAddInfo) {
  const char *kernel_type = "LaunchKernelWithHandle";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, true);
  EXPECT_EQ(type_hash, MSPROF_REPORT_NODE_LAUNCH_TYPE);
}

TEST_F(SubscriberUtilsUT, GetKernelType_GetInValidKernelTypeHash_WhenForApiInfoAndAddInfoOn) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  const char *kernel_type = "LaunchKernelWithHandle";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, false);
  EXPECT_EQ(type_hash, kInvalidProfKernelType);
}

TEST_F(SubscriberUtilsUT, GetKernelType_GetValidKernelTypeHash_WhenForApiInfoAndAddInfoOff) {
  ge::diagnoseSwitch::DisableProfiling();
  const char *kernel_type = "LaunchKernelWithHandle";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, false);
  EXPECT_EQ(type_hash, MSPROF_REPORT_NODE_LAUNCH_TYPE);
}

TEST_F(SubscriberUtilsUT, GetKernelType_GetValidTilingHash_WhenForApiInfoAndAddInfoOn) {
  ge::diagnoseSwitch::DisableProfiling();
  const char *kernel_type = "Tiling";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, false);
  EXPECT_EQ(type_hash, static_cast<uint32_t>(GeProfInfoType::kTiling));
}

TEST_F(SubscriberUtilsUT, GetKernelType_GetValidTilingHash_WhenForApiInfoAndAddInfoOff) {
  ge::diagnoseSwitch::DisableProfiling();
  const char *kernel_type = "Tiling";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, false);
  EXPECT_EQ(type_hash, static_cast<uint32_t>(GeProfInfoType::kTiling));
}

TEST_F(SubscriberUtilsUT, GetKernelType_GetValidTilingHash_OnCannHostL1) {
  ge::diagnoseSwitch::DisableProfiling();
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kCannHost, gert::ProfilingType::kCannHostL1});
  const char *kernel_type = "Test";
  auto type_hash = SubscriberUtils::GetProfKernelType(kernel_type, false);
  size_t idx = 0;
  auto idx_to_str = GlobalProfilingWrapper::GetInstance()->GetIdxToStr();
  for (size_t i = 0; i < idx_to_str.size(); ++i) {
    if (idx_to_str[i] == kernel_type) {
      idx = i;
      break;
    }
  }
  EXPECT_EQ(type_hash, idx + static_cast<uint32_t>(GeProfInfoType::kNodeLevelEnd));
}
}  // namespace gert