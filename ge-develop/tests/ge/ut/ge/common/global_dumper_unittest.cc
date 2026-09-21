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
#include "runtime/subscriber/global_dumper.h"
#include "common/global_variables/diagnose_switch.h"
namespace gert {
class GlobalDumperUT : public testing::Test {};
TEST_F(GlobalDumperUT, RegisterCallback_Ok) {
  const auto global_dumper = GlobalDumper::GetInstance();
  global_dumper->SetEnableFlags(0);
  EXPECT_FALSE(global_dumper->IsEnable(DumpType::kDataDump));
  ge::diagnoseSwitch::EnableDataDump();
  ge::diagnoseSwitch::EnableExceptionDump();
  ge::diagnoseSwitch::EnableHostDump();
  EXPECT_TRUE(global_dumper->IsEnable(DumpType::kDataDump));
  EXPECT_TRUE(global_dumper->IsEnable(DumpType::kExceptionDump));
  EXPECT_TRUE(global_dumper->IsEnable(DumpType::kHostDump));
}
}
