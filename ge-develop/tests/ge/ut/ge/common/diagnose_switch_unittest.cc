/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/global_variables/diagnose_switch.h"
#include <gtest/gtest.h>
namespace ge {
namespace {
struct TestGlobalArg {
  uint64_t GetCount(uint64_t event) {
    auto iter = count.find(event);
    if (iter == count.end()) {
      return 0;
    }
    return iter->second;
  }
  std::map<uint64_t, uint64_t> &MutableCount() {
    return count;
  }
  void Clear() {
    count.clear();
  }
  int64_t Get(uint64_t event) {
    return count[event];
  }
 private:
  std::map<uint64_t, uint64_t> count;
};
TestGlobalArg global_arg;
void TestOnEvent(void *arg, uint64_t event) {
  static_cast<TestGlobalArg *>(arg)->MutableCount()[static_cast<size_t>(event)] += 1;
}
}  // namespace
class DiagnoseSwitchUT : public testing::Test {};
TEST_F(DiagnoseSwitchUT, GetSingleSwitch_Ok) {
  auto &prof_switch = diagnoseSwitch::GetProfiling();
  auto &dumper_switch = diagnoseSwitch::GetDumper();
  EXPECT_EQ(prof_switch.GetEnableFlag(), 0UL);
  EXPECT_EQ(dumper_switch.GetEnableFlag(), 0UL);
}
TEST_F(DiagnoseSwitchUT, SetEnable) {
  SingleDiagnoseSwitch sds;
  sds.SetEnableFlag(1);
  EXPECT_EQ(sds.GetEnableFlag(), 1);

  sds.SetEnableFlag(0);
  EXPECT_EQ(sds.GetEnableFlag(), 0);
}
TEST_F(DiagnoseSwitchUT, RegisterUnregisterHandler) {
  SingleDiagnoseSwitch sds;

  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  EXPECT_EQ(sds.GetHandleSize(), 1);
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  EXPECT_EQ(sds.GetHandleSize(), 1);

  sds.UnregisterHandler(&global_arg);
  EXPECT_EQ(sds.GetHandleSize(), 0);
}
TEST_F(DiagnoseSwitchUT, CallbackWhenEnableChanged) {
  SingleDiagnoseSwitch sds;
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  global_arg.Clear();

  sds.SetEnableFlag(1);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);
  sds.SetEnableFlag(0);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 1);
}
TEST_F(DiagnoseSwitchUT, DoNotCallbackWhenEnableNotChanged) {
  SingleDiagnoseSwitch sds;
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  global_arg.Clear();

  sds.SetEnableFlag(1);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);

  // duplicated call enable, do not callback anymore
  sds.SetEnableFlag(1);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);

  sds.SetEnableFlag(0);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 1);
}

TEST_F(DiagnoseSwitchUT, CallbackWhenDifferentEnableFlags) {
  SingleDiagnoseSwitch sds;
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  global_arg.Clear();

  sds.SetEnableFlag(1);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);

  // different enable flag -> 2|1 = 3
  sds.SetEnableFlag(2);
  EXPECT_EQ(global_arg.GetCount(3), 1);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);

  sds.SetEnableFlag(0);
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 1);
}

TEST_F(DiagnoseSwitchUT, AutoCallCallbackWhenReigster_WhenEnable) {
  SingleDiagnoseSwitch sds;
  sds.SetEnableFlag(1);

  global_arg.Clear();
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);
}
TEST_F(DiagnoseSwitchUT, AutoCallCallbackWhenReigster_WhenDisable) {
  SingleDiagnoseSwitch sds;

  global_arg.Clear();
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  EXPECT_EQ(global_arg.GetCount(1), 0);
  EXPECT_EQ(global_arg.GetCount(0), 1);
}
TEST_F(DiagnoseSwitchUT, DoNotAutoCallCallbackWhenDuplicateReigster_WhenEnable) {
  SingleDiagnoseSwitch sds;
  sds.SetEnableFlag(1);

  global_arg.Clear();
  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);

  sds.RegisterHandler(&global_arg, {&global_arg, TestOnEvent});
  EXPECT_EQ(global_arg.GetCount(1), 1);
  EXPECT_EQ(global_arg.GetCount(0), 0);
}
}  // namespace ge
