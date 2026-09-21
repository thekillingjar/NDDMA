/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#include "mockcpp/mockcpp.hpp"
#define private public
#include "flow_func/flow_func_timer.h"
#undef private

using namespace std;
namespace FlowFunc {
class FLOW_FUNC_TIMER_UTEST : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(FLOW_FUNC_TIMER_UTEST, timer_test) {
  FlowFuncTimer::Instance().Init(0);
  int32_t ret = FlowFuncTimer::Instance().StartTimer(nullptr, 100, false);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_PARAM_INVALID);
  ret = FlowFuncTimer::Instance().StopTimer(nullptr);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_PARAM_INVALID);
  void *handle = FlowFuncTimer::Instance().CreateTimer(nullptr);
  EXPECT_EQ(handle, nullptr);
  handle = FlowFuncTimer::Instance().CreateTimer([]() { return; });
  ret = FlowFuncTimer::Instance().StartTimer(handle, 100, false);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = FlowFuncTimer::Instance().StopTimer(handle);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = FlowFuncTimer::Instance().DeleteTimer(handle);
  uint64_t timestamp = FlowFuncTimer::Instance().GetCurrentTimestamp();
  EXPECT_NE(timestamp, 0);
  MOCKER(halEschedSubmitEvent).expects(once()).will(returnValue(DRV_ERROR_NONE));
  FlowFuncTimer::Instance().TriggerTimerEvent(0);
  FlowFuncTimer::Instance().Finalize();
}

TEST_F(FLOW_FUNC_TIMER_UTEST, loop_thread_test) {
  FlowFuncTimer timer;
  std::shared_ptr<TimerInfo> timer_info1 = std::make_shared<TimerInfo>();
  timer_info1->is_start = false;
  timer_info1->timer_id = 1;
  timer_info1->invoke_by_worker = true;
  timer_info1->timer_callback = []() {};
  timer.timer_info_map_[0] = timer_info1;
  std::shared_ptr<TimerInfo> timer_info2 = std::make_shared<TimerInfo>();
  timer_info2->is_start = true;
  timer_info2->timer_id = 2;
  timer_info2->oneshot_flag = true;
  timer_info1->invoke_by_worker = true;
  timer_info2->timer_callback = []() {};
  timer.timer_info_map_[1] = timer_info2;
  std::shared_ptr<TimerInfo> timer_info3 = std::make_shared<TimerInfo>();
  timer_info3->is_start = true;
  timer_info3->oneshot_flag = false;
  timer_info3->timer_id = 3;
  timer_info1->invoke_by_worker = true;
  timer_info3->timer_callback = []() {};
  timer.timer_info_map_[2] = timer_info3;
  timer.ProcessEachTimerInContext();
  EXPECT_EQ(timer.timer_info_map_[1]->is_start, false);
}

TEST_F(FLOW_FUNC_TIMER_UTEST, invoke_by_worker_false) {
  FlowFuncTimer timer;
  std::shared_ptr<TimerInfo> timer_info1 = std::make_shared<TimerInfo>();
  timer_info1->is_start = true;
  timer_info1->timer_id = 1;
  timer_info1->invoke_by_worker = false;
  int test_value = 0;
  timer_info1->timer_callback = [&test_value]() { test_value = 100; };
  timer.timer_info_map_[0] = timer_info1;
  timer.ProcessEachTimerInContext();
  EXPECT_EQ(test_value, 100);
}
}  // namespace FlowFunc
