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
#include <nlohmann/json.hpp>
#define protected public
#define private public
#include "trace_handle_manager/trace_handle_manager.h"
#undef private
#undef protected
#include "trace_handle_manager/trace_msg/compile_process_trace_msg.h"
#include "trace_handle_manager/trace_msg/long_time_trace_msg.h"

using namespace std;
using namespace fe;

class TraceHandleManagerUnitTest : public testing::Test
{
protected:
  static void SetUpTestCase() {
    cout << "TraceHandleManagerUnitTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "TraceHandleManagerUnitTest TearDown" << endl;
  }
};

TEST_F(TraceHandleManagerUnitTest, init_case_01) {
  TraceHandleManager handle_manager;
  EXPECT_EQ(handle_manager.Initialize(), SUCCESS);
  EXPECT_EQ(handle_manager.Initialize(), SUCCESS);

  handle_manager.AddSubGraphTraceHandle();
  handle_manager.AddSubGraphTraceHandle();


  handle_manager.SubmitGlobalTrace("Test01");
  handle_manager.SubmitGlobalTrace("Test02");
  handle_manager.SubmitGlobalTrace("");

  TraceMsgBasePtr msg1 = std::make_shared<CompileProcessTraceMsg>(156, 48);
  handle_manager.SubmitGlobalTrace(msg1);
  TraceMsgBasePtr msg2 = std::make_shared<CompileProcessTraceMsg>(86, 0);
  handle_manager.SubmitGlobalTrace(msg2);

  TraceMsgBasePtr msg3 = std::make_shared<LongTimeTraceMsg>(true, 123, "AAA", 346);
  handle_manager.SubmitGlobalTrace(msg3);
  TraceMsgBasePtr msg4 = std::make_shared<LongTimeTraceMsg>(false, 234, "BBB", 486);
  handle_manager.SubmitGlobalTrace(msg4);

  handle_manager.Finalize();
  handle_manager.Finalize();
}
