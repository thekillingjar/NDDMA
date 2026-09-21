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

#include "macro_utils/dt_public_scope.h"
#include "session/session_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
class Utest_SessionManager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(Utest_SessionManager, Initialize) {
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = true;
  EXPECT_EQ(sm->Initialize(), SUCCESS);
}

TEST_F(Utest_SessionManager, Finalize) {
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->Finalize(), SUCCESS);
}

TEST_F(Utest_SessionManager, CreateSession) {
  std::map<std::string, std::string> options;
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->CreateSession(options, session_id), GE_SESSION_MANAGER_NOT_INIT);
}

TEST_F(Utest_SessionManager, DestroySession) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->DestroySession(session_id), SUCCESS);
  sm->init_flag_ = true;
  EXPECT_EQ(sm->DestroySession(session_id), GE_SESSION_NOT_EXIST);
  std::map<std::string, std::string> options;
  sm->session_manager_map_[0] = std::make_shared<InnerSession>(session_id, options);
  EXPECT_EQ(sm->DestroySession(session_id), SUCCESS);
}

TEST_F(Utest_SessionManager, GetNextSessionId) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->GetNextSessionId(session_id), GE_SESSION_MANAGER_NOT_INIT);
}

TEST_F(Utest_SessionManager, GetSession_before_init) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  SessionPtr session = sm->GetSession(session_id);
  EXPECT_EQ(session, nullptr);
}

TEST_F(Utest_SessionManager, GetSession_not_exits) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = true;
  SessionPtr session = sm->GetSession(session_id);
  EXPECT_EQ(session, nullptr);
}
}  // namespace ge
