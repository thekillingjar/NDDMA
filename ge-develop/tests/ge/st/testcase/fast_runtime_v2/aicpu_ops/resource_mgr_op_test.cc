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
#include "aicpu/converter/sequence/resource_mgr.h"

using namespace std;
using namespace gert;
class TEST_RESOURCE_MGR_UT : public testing::Test {};

TEST_F(TEST_RESOURCE_MGR_UT, TestSession) {
  uint64_t session_id = 10;
  uint64_t invalid_session_id = 1000;
  auto ret = SessionMgr::GetInstance()->CreateSession(session_id);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ret = SessionMgr::GetInstance()->CreateSession(session_id);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  SessionPtr sess;
  ret = SessionMgr::GetInstance()->GetSession(session_id, sess);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ret = SessionMgr::GetInstance()->GetSession(invalid_session_id, sess);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ret = SessionMgr::GetInstance()->DestroySession(invalid_session_id);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ret = SessionMgr::GetInstance()->DestroySession(session_id);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_RESOURCE_MGR_UT, TestClearStepResource) {
  uint64_t session_id = 1;
  uint64_t container_id = 2;
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, rm);
  const uint64_t handle = rm->GetHandle();
  rm->StoreStepHandle(handle);
  TensorSeqPtr tensor_seq_ptr = std::make_shared<TensorSeq>(static_cast<ge::DataType>(2));
  auto ret = rm->Create(handle, tensor_seq_ptr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ret = rm->Create(handle, tensor_seq_ptr);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  TensorSeqPtr seq;
  ret = rm->Lookup(handle, &seq);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  uint64_t invalid_handle = 100;
  ret = rm->Lookup(invalid_handle, &seq);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  rm->ClearStepResource();
}

TEST_F(TEST_RESOURCE_MGR_UT, TestClearAllResource) {
  uint64_t session_id = 3;
  uint64_t container_id = 4;
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, rm);
  const uint64_t handle = rm->GetHandle();
  rm->StoreStepHandle(handle);
  TensorSeqPtr tensor_seq_ptr = std::make_shared<TensorSeq>(static_cast<ge::DataType>(2));
  auto ret = rm->Create(handle, tensor_seq_ptr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  TensorSeqPtr seq;
  ret = rm->Lookup(handle, &seq);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  uint64_t invalid_handle = 200;
  ret = rm->Lookup(invalid_handle, &seq);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  rm->ClearAllResource();
}

TEST_F(TEST_RESOURCE_MGR_UT, TestClearSpecialStepResource) {
  uint64_t session_id = 5;
  uint64_t container_id = 6;
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, rm);
  const uint64_t handle = rm->GetHandle();
  rm->StoreStepHandle(handle);
  TensorSeqPtr tensor_seq_ptr = std::make_shared<TensorSeq>(static_cast<ge::DataType>(2));
  auto ret = rm->Create(handle, tensor_seq_ptr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  TensorSeqPtr seq;
  ret = rm->Lookup(handle, &seq);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  uint64_t invalid_handle = 300;
  ret = rm->Lookup(invalid_handle, &seq);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ret = rm->ClearSpecialStepResource(invalid_handle);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ret = rm->ClearSpecialStepResource(handle);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_RESOURCE_MGR_UT, TestRm1) {
  uint64_t session_id = 10;
  SessionMgr::GetInstance()->CreateSession(session_id);
  SessionPtr sess;
  SessionMgr::GetInstance()->GetSession(session_id, sess);
  uint64_t container_id = 11;
  uint64_t invalid_container_id = 12;
  auto ret = sess->CreateRm(container_id);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ret = sess->CreateRm(container_id);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ResourceMgrPtr rm;
  sess->GetRm(container_id, rm);
  sess->GetRm(invalid_container_id, rm);
  ret = sess->ClearRm(invalid_container_id);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ret = sess->ClearRm(container_id);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}