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

#include "register/stream_manage_func_registry.h"

namespace ge {
uint32_t FakeCallbackFunc(MngActionType action_type, MngResourceHandle handle) {
  (void) action_type;
  (void) handle;
  return SUCCESS;
}

uint32_t FakeCallbackFuncFailed(MngActionType action_type, MngResourceHandle handle) {
  (void) action_type;
  (void) handle;
  return FAILED;
}

class StreamMngFuncRegistryUT : public testing::Test {};

TEST_F(StreamMngFuncRegistryUT, TryCallStreamMngFunc_Ok) {
  const auto func_type = StreamMngFuncType::ACLNN_STREAM_CALLBACK;
  const MngActionType action_type = MngActionType::DESTROY_STREAM;
  const MngResourceHandle handle = {.stream = (void *) 0x11};
  // case: no callback function is registered
  EXPECT_EQ(StreamMngFuncRegistry::GetInstance().TryCallStreamMngFunc(func_type, action_type, handle), SUCCESS);

  // case: run callback function successfully
  REG_STREAM_MNG_FUNC(func_type, FakeCallbackFunc);
  EXPECT_EQ(StreamMngFuncRegistry::GetInstance().TryCallStreamMngFunc(func_type, action_type, handle), SUCCESS);
}

TEST_F(StreamMngFuncRegistryUT, TryCallStreamMngFunc_Ok_CallFuncReturnNonzero) {
  const auto func_type = StreamMngFuncType::ACLNN_STREAM_CALLBACK;
  const MngActionType action_type = MngActionType::RESET_DEVICE;
  const MngResourceHandle handle = {.device_id = 0};
  // case: run callback function failed
  REG_STREAM_MNG_FUNC(func_type, FakeCallbackFuncFailed);
  EXPECT_EQ(StreamMngFuncRegistry::GetInstance().TryCallStreamMngFunc(func_type, action_type, handle), SUCCESS);
}

TEST_F(StreamMngFuncRegistryUT, TryCallStreamMngFunc_Ok_MultipleRegister) {
  const auto func_type = StreamMngFuncType::ACLNN_STREAM_CALLBACK;
  // register for the first time
  REG_STREAM_MNG_FUNC(func_type, FakeCallbackFuncFailed);
  EXPECT_EQ(StreamMngFuncRegistry::GetInstance().LookUpStreamMngFunc(func_type), FakeCallbackFuncFailed);
  // register for the second time
  REG_STREAM_MNG_FUNC(func_type, FakeCallbackFunc);
  EXPECT_EQ(StreamMngFuncRegistry::GetInstance().LookUpStreamMngFunc(func_type), FakeCallbackFunc);
}

}  // namespace ge
