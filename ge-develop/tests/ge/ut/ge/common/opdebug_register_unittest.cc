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

#include "common/dump/opdebug_register.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"

namespace ge {
class UTEST_opdebug_register : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
 
TEST_F(UTEST_opdebug_register, register_debug_for_model_success) {
  OpdebugRegister opdebug_register;
  rtModel_t model_handle = (void*)0x111;
  uint32_t op_debug_mode = 1;
  DataDumper data_dumper({});
  auto ret = opdebug_register.RegisterDebugForModel(model_handle, op_debug_mode, data_dumper);
  opdebug_register.UnregisterDebugForModel(model_handle);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_opdebug_register, register_debug_for_stream_success) {
  OpdebugRegister opdebug_register;
  rtStream_t stream = (void*)0x111;
  uint32_t op_debug_mode = 1;
  DataDumper data_dumper({});
  auto ret = opdebug_register.RegisterDebugForStream(stream, op_debug_mode, data_dumper);
  opdebug_register.UnregisterDebugForStream(stream);
  EXPECT_EQ(ret, ge::SUCCESS);
}


}  // namespace ge
