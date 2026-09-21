/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_LLT_DBG_STUB_H
#define __INC_LLT_DBG_STUB_H
#include <vector>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "toolchain/prof_api.h"
typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);
class DbgStubMock {
public:
  static DbgStubMock& GetInstance() {
    static DbgStubMock mock;
    return mock;
  }
  MOCK_METHOD0(rtDumpInit, rtError_t());
  MOCK_METHOD3(MsprofInit, int32_t(uint32_t dataType, void *data, uint32_t dataLen));
  MOCK_METHOD2(MsprofRegisterCallback, int32_t(uint32_t moduleId, ProfCommandHandle handle));
  MOCK_METHOD3(MsprofReportAdditionalInfo, int32_t(uint32_t agingFlag, const VOID_PTR data, uint32_t length));
  MOCK_METHOD0(MsprofFinalize, int32_t());
  MOCK_METHOD3(MsprofNotifySetDevice, int32_t(uint32_t chipId, uint32_t deviceId, bool isOpen));
};

#endif
