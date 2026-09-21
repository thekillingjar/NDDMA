/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_LLT_GE_EXECUTOR_STUB_H
#define __INC_LLT_GE_EXECUTOR_STUB_H
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "acl/acl_base.h"
#include "framework/executor_c/ge_executor.h"
#include "runtime/rt.h"

class GeExecutorStubMock {
public:
  static GeExecutorStubMock& GetInstance() {
    static GeExecutorStubMock mock;
    return mock;
  }

  MOCK_METHOD2(GetPartitionSize, Status(const char *fileName, GePartitionSize *mdlPartitionSize));
  MOCK_METHOD3(GetMemAndWeightSize, Status(const char *fileName, size_t *workSize, size_t *weightSize));
  MOCK_METHOD5(ExecModel, Status(uint32_t modelId, ExecHandleDesc *execDesc, bool sync,
               const InputData *inputData, OutputData *outputData));
  MOCK_METHOD2(GeLoadModelFromData, Status(uint32_t *modelId, const ModelData *modelData));
  MOCK_METHOD1(UnloadModel, Status(uint32_t modelId));
  MOCK_METHOD1(DestroyModelInOutInfo, Status(ModelInOutInfo *info));
  MOCK_METHOD0(GeInitialize, Status());
  MOCK_METHOD0(GeFinalize, Status());
  MOCK_METHOD1(GeDbgInit, Status(const char *configPath));
  MOCK_METHOD0(GeDbgDeInit, Status());
  MOCK_METHOD2(GeNofifySetDevice, Status(uint32_t chipId, uint32_t deviceId));
};

#endif