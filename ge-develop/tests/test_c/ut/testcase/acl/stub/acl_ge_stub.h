/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_TESTCASE_C_ACL_GE_STUB_H_
#define UT_TESTCASE_C_ACL_GE_STUB_H_

#include <gmock/gmock.h>
#include "framework/executor_c/ge_executor.h"

class MockFunctionTest {
 public:
  MockFunctionTest(){};
  static MockFunctionTest &aclStubInstance();

  // ge function stub
  MOCK_METHOD2(GetModelDescInfo,
               Status(uint32_t modelId, ModelInOutInfo *info));
  MOCK_METHOD2(GetModelDescInfoFromMem,
               Status(const ModelData *modelData, ModelInOutInfo *info));
  MOCK_METHOD2(LoadDataFromFile,
               Status(const char *modelPath, ModelData *data));
  MOCK_METHOD1(DestroyModelInOutInfo, void(ModelInOutInfo *info));
  MOCK_METHOD1(FreeModelData, void(ModelData *data));
};

#endif