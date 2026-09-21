/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_ge_stub.h"
#include "framework/executor_c/ge_executor.h"
#include "vector.h"

MockFunctionTest &MockFunctionTest::aclStubInstance() {
  static MockFunctionTest stub;
  return stub;
};

Status GetModelDescInfo(uint32_t modelId, ModelInOutInfo *info) {
  return MockFunctionTest::aclStubInstance().GetModelDescInfo(modelId, info);
}

Status GetModelDescInfoFromMem(const ModelData *modelData,
                               ModelInOutInfo *info) {
  return MockFunctionTest::aclStubInstance().GetModelDescInfoFromMem(modelData,
                                                                     info);
}

Status LoadDataFromFile(const char *modelPath, ModelData *data) {
  return MockFunctionTest::aclStubInstance().LoadDataFromFile(modelPath, data);
}

void FreeModelData(ModelData *data) {
  (void)data;
  return;
}

void DestroyModelInOutInfo(ModelInOutInfo *info) {
  DeInitVector(&info->input_desc);
  DeInitVector(&info->output_desc);
  return;
}