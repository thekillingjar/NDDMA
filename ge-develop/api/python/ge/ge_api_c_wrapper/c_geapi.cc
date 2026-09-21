/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/graph.h"
#include "utils/graph_utils.h"
#include "ge/ge_api.h"
#include "utils/graph_utils_ex.h"
#include "common/checker.h"
#include "ge_api_c_wrapper_utils.h"
#include <iostream>

#ifdef __cplusplus
using namespace ge;
using namespace ge::c_wrapper;
using Status = uint32_t;
using AscendString = ge::AscendString;
extern "C" {
#endif

Status GeApiWrapper_GEFinalize() {
  return ge::GEFinalize();
}

Status GeApiWrapper_GEInitialize(char **keys, char **values, int size) {
  GE_ASSERT_NOTNULL(keys);
  GE_ASSERT_NOTNULL(values);
  std::map<AscendString, AscendString> options;
  for (int i = 0; i < size; i++) {
    AscendString key = keys[i];
    AscendString value = values[i];
    options[key] = value;
  }
  return  ge::GEInitialize(options);
}

#ifdef __cplusplus
}
#endif
