/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_CXX_AUTOFUSE_BASE_TYPE_H
#define ATT_CXX_AUTOFUSE_BASE_TYPE_H

#include <cstdint>
namespace ge {
enum class AutoFuseConfigType : int32_t {
  INI_CONFIG_FILE,
  ENV_CONFIG,
  INVALID_CONFIG_TYPE,
};

enum class AutoFuseFwkType : int32_t {
  kDefault = 0,
  kGe,
  kTorch
};
}

#endif  // ATT_CXX_AUTOFUSE_BASE_TYPE_H
