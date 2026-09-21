/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_fuse_config.h"
namespace ge {
namespace autofuse {
AutoFuseConfig &AutoFuseConfig::Instance() {
  static AutoFuseConfig config;
  return config;
}

const AutoFuseConfig &AutoFuseConfig::Config() {
  return Instance();
}

AutoFuseConfig &AutoFuseConfig::MutableConfig() {
  return Instance();
}
} // namespace autofuse
}  // namespace ge