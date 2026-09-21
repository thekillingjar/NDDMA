/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GUARD_DFX_CONTEXT_H_
#define GUARD_DFX_CONTEXT_H_
 
#include <string>

namespace ge {
class GuardDfxContext {
 public:
  explicit GuardDfxContext(const std::string &guard_dfx_info);
  ~GuardDfxContext();
  GuardDfxContext(const GuardDfxContext &) = delete;
  GuardDfxContext(const GuardDfxContext &&) = delete;
  GuardDfxContext &operator=(const GuardDfxContext &) = delete;
  GuardDfxContext &&operator=(const GuardDfxContext &&) = delete;
};
}
#endif // GUARD_DFX_CONTEXT_H_
