/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_GE_OP_TILING_MANAGER_H_
#define GE_COMMON_GE_OP_TILING_MANAGER_H_

#include <map>
#include <string>

namespace ge {

class OpTilingManager {
 public:
  OpTilingManager() = default;
  static OpTilingManager &GetInstance();
  ~OpTilingManager();
  void LoadSo();

 private:
  void ClearHandles() noexcept;
  std::map<std::string, void *> handles_;
};
}  // namespace ge

#endif  // GE_COMMON_GE_OP_TILING_MANAGER_H_

