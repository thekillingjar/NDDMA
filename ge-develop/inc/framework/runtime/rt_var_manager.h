/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_VAR_MANAGER_H
#define AIR_RUNTIME_VAR_MANAGER_H
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor_data.h"
#include "ge/ge_api_types.h"
namespace gert {
class RtVarManager {
 public:
  RtVarManager() = default;
  virtual ~RtVarManager() = default;
  virtual ge::Status GetVarShapeAndMemory(const std::string &id, StorageShape &shape,
                                          TensorData &memory) const = 0;
};
} // namespace gert
#endif  // AIR_RUNTIME_VAR_MANAGER_H
