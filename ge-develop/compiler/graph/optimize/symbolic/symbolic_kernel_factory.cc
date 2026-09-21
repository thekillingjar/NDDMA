/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
InferSymbolComputeKernelFunc SymbolicKernelFactory::Create(const std::string &op_type) const {
  const auto it = creators_.find(op_type);
  if (it == creators_.end()) {
    GELOGW("Cannot find op type %s to do host symbolic compute.", op_type.c_str());
    return nullptr;
  }
  return it->second;
}

Status SymbolicKernelFactory::RegisterCreator(const std::string &op_type, const InferSymbolComputeKernelFunc &creator) {
  GE_ASSERT_NOTNULL(creator);
  if (creators_.find(op_type) != creators_.cend()) {
    GELOGW("Ops host symbolic kernel %s already exist", op_type.c_str());
    return GRAPH_FAILED;
  }
  GELOGI("Register host symbolic kernel for node type %s", op_type.c_str());
  creators_[op_type] = creator;
  return GRAPH_SUCCESS;
}
}  // namespace ge