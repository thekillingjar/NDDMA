/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_KNOWN_SUBGRAPH_DAVINCI_MODEL_TRACING_H
#define AIR_CXX_RUNTIME_V2_KERNEL_KNOWN_SUBGRAPH_DAVINCI_MODEL_TRACING_H

#include <vector>
#include "exe_graph/runtime/kernel_context.h"
#include "graph/load/model_manager/davinci_model.h"

namespace gert {
namespace kernel {
std::vector<std::string> PrintModelCreate(const KernelContext *context);
std::vector<std::string> PrintModelExecute(const KernelContext *context);
std::vector<std::string> PrintWorkspaces(const KernelContext *context);
std::vector<std::string> PrintGetRunAddress(const KernelContext *context);
} // namespace kernel
} // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_KNOWN_SUBGRAPH_DAVINCI_MODEL_TRACING_H
