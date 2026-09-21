/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#ifdef __cplusplus
extern "C" {
#endif

namespace tune {
ge::Status FFTSOptimize(const ge::ComputeGraph &compute_graph) {
  return ge::SUCCESS;
}

ge::Status FFTSGraphPreThread(const ge::ComputeGraph &compute_graph) {
  return ge::SUCCESS;
}

ge::Status FFTSNodeThread(const ge::ComputeGraph &compute_graph, const ge::NodePtr &node) {
  return ge::SUCCESS;
}
} // namespace tune
#ifdef __cplusplus
}
#endif
