/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_INFO_PRE_PROCESSOR_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_INFO_PRE_PROCESSOR_H_

#include "ge_common/ge_api_types.h"
#include "graph/any_value.h"

namespace ge{
class SymbolicInfoPreProcessor {
 public:
  static Status Run(const ComputeGraphPtr &graph, const std::vector<GeTensor> &graph_inputs);
};
} // namespace ge
#endif // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_INFO_PRE_PROCESSOR_H_