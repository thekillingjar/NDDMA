/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_PATTERN_FUSION_H_
#define AUTOFUSE_PATTERN_FUSION_H_

#include "graph/node.h"
#include "autofuse_frame/autofuse_frames.h"

namespace ge {
class PatternFusion {
public:
  // 在符号化推导之前执行的 Pass，不需要符号化信息、不依赖 lowering
  static graphStatus RunEarlyPasses(const ComputeGraphPtr &graph, const GraphPasses &graph_passes = {});

  // 在符号化推导之后执行的 Pass，可能需要符号化信息
  static graphStatus RunAllPatternFusion(const ComputeGraphPtr &graph);
};
} // namespace ge

#endif  // AUTOFUSE_PATTERN_FUSION_H_
