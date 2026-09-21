/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_FUSION_BASE_PASS_H
#define INC_EXTERNAL_GE_GRAPH_FUSION_FUSION_BASE_PASS_H
#include "graph/graph.h"
#include "ge_common/ge_api_types.h"
#include "register/register_custom_pass.h"

namespace ge {
namespace fusion {
/**
 * 融合pass基类
 */
class FusionBasePass {
 public:
  /**
   * fusion类pass的执行入口函数
   * @param graph 目标图
   * @param pass_context 上下文，可用于传递error msg等信息
   * @return
   */
  virtual Status Run(GraphPtr &graph, CustomPassContext &pass_context) = 0;
  virtual ~FusionBasePass() = default;
};
} // namespace fusion
} // namespace ge
#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_FUSION_BASE_PASS_H
