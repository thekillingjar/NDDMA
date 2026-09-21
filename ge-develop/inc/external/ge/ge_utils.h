/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "graph/graph.h"
#include "ge_common/ge_api_types.h"

#ifndef INC_EXTERNAL_GRAPH_ENGINE_GE_UTILS_H
#define INC_EXTERNAL_GRAPH_ENGINE_GE_UTILS_H

namespace ge {
class GeUtils {
 public:
  /**
   * 給定输入shape, 对传入的graph做全图shape推导
   * 本接口只做shape推导，不对图做任何其他优化（如常量折叠、死边消除等）
   * @param graph
   * @return
   */
  static Status InferShape(const Graph &graph, const std::vector<Shape> &input_shape);

  /**
   * 對传入的node做校验，校验其是否支持在aicore上执行
   * @param node
   * @param unsupported_reason
   */
  static Status CheckNodeSupportOnAicore(const GNode &node, bool &is_supported, AscendString &unsupported_reason);
};
} // namespace ge
#endif  // INC_EXTERNAL_GRAPH_ENGINE_GE_UTILS_H
