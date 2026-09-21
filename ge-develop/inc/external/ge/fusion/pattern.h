/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_H_
#define INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_H_

#include <memory>
#include "graph/graph.h"
#include "subgraph_boundary.h"

namespace ge {
namespace fusion {
class PatternImpl;
class Pattern {
 public:
  /**
   * 使用graph定义匹配pattern
   * @param pattern_graph
   */
  explicit Pattern(Graph &&pattern_graph);

  /**
   * 获取pattern中的pattern graph
   * @return
   */
  const Graph &GetGraph() const;

  /**
   * 捕获pattern中一个tensor，从而在match result中可以按序获取
   * @param node_output tensor的来源，即来自某个node的某个输出
   *        因node_output可唯一标定一个图中tensor, 因此常用node_output指代tensor
   * @return
   */
  Pattern &CaptureTensor(const NodeIo &node_output);

  /**
   * 获取pattern中捕获的所有tensor，vector中顺序为捕获顺序
   * @param node_outputs
   * @return
   */
  Status GetCapturedTensors(std::vector<NodeIo> &node_outputs) const;

  ~Pattern();

 private:
  std::unique_ptr<PatternImpl> impl_;
};
}  // namespace fusion
} // namespace ge

#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_H_
