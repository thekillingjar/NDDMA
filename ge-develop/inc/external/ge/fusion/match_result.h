/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_MATCH_RESULT_H
#define INC_EXTERNAL_GE_GRAPH_FUSION_MATCH_RESULT_H

#include "pattern.h"
#include "subgraph_boundary.h"
#include "ge_common/ge_api_types.h"

namespace ge {
namespace fusion {
class MatchResultImpl;
class MatchResult {
 public:
  /**
   * 定义匹配结果，入参为pattern即匹配定义。
   * 在match result中，matched node为根据pattern node匹配到的节点，与pattern node是一一对应的
   * @param pattern
   */
  explicit MatchResult(const Pattern *pattern);
  MatchResult(const MatchResult &other) noexcept;
  MatchResult(MatchResult &&other) noexcept;
  MatchResult &operator=(const MatchResult &other) noexcept;
  MatchResult &operator=(MatchResult &&other) noexcept;

  /**
   * 增加一个匹配对
   * @param pattern_node_out_tensor
   * @param target_node_out_tensor
   * @return
   */
  Status AppendNodeMatchPair(const NodeIo &pattern_node_out,
                             const NodeIo &matched_node_out);

  /**
   * 根据pattern node获取到匹配到的target node
   * @param pattern_node
   * @return
   */
  Status GetMatchedNode(const GNode &pattern_node, GNode &matched_node) const;

  /**
   * 获取所有匹配到的节点。
   * @param pattern_node
   * @return
   */
  [[nodiscard]] std::vector<GNode> GetMatchedNodes() const;

  /**
   * 根据捕获的索引，获取匹配中对应的node output
   * @param capture_idx
   * @param node_output
   * @return
   */
  Status GetCapturedTensor(size_t capture_idx, NodeIo &node_output) const;

  /**
   * 获取pattern中的pattern graph
   * @return
   */
  const Graph &GetPatternGraph() const;

  /**
   * 获取匹配结果对应的子图边界
   * @return
   */
  [[nodiscard]] std::unique_ptr<SubgraphBoundary> ToSubgraphBoundary() const;

  /**
   * 将match result序列化为字符串
   */
  [[nodiscard]] AscendString ToAscendString() const;

  ~MatchResult();

 private:
  std::unique_ptr<MatchResultImpl> impl_;
};
}  // namespace fusion
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_MATCH_RESULT_H
