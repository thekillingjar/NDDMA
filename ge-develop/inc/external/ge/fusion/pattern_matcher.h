/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_MATCHER_H
#define INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_MATCHER_H

#include <string>
#include "graph/graph.h"
#include "pattern.h"
#include "match_result.h"
#include "pattern_matcher_config.h"

namespace ge {
namespace fusion {
class PatternMatcherImpl;
class PatternMatcher {
 public:
  /**
   * 构造PatternMatcher
   * 当使用这个构造函数时，match config中的配置项均为默认值，即不开启任何配置的状态
   * @param pattern
   * @param target_graph
   */
  explicit PatternMatcher(std::unique_ptr<Pattern> pattern, const GraphPtr &target_graph);
  /**
   * 构造PatternMatcher
   * @param pattern
   * @param target_graph
   * @param matcher_config  匹配选项配置
   */
  PatternMatcher(std::unique_ptr<Pattern> pattern, const GraphPtr &target_graph,
                 std::unique_ptr<PatternMatcherConfig> matcher_config);

  ~PatternMatcher();
  /**
   * 按照当前target graph的拓扑顺序，返回下一个符合pattern定义的匹配结果
   * 注意：
   * (1)当前接口返回的匹配结果可能存在重叠，调用者需要自己处理
   *    考虑如下结构，当对 ABC做匹配时，C会先后出现在2个匹配结果中。选取哪个结果需要调用者自行决定
   *      B  C   B
   *      \ / \ /
   *       A   A
   * (2) 该函数只匹配当前图，若target graph带子图，调用者需自行对子图进行匹配
   * @return
   */
  [[nodiscard]] std::unique_ptr<MatchResult> MatchNext();

 private:
  std::unique_ptr<PatternMatcherImpl> impl_;
};
}  // namespace fusion
} // namespace ge

#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_MATCHER_H
