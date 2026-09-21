/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HINT_GRAPH_INFO_COMPLETE_H
#define HINT_GRAPH_INFO_COMPLETE_H

#include "ascendc_ir.h"
#include "ascgen_log.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
struct ExpressionComparator {
  bool operator()(const ge::Expression &lhs, const ge::Expression &rhs) const {
    return ge::SymbolicUtils::ToString(lhs) < ge::SymbolicUtils::ToString(rhs);
  }
};
using SizeVarSet = std::set<ge::Expression, ExpressionComparator>;

class AscGraphInfoComplete {
 public:
  /**
   * 根据HintGraph，设置ImplGraph中的Api信息
   * @param [in,out] optimize_graph 优化后的图，同时也将api信息设置在这个图上
   */
  static Status CompleteApiInfo(const ge::AscGraph &optimize_graph);

  static void AppendOriginalSizeVar(const ge::AscGraph &graph, SizeVarSet &size_vars);
};
}  // namespace optimize

#endif  // HINT_GRAPH_INFO_COMPLETE_H
