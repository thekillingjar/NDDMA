/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_EASY_GRAPH_H
#define AUTOFUSE_EASY_GRAPH_H

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

namespace ge {
class EaseAscGraph {
 public:
  explicit EaseAscGraph(ge::AscGraph &graph) : asc_graph_(graph) {}
  EaseAscGraph &Loops(const std::vector<ge::Symbol> &loops);
  EaseAscGraph &Broadcast(const std::string &name, const std::set<size_t> &brc_index);
  // 暂时只支持elewise + brc, 不支持非连续搬运
  void Build();

 private:
  ge::AscGraph &asc_graph_;
  std::unordered_map<ge::Node *, std::set<size_t>> size1_node_to_index_;
};
}  // namespace ge

#endif  // AUTOFUSE_EASY_GRAPH_H
