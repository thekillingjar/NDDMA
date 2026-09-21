/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _ASC_GRAPH_DUMPER_CONTEXT_H_
#define _ASC_GRAPH_DUMPER_CONTEXT_H_
#include <vector>
#include <string>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
namespace ascir {
class AscGraphDumperContext {
 public:
  static AscGraphDumperContext &GetThreadLocalCtx();
  void AddWatchGraph(const std::string &suffix, const ge::AscGraph &graph);
  void ClearAllWatchGraphs();
  void DumpWatchedGraphs();

 private:
  AscGraphDumperContext() = default;
  ~AscGraphDumperContext() = default;
  std::list<std::pair<std::string, ge::AscGraph>> orderd_graphs_;
  // 最终dump的时候只能把dump触发时刻对应的状态的图落盘, 所以相同的图对象会去重
  std::unordered_map<const ge::AscGraph *, typename std::list<std::pair<std::string, ge::AscGraph>>::iterator>
      watched_graphs_;
};
}  // namespace ascir
#endif  //_ASC_GRAPH_DUMPER_CONTEXT_H_
