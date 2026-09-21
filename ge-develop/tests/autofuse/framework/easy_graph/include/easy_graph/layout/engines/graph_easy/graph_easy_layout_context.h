/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HB8CC77BE_6A2E_4EB4_BE59_CA85DE56C027
#define HB8CC77BE_6A2E_4EB4_BE59_CA85DE56C027

#include "easy_graph/eg.h"
#include <string>
#include <deque>

EG_NS_BEGIN

struct GraphEasyOption;
struct Graph;

struct GraphEasyLayoutContext {
  GraphEasyLayoutContext(const GraphEasyOption &);

  const Graph *GetCurrentGraph() const;

  void EnterGraph(const Graph &);
  void ExitGraph();

  void LinkBegin();
  void LinkEnd();

  bool InLinking() const;

  std::string GetGroupPath() const;
  const GraphEasyOption &GetOptions() const;

 private:
  std::deque<const Graph *> graphs_;
  const GraphEasyOption &options_;
  bool is_linking_{false};
};

EG_NS_END

#endif
