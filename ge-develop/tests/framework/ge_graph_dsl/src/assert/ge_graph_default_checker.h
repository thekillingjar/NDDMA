/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BCF4D96BE9FC48938DE7B7E93B551C54
#define BCF4D96BE9FC48938DE7B7E93B551C54

#include "ge_graph_dsl/ge.h"
#include "ge_graph_checker.h"
#include "graph/compute_graph.h"

GE_NS_BEGIN

using GraphCheckFun = std::function<void(const ::GE_NS::ComputeGraphPtr &)>;

struct GeGraphDefaultChecker : GeGraphChecker {
  GeGraphDefaultChecker(const std::string &, const GraphCheckFun &);

 private:
  const std::string &PhaseId() const override;
  void Check(const ge::ComputeGraphPtr &graph) const override;

 private:
  const std::string phase_id_;
  const GraphCheckFun check_fun_;
};

GE_NS_END

#endif
