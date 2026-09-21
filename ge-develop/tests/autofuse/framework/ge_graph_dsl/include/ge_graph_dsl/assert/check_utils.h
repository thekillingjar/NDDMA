/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_31309AA0A4E44C009C22AD9351BF3410
#define INC_31309AA0A4E44C009C22AD9351BF3410

#include "ge_graph_dsl/ge.h"
#include "graph/compute_graph.h"

GE_NS_BEGIN

using GraphCheckFun = std::function<void(const ::GE_NS::ComputeGraphPtr &)>;
struct CheckUtils {
  static bool CheckGraph(const std::string &phase_id, const GraphCheckFun &fun);
  static void init();
};

GE_NS_END

#endif
