/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H6DCF6C61_7C9B_4048_BB5D_E748142FF7F8
#define H6DCF6C61_7C9B_4048_BB5D_E748142FF7F8

#include "ge_graph_dsl/ge.h"
#include "easy_graph/graph/node_id.h"
#include "easy_graph/graph/box.h"
#include "graph/gnode.h"

GE_NS_BEGIN

struct OpBox : ::EG_NS::Box {
  ABSTRACT(OpDescPtr Build(const ::EG_NS::NodeId &) const);
};

GE_NS_END

#endif /* H6DCF6C61_7C9B_4048_BB5D_E748142FF7F8 */
