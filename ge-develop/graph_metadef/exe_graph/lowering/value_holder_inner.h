/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_EXE_GRAPH_LOWERING_VALUE_HOLDER_INNER_H_
#define METADEF_CXX_EXE_GRAPH_LOWERING_VALUE_HOLDER_INNER_H_
#include <deque>
#include "exe_graph/lowering/builtin_node_types.h"
#include "exe_graph/lowering/graph_frame.h"
namespace gert {
namespace bg {
void SetCurrentFrame(GraphFrame *frame);
GraphFrame *GetCurrentFrame();
std::deque<std::unique_ptr<GraphFrame>> &GetGraphFrames();
}  // namespace bg
}  // namespace gert
#endif  // METADEF_CXX_EXE_GRAPH_LOWERING_VALUE_HOLDER_INNER_H_
