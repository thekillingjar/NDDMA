/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H7C82E219_BDEF_4480_A2D9_30F0590C8AC5
#define H7C82E219_BDEF_4480_A2D9_30F0590C8AC5

#include "easy_graph/graph/graph.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_desc_node_builder.h"
#include "graph/graph.h"
#include "graph/fast_graph/execute_graph.h"

GE_NS_BEGIN

Graph ToGeGraph(const ::EG_NS::Graph &graph);
ComputeGraphPtr ToComputeGraph(const ::EG_NS::Graph &graph);
ExecuteGraphPtr ToExecuteGraph(const ::EG_NS::Graph &graph);

#define DATA_EDGE(...) Data(__VA_ARGS__)
#define CTRL_EDGE(...) Ctrl(__VA_ARGS__)
#define NODE(...) Node(::GE_NS::OpDescNodeBuild(__VA_ARGS__))
#define EDGE(...) DATA_EDGE(__VA_ARGS__)

GE_NS_END

#endif
