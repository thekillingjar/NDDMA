/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/multi_stream_share_graph.h"
#include "common/share_graph.h"
#include "common/ge_inner_attrs.h"
#include "framework/common/types.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "array_ops.h"
#include "runtime/mem.h"
#include "faker/fake_value.h"

namespace ge {
/*
    ┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
    │ data1  │ ───────> │ trans1 │ ───────> │ NetOutput │
    └────────┘          └────────┘          └───────────┘
      │                                       ∧
      │ (1,0)                                 │
      │                                       │
      │               ┌────────┐  (0,1)       │
      +───────────>   │ cast   │ ─────────────┘
                      └────────┘
 */
Graph MultiStreamShareGraph::TwoNodeGraphWithUserStreamLabel() {
    DEF_GRAPH(g1) {
        CHAIN(NODE("data1", "Data")->NODE("trans1", "TransData")->NODE("NetOutput", "NetOutput"));
        CHAIN(NODE("data1", "Data")->EDGE(0,0)->NODE("cast", "Cast")->NODE("NetOutput", "NetOutput"));
    };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  AttrUtils::SetInt(compute_graph->FindNode("data1")->GetOpDesc(), "index", 0);
  auto trans1 = compute_graph->FindNode("trans1");
  AttrUtils::SetStr(trans1->GetOpDesc(), "_user_stream_label", "test_label");
  auto net_output = compute_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"trans1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  gert::AddCompileResult(trans1, true,
                         "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                         "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  auto cast = compute_graph->FindNode("cast");
  cast->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  cast->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  return graph;
}
}  // namespace ge

