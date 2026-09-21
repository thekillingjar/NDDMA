/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/feature/set_ffts_plus_attr_pass.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/pass_manager.h"
using namespace testing;

namespace ge {
static void BuildFftsDynamicGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &dsp_graph, ComputeGraphPtr &ffts_graph) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
  };

  root_graph = ToComputeGraph(g1);
  const auto root_call_0 = root_graph->FindNode("PartitionedCall_0");
  EXPECT_NE(root_call_0, nullptr);

  auto dsp_graph_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto dsp_graph_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(g2) {
    CHAIN(NODE("dsp_graph/_arg_0", dsp_graph_data_0)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_0", IDENTITY)->EDGE(0, 0)
              ->NODE("dsp_graph/PartitionedCall_0", PARTITIONEDCALL)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_2", IDENTITY)->EDGE(0, 0)
              ->NODE("dsp_graph/Node_Output", NETOUTPUT)
    );
    CHAIN(NODE("dsp_graph/_arg_1", dsp_graph_data_1)->EDGE(0, 0)
              ->NODE("dsp_graph/trans_TransData_1", IDENTITY)->EDGE(0, 1)
              ->NODE("dsp_graph/PartitionedCall_0")
    );
  };

  dsp_graph = ToComputeGraph(g2);
  dsp_graph->SetParentNode(root_call_0);
  dsp_graph->SetParentGraph(root_graph);
  root_call_0->GetOpDesc()->AddSubgraphName("f");
  root_call_0->GetOpDesc()->SetSubgraphInstanceName(0, dsp_graph->GetName());
  root_graph->AddSubgraph(dsp_graph);
  const auto dsp_graph_call_0 = dsp_graph->FindNode("dsp_graph/PartitionedCall_0");
  EXPECT_NE(dsp_graph_call_0, nullptr);
  AttrUtils::SetBool(dsp_graph_call_0->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);

  auto sgt_graph_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sgt_graph_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto sgt_graph_conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
      .Attr(ATTR_NAME_IMPLY_TYPE, 1)           // domi::ImplyType::TVM
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto sgt_graph_relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
      .Attr(ATTR_NAME_IMPLY_TYPE, 1)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  DEF_GRAPH(g3) {
    CHAIN(NODE("sgt_graph/_arg_0", sgt_graph_data_0)->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", sgt_graph_conv_0)->EDGE(0, 0)
              ->NODE("sgt_graph/Relu", sgt_graph_relu_0)->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT)
    );
    CHAIN(NODE("sgt_graph/_arg_1", sgt_graph_data_1)->EDGE(0, 1)
              ->NODE("sgt_graph/Conv2D", sgt_graph_conv_0)
    );
  };

  ffts_graph = ToComputeGraph(g3);
  ffts_graph->SetParentNode(dsp_graph_call_0);
  ffts_graph->SetParentGraph(dsp_graph);
  dsp_graph_call_0->GetOpDesc()->AddSubgraphName("f");
  dsp_graph_call_0->GetOpDesc()->SetSubgraphInstanceName(0, ffts_graph->GetName());
  root_graph->AddSubgraph(ffts_graph);
}
class UtestGraphPassesSetFftsPlusAttrPass : public Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


TEST_F(UtestGraphPassesSetFftsPlusAttrPass, pass_run_success) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr dsp_graph;
  ComputeGraphPtr ffts_graph;
  BuildFftsDynamicGraph(root_graph, dsp_graph, ffts_graph);
  SetFftsPlusAttrPass set_ffts_plus_attr_pass;
  Status ret = set_ffts_plus_attr_pass.Run(ffts_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool b1 = false, b2 = false;
  for (NodePtr &node : ffts_graph->GetAllNodes()) {
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_ffts_plus", b1);
    EXPECT_EQ(true, b1);
    if (node->GetOpDesc()->GetType() == NOOP) {
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NOTASK, b2);
      EXPECT_EQ(true, b2);
    }
  }
}
}  // namespace ge

