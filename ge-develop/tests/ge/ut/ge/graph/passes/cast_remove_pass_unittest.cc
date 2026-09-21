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
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/format_optimize/cast_remove_pass.h"
#include "macro_utils/dt_public_unscope.h"

#include "anchor.h"
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/pass_manager.h"
#include "graph_builder_utils.h"
#include <string>
#include <iostream>
#include <vector>
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "omg/omg_inner_types.h"


using namespace testing;
using namespace ge;
using namespace std;

class UtestGraphPassesCastRemovePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/*
 *   data1    const1
 *      |       |
 *        add1
 *         |
 *     transdata1
 *         |
 *       cast1
 *         |
 *     transposed1
 *         |
 *       cast2
 *         |
 *     transdata2
 *         |
 *       conv2d
 */
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  auto transposed1 = builder.AddNode("transposed1", TRANSPOSED, 1, 1);
  auto cast2 = builder.AddNode("cast2", CAST, 1, 1);
  auto transdata2 = builder.AddNode("transdata2", TRANSDATA, 1, 1);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 1, 1);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, transposed1, 0);
  builder.AddDataEdge(transposed1, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, conv2d, 0);

  return builder.GetGraph();
}

TEST_F(UtestGraphPassesCastRemovePass, DoFuseProcess) {
  std::vector<NodePtr> nodes_to_fuse;

  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data", DATA, 1, 1);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto trans = builder.AddNode("trans", TRANSPOSE, 1, 1, FORMAT_NCHW, DT_FLOAT16);
  auto cast2 = builder.AddNode("cast2", CAST, 1, 1);
  cast2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  auto net = builder.AddNode("netout", NETOUTPUT, 1, 1);

  builder.AddDataEdge(data, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, trans, 0);
  builder.AddDataEdge(trans, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, net, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();

  map<string, string> options;

  CastRemovePass cast_remove_pass;
  DataType type = DT_FLOAT;
  nodes_to_fuse.emplace_back(cast1);
  nodes_to_fuse.emplace_back(trans);
  nodes_to_fuse.emplace_back(cast2);
  cast_remove_pass.RemoveCast(type, nodes_to_fuse);
  EXPECT_EQ(compute_graph->GetAllNodesSize(),3);
}

TEST_F(UtestGraphPassesCastRemovePass, success) {
  auto compute_graph = BuildGraph1();
  compute_graph->SetSessionID(0);

  auto data1 = compute_graph->FindNode("data1");
  CastRemovePass cast_pass;
  Status ret = cast_pass.Run(data1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 9);

  auto add1 = compute_graph->FindNode("add1");
  ret = cast_pass.Run(add1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 7);

  auto cast1 = compute_graph->FindNode("cast1");
  EXPECT_EQ(cast1, nullptr);

  auto transposed1 = compute_graph->FindNode("transposed1");
  EXPECT_EQ(transposed1, nullptr);

  NodePtr test_node;
  // test nullptr opdesc
  ret = cast_pass.Run(test_node);
  EXPECT_EQ(ret, PARAM_INVALID);

  test_node = nullptr;
  // test nullptr node
  ret = cast_pass.Run(test_node);
  EXPECT_EQ(ret, PARAM_INVALID);
}

/*
 *        ...
 *         |
 *     transdata1
 *         |
 *       cast1() -- (FP16->FP16)
 *         |
 *     transposed1
 *         |
 *       cast2 -- (FP16->INT32)
 *         |
 *        ...
 */
TEST_F(UtestGraphPassesCastRemovePass, RemoveRedundantCastNodeSuccess) {
  auto compute_graph = BuildGraph1();
  compute_graph->SetSessionID(0);

  auto cast_node2 = compute_graph->FindNode("cast2");
  EXPECT_NE(cast_node2, nullptr);
  auto input_desc = cast_node2->GetOpDesc()->GetInputDesc(0);
  input_desc.SetDataType(DT_FLOAT16);
  input_desc.SetOriginDataType(DT_FLOAT16);
  cast_node2->GetOpDesc()->UpdateInputDesc(0, input_desc);

  auto output_desc = cast_node2->GetOpDesc()->GetOutputDesc(0);
  output_desc.SetDataType(DT_INT32);
  output_desc.SetOriginDataType(DT_INT32);
  cast_node2->GetOpDesc()->UpdateOutputDesc(0, output_desc);

  // run on add1, cannot delete cast node
  auto add1 = compute_graph->FindNode("add1");
  CastRemovePass cast_pass;
  Status ret = cast_pass.Run(add1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 9);

  auto cast1 = compute_graph->FindNode("cast1");
  EXPECT_NE(cast1, nullptr);

  auto cast2 = compute_graph->FindNode("cast2");
  EXPECT_NE(cast2, nullptr);

  // remove redundant cast node: add1
  auto cast_node1 = compute_graph->FindNode("cast1");
  ret = cast_pass.Run(cast_node1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 8);

  cast1 = compute_graph->FindNode("cast1");
  EXPECT_EQ(cast1, nullptr);

  // failed remove cast node add2
  cast_node2 = compute_graph->FindNode("cast2");
  ret = cast_pass.Run(cast_node2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 8);

  cast1 = compute_graph->FindNode("cast2");
  EXPECT_NE(cast1, nullptr);
}

/*
 *   data1    const1
 *      |       |
 *        add1
 *         |
 *     layernorm1
 *         |
 *     transdata1
 *         |
 *       cast1
 *         |
 *     transposed1
 *         |
 *       conv2d
 */
TEST_F(UtestGraphPassesCastRemovePass, NodeWithMultiOutDataAnchorsButOneOutDataNodeRunSuccess) {
  auto builder = ut::GraphBuilder("g2");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  // 3 out data anchors
  auto layernorm1 = builder.AddNode("layernorm1", LAYERNORM, 3, 3);
  auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  auto transposed1 = builder.AddNode("transposed1", TRANSPOSED, 1, 1);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 1, 1);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, layernorm1, 0);
  builder.AddDataEdge(layernorm1, 1, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, transposed1, 0);
  builder.AddDataEdge(transposed1, 0, conv2d, 0);

  auto compute_graph = builder.GetGraph();
  compute_graph->SetSessionID(0);

  layernorm1 = compute_graph->FindNode("layernorm1");
  CastRemovePass cast_pass;
  Status ret = cast_pass.Run(layernorm1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 7);

  cast1 = compute_graph->FindNode("cast1");
  EXPECT_EQ(cast1, nullptr);
}

/*
 *   data1    const1
 *      |       |
 *        add1
 *         |
 *     layernorm1
 *  /fp16  |bool
 *     transdata1
 *         |bool
 *       cast1
 *         |fp16
 *     transposed1
 *         |fp16
 *       conv2d
 */
TEST_F(UtestGraphPassesCastRemovePass, NodeWithMultiOutDataAnchorsButOneOutDataNode_SkipRemove) {
  auto builder = ut::GraphBuilder("g2");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  // 3 out data anchors
  auto layernorm1 = builder.AddNode("layernorm1", LAYERNORM, 3, 3);
  layernorm1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  layernorm1->GetOpDesc()->MutableOutputDesc(1)->SetDataType(DT_BOOL);
  layernorm1->GetOpDesc()->MutableOutputDesc(2)->SetDataType(DT_FLOAT16);
  auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
  transdata1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_BOOL);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_BOOL);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  cast1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_BOOL);
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto transposed1 = builder.AddNode("transposed1", TRANSPOSED, 1, 1);
  transposed1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  transposed1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 1, 1);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, layernorm1, 0);
  builder.AddDataEdge(layernorm1, 1, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, transposed1, 0);
  builder.AddDataEdge(transposed1, 0, conv2d, 0);

  auto compute_graph = builder.GetGraph();
  compute_graph->SetSessionID(0);

  layernorm1 = compute_graph->FindNode("layernorm1");
  CastRemovePass cast_pass;
  Status ret = cast_pass.Run(layernorm1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 8);

  cast1 = compute_graph->FindNode("cast1");
  EXPECT_NE(cast1, nullptr);
}

/*
 *       data
 *         |float
 *        cast  const1
 *         |fp16  /
 *         reshape
 *           |fp16
 *         cast1
 *        /float \float
 *       netoutput
 *   链路中尾节点为单输出多引用场景。
 *
 */
TEST_F(UtestGraphPassesCastRemovePass, EndNodeMultiConsumer_SUCCESS) {
  auto builder = ut::GraphBuilder("g2");
  auto data = builder.AddNode("data", DATA, 0, 1, FORMAT_ND, DT_FLOAT);
  auto cast = builder.AddNode("cast", CAST, 1, 1, FORMAT_ND, DT_FLOAT);
  cast->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto reshape = builder.AddNode("reshape", RESHAPE, 2, 1, FORMAT_ND, DT_FLOAT16);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_ND, DT_FLOAT16);
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 2, 1, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(data, 0, cast, 0);
  builder.AddDataEdge(const1, 0, reshape, 1);
  builder.AddDataEdge(cast, 0, reshape, 0);
  builder.AddDataEdge(reshape, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, netoutput, 0);
  builder.AddDataEdge(cast1, 0, netoutput, 1);
  auto compute_graph = builder.GetGraph();
  compute_graph->SetSessionID(0);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 6);

  auto data_begin_node = compute_graph->FindNode("data");
  CastRemovePass cast_pass;
  Status ret = cast_pass.Run(data_begin_node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 4);

  cast1 = compute_graph->FindNode("cast1");
  EXPECT_EQ(cast1, nullptr);
}
