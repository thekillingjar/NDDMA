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

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/shape_optimize/infershape_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph_builder_utils.h"
#include "common/mem_conflict_share_graph.h"
#include "graph/resource_context_mgr.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "graph/debug/ge_attr_define.h"
#include "stub/gert_runtime_stub.h"

using namespace std;
using namespace testing;
namespace ge {
namespace {
const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };

/**
 *     data1  const_op
 *        \ /              -----------------
 *       while   -------- | data_1   data_2 |
 *     /     \.           |       \ /       |
 *    |      |            |     write2      |
 *   read   write1        -----------------
 */
ComputeGraphPtr BuildWhileGraph1() {
  const auto write_infer_func = [](Operator &op) {
    auto inference_context = op.GetInferenceContext();
    inference_context->AddChangedResourceKey("123");
    return GRAPH_SUCCESS;
  };
  const auto read_infer_func = [](Operator &op) {
    auto inference_context = op.GetInferenceContext();
    inference_context->RegisterReliedOnResourceKey("123");
    return GRAPH_SUCCESS;
  };
  // build sub graph
  auto builder_sub = ut::GraphBuilder("sub");
  auto data_1 = builder_sub.AddNode("data_1", DATA, 0, 1);
  auto data_2 = builder_sub.AddNode("data_2", DATA, 0, 1);
  auto write2 =  builder_sub.AddNode("write2", "TensorArrayWrite", 2, 1);
  auto out_0 =  builder_sub.AddNode("out", NETOUTPUT, 1, 1);
  write2->GetOpDesc()->AddInferFunc(write_infer_func);

  builder_sub.AddDataEdge(data_1, 0, write2, 0);
  builder_sub.AddDataEdge(data_2, 0, write2, 1);
  builder_sub.AddDataEdge(write2, 0, out_0, 0);
  auto sub_graph = builder_sub.GetGraph();
  sub_graph->SetName("while_sub");
  // build root graph
  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto const_op = builder.AddNode("const_op", CONSTANT, 0, 1);
  auto read = builder.AddNode("read", "TensorArrayRead", 1, 1);
  read->GetOpDesc()->AddInferFunc(read_infer_func);
  auto write1 = builder.AddNode("write1", "TensorArrayWrite", 1, 1);
  write1->GetOpDesc()->AddInferFunc(write_infer_func);
  // add while op
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1,1,1,1}));
  tensor_desc->SetFormat(FORMAT_ND);
  tensor_desc->SetDataType(DT_INT32);

  auto op_desc = std::make_shared<OpDesc>("while", WHILE);
  for (int i = 0; i < 2; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < 2; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  AttrUtils::SetBool(op_desc,"_need_infer_again", true);
  op_desc->AddSubgraphName(sub_graph->GetName());
  op_desc->SetSubgraphInstanceName(0,sub_graph->GetName());
  auto root_graph = builder.GetGraph();
  auto while_op = root_graph->AddNode(op_desc);

  builder.AddDataEdge(data, 0, while_op, 0);
  builder.AddDataEdge(const_op, 0, while_op, 1);
  builder.AddDataEdge(while_op, 0, read, 0);
  builder.AddDataEdge(while_op, 1, write1, 0);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(while_op);
  root_graph->AddSubgraph(sub_graph);
  return root_graph;
}

/*
 *             netoutput1
 *                |
 *              relu1
 *                |
 *            Add1(locked)
 *            /        \
 *         var1(NCHW)  var2(NCHW)
 */
ut::GraphBuilder BuildGraphWithShapeLocked() {
  auto builder = ut::GraphBuilder("glocked");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1, FORMAT_NCHW, DT_INT8, {1, 1, 28, 28});
  auto var2 = builder.AddNode("var2", "Variable", 0, 1, FORMAT_NCHW, DT_INT8, {1, 1, 28, 28});
  auto add1 = builder.AddNode("add1", "Add", 2, 1, FORMAT_ND, DT_INT8, {-1, -1, 28, 28});
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1, FORMAT_NCHW, DT_INT8);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  auto add1_op = OpDescUtils::CreateOperatorFromNode(add1);
  add1_op.SetAttr(ATTR_NAME_OUT_SHAPE_LOCKED , true);

  var1->GetOpDescBarePtr()->AddInferFunc(stub_func);
  var2->GetOpDescBarePtr()->AddInferFunc(stub_func);
  relu1->GetOpDescBarePtr()->AddInferFunc(stub_func);
  netoutput1->GetOpDescBarePtr()->AddInferFunc(stub_func);

  builder.AddDataEdge(var1, 0, add1, 0);
  builder.AddDataEdge(var2, 0, add1, 1);
  builder.AddDataEdge(add1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput1, 0);

  return builder;
}
}

class UtestGraphInfershapePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph.AddNode(op_desc);
}

TEST_F(UtestGraphInfershapePass, infershape_pass_failed) {
  int event_log_level;
  int log_level_backup = dlog_getlevel(GE_MODULE_NAME, &event_log_level);
  dlog_setlevel(GE_MODULE_NAME, 1, event_log_level);
  GeTensorDesc ge_tensor_desc(GeShape({-2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{0,-1},{2,2},{3,3},{4,4}};
  ge_tensor_desc.SetShapeRange(shape_range);
  ge_tensor_desc.SetOriginShapeRange(shape_range);
  string type = "AddN";
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", type);
  addn_op_desc->AddInputDesc(ge_tensor_desc);
  addn_op_desc->AddOutputDesc(ge_tensor_desc);
  auto graph = std::make_shared<ComputeGraph>("test");
  auto addn_node = std::make_shared<Node>(addn_op_desc, graph);
  addn_node->Init();

  InferShapePass infershape_pass;
  EXPECT_EQ(infershape_pass.Run(addn_node), GRAPH_FAILED);
  dlog_setlevel(GE_MODULE_NAME, log_level_backup, event_log_level);
}

TEST_F(UtestGraphInfershapePass, stop_node_for_while_loop) {
/*******************************************************************************
 *      Exit         Identify
 *        \         /       \.
 *         \       /         \.
 *          Switch           Add
 *         /     |            |
 *        /      |            |
 *       /       |            |
 *  LoopCond     |            |
 *      \        |            |
 *       \       |            |
 *        \      |            |
 *       Less    |            |
 *          \    |       NextIteration
 *           \   |            |
 *            \  |            |
 *            Merge <---------|
 *              |
 *              |
 *            Enter
 ******************************************************************************/
  auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto data1 = CreateNode(*graph, "data", DATA, 1, 1);
  auto enter1 = CreateNode(*graph, "enter", ENTER, 1, 1);
  auto merge1 = CreateNode(*graph, "merge", MERGE, 2, 2);
  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);
  auto loop1 = CreateNode(*graph, "loopcond", LOOPCOND, 1, 1);
  auto switch1 = CreateNode(*graph, "switch", SWITCH, 2, 2);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 1, 1);
  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  auto next1 = CreateNode(*graph, "next", NEXTITERATION, 1, 1);
  auto exit1 = CreateNode(*graph, "exit", EXIT, 1, 1);
  auto value0 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto value1 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), enter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(enter1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), loop1->GetInDataAnchor(0));

  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch1->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), switch1->GetInDataAnchor(0));

  GraphUtils::AddEdge(switch1->GetOutDataAnchor(0), exit1->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch1->GetOutDataAnchor(1), ident1->GetInDataAnchor(0));

  GraphUtils::AddEdge(ident1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), next1->GetInDataAnchor(0));

  GraphUtils::AddEdge(next1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
  GraphUtils::AddEdge(exit1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  GEPass ge_passes(graph);
  NamesToPass names_to_passes;
  InferShapePass infer_shape_pass;
  names_to_passes.emplace_back("InferShapePass", &infer_shape_pass);

  EXPECT_EQ(infer_shape_pass.Run(switch1), SUCCESS);
  auto suspend_nodes = infer_shape_pass.GetNodesSuspend();
  auto exit_node = graph->FindNode("exit");
  EXPECT_EQ(suspend_nodes.count(exit_node), 1);
  infer_shape_pass.OnSuspendNodesLeaked();
  auto resume_nodes = infer_shape_pass.GetNodesResume();
  EXPECT_EQ(resume_nodes.count(exit_node), 1);
}
TEST_F(UtestGraphInfershapePass, update_tensordesc_when_changed) {
  GeTensorDesc src_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dst_ge_tensor_desc(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDescPtr src_tensor_desc_ptr = std::make_shared<GeTensorDesc>(src_ge_tensor_desc);
  GeTensorDescPtr dst_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  bool changed = false;
  infershape_pass.UpdateTensorDesc(src_tensor_desc_ptr, dst_tensor_desc_ptr, changed);
  EXPECT_EQ(changed, true);
  EXPECT_EQ(dst_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST_F(UtestGraphInfershapePass, update_tensordesc_when_not_changed) {
  GeTensorDesc src_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDescPtr src_tensor_desc_ptr = std::make_shared<GeTensorDesc>(src_ge_tensor_desc);
  GeTensorDescPtr dst_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  bool changed = false;
  infershape_pass.UpdateTensorDesc(src_tensor_desc_ptr, dst_tensor_desc_ptr, changed);
  EXPECT_TRUE(changed == false);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_failed) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc ge_tensor_desc2(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphs({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr}, dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_get_unknown_rank) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphs({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr}, dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), UNKNOWN_RANK);

  std::vector<std::pair<int64_t, int64_t>> ge_tensor_desc1_ptr_shape_range;
  EXPECT_EQ(ge_tensor_desc1_ptr->GetShapeRange(ge_tensor_desc1_ptr_shape_range), GRAPH_SUCCESS);
  EXPECT_EQ(ge_tensor_desc1_ptr_shape_range.empty(), true);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_get_unknown_shape) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphs({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr}, dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({-1,2,3,4}));
  // todo shape range?
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_failed) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc ge_tensor_desc2(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_with_dynamic_shape_failed) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, -1, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc ge_tensor_desc2(GeShape({1, -1, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_failed_shape_size_overflow) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({INT64_MAX, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_success) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({2,2,3,4}));
}

TEST_F(UtestGraphInfershapePass, repass_relied_node_if_resource_changed) {
  auto graph = BuildWhileGraph1();
  ResourceContextMgr resource_context_mgr;
  GraphRebuildStateCtrl resource_op_access_ctrl;
  InferShapePass infershape_pass(&resource_op_access_ctrl, &resource_context_mgr);
  auto read_node = graph->FindNode("read");
  EXPECT_EQ(infershape_pass.Run(read_node), SUCCESS);
  auto relied_nodes = resource_context_mgr.MutableNodesReliedOnResource("123");
  EXPECT_EQ(relied_nodes.size(), 1);
  EXPECT_EQ((*relied_nodes.begin())->GetName(), "read");

  // test write1 change resource, read get repassed. both two node in curr graph
  auto write1_node = graph->FindNode("write1");
  EXPECT_EQ(infershape_pass.Run(write1_node), SUCCESS);
  auto need_repass_immediate_set = infershape_pass.GetNodesNeedRePassImmediately();
  EXPECT_EQ(need_repass_immediate_set.size(), 1);
  EXPECT_EQ((*need_repass_immediate_set.begin())->GetName(), "read");

  // test write2 change resource, read get repassed. they are in different graph
  NodePtr write2_node = nullptr;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "write2") {
      write2_node = node;
    }
  }
  EXPECT_NE(write2_node, nullptr);
  EXPECT_EQ(infershape_pass.Run(write2_node), SUCCESS);
  auto global_need_repass_immediate_set = infershape_pass.GetGlobalNodesNeedRePassImmediately();
  EXPECT_EQ(global_need_repass_immediate_set.size(), 1);
  EXPECT_EQ((*global_need_repass_immediate_set.begin())->GetName(), "read");
}

TEST_F(UtestGraphInfershapePass, ForbiddenReCompile_WhenAllowMultiGraphParallelCompile) {
  auto graph = BuildWhileGraph1();
  ResourceContextMgr resource_context_mgr;
  GraphRebuildStateCtrl resource_op_access_ctrl;
  InferShapePass infershape_pass(&resource_op_access_ctrl, &resource_context_mgr);
  auto read_node = graph->FindNode("read");
  EXPECT_EQ(infershape_pass.Run(read_node), SUCCESS);
  auto relied_nodes = resource_context_mgr.MutableNodesReliedOnResource("123");
  EXPECT_EQ(relied_nodes.size(), 1);
  EXPECT_EQ((*relied_nodes.begin())->GetName(), "read");

  // test write1 change resource, read get repassed. both two node in curr graph
  auto write1_node = graph->FindNode("write1");
  EXPECT_EQ(infershape_pass.Run(write1_node), SUCCESS);
  auto need_repass_immediate_set = infershape_pass.GetNodesNeedRePassImmediately();
  EXPECT_EQ(need_repass_immediate_set.size(), 1);
  EXPECT_EQ((*need_repass_immediate_set.begin())->GetName(), "read");

  // test write2 change resource, read get repassed. they are in different graph
  auto graph2 = BuildWhileGraph1();
  graph2->SetName("i_am_a_new_graph, ha ha ha");
  NodePtr write2_node = nullptr;
  for (const auto &node : graph2->GetAllNodes()) {
    if (node->GetName() == "write2") {
      write2_node = node;
    }
  }
  OptionSetter setter{{OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE, "1"}};
  EXPECT_NE(write2_node, nullptr);
  gert::GertRuntimeStub stub;
  EXPECT_NE(infershape_pass.Run(write2_node), SUCCESS);
  auto log_ret = stub.GetSlogStub().FindLog(DLOG_ERROR, "Graph g1 has node read relied on resource 123. Resource changes trigger graph rebuild, which is disabled when ge.AllowMultiGraphParallelCompile is set to \"1\"");
  EXPECT_NE(log_ret, -1);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_subgraph_multiDims_success) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = 
    infershape_pass.UpdateOutputFromSubgraphsForSubgraphMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                  dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({-1,2,3,4}));
  std::vector<std::pair<int64_t, int64_t>> range;
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShapeRange(range), GRAPH_SUCCESS);
  EXPECT_EQ(range[0].first, 1);
  EXPECT_EQ(range[0].second, 2);
}


TEST_F(UtestGraphInfershapePass, node_with_out_shape_locked) {
  auto builder = BuildGraphWithShapeLocked();
  auto graph = builder.GetGraph();
  ResourceContextMgr resource_context_mgr;
  GraphRebuildStateCtrl resource_op_access_ctrl;
  InferShapePass infershape_pass(&resource_op_access_ctrl, &resource_context_mgr);
  auto var1_node = graph->FindNode("var1");
  EXPECT_EQ(infershape_pass.Run(var1_node), SUCCESS);
  auto var_tensor_desc = var1_node->GetOpDesc()->GetOutputDesc(0);
  EXPECT_EQ(var_tensor_desc.GetShape().GetDims(),  std::vector<int64_t>({1, 1, 28, 28}));

  auto var2_node = graph->FindNode("var2");
  EXPECT_EQ(infershape_pass.Run(var2_node), SUCCESS);
  var_tensor_desc = var2_node->GetOpDesc()->GetOutputDesc(0);
  EXPECT_EQ(var_tensor_desc.GetShape().GetDims(),  std::vector<int64_t>({1, 1, 28, 28}));

  auto add1_node = graph->FindNode("add1");
  EXPECT_EQ(infershape_pass.Run(add1_node), SUCCESS);
  auto add_out_tensor_desc = add1_node->GetOpDesc()->GetOutputDesc(0);
  EXPECT_EQ(add_out_tensor_desc.GetShape().GetDims(),  std::vector<int64_t>({-1, -1, 28, 28}));
  auto add_in_tensor_desc0 = add1_node->GetOpDesc()->GetInputDesc(0);
  EXPECT_EQ(add_in_tensor_desc0.GetShape().GetDims(),  std::vector<int64_t>({1, 1, 28, 28}));
  auto add_in_tensor_desc1 = add1_node->GetOpDesc()->GetInputDesc(1);
  EXPECT_EQ(add_in_tensor_desc1.GetShape().GetDims(),  std::vector<int64_t>({1, 1, 28, 28}));

  auto relu1_node = graph->FindNode("relu1");
  auto relu1_in_tensor_desc = relu1_node->GetOpDesc()->GetInputDesc(0);
  EXPECT_EQ(relu1_in_tensor_desc.GetShape().GetDims(),  std::vector<int64_t>({-1, -1, 28, 28}));

  EXPECT_EQ(infershape_pass.Run(relu1_node), SUCCESS);
  EXPECT_EQ(relu1_in_tensor_desc.GetShape().GetDims(),  std::vector<int64_t>({-1, -1, 28, 28}));
}
TEST_F(UtestGraphInfershapePass, infershape_pass_infer_shape_and_type1) {
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 0, 1, ge::FORMAT_NCHW);
  auto aipp_op = builder.AddNode("aipp", AIPP, 1, 1, ge::FORMAT_NCHW);
  auto conv = builder.AddNode("conv", CONV2D, 1, 1, ge::FORMAT_NCHW);
  AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);

  builder.AddDataEdge(data, 0, aipp_op, 0);
  builder.AddDataEdge(aipp_op, 0, conv, 0);
  InferShapePass infershape_pass;
  EXPECT_EQ(infershape_pass.InferShapeAndType(aipp_op), ge::GRAPH_SUCCESS);
}
}  // namespace ge
