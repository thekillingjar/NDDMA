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

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/pass_manager.h"
#include "graph/utils/op_desc_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/control_flow_and_stream/flow_ctrl_pass.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestGraphPassesFlowCtrlPass : public testing::Test {
 protected:
  void SetUp() {
    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->Init(session_version, session_id, device_id, job_id));
  }

  void TearDown() { VarManagerPool::Instance().Destory(); }

 public:
  /// Set up a graph with the following network structure
  ///        IteratorGetNext
  ///              |
  ///          MemcpyAsync
  ///              |
  ///              A
  ///              |
  ///          NetOutput
  void MakeGraph(ge::ComputeGraphPtr &graph) {
    auto desc_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    ge::OpDescPtr op_desc_get_next = std::make_shared<ge::OpDesc>("IteratorGetNext", FRAMEWORKOP);

    op_desc_get_next->AddOutputDesc(desc);

    ge::OpDescPtr op_desc_memcpy = std::make_shared<ge::OpDesc>("MemcpyAsync", MEMCPYASYNC);
    op_desc_memcpy->AddInputDesc(desc);
    op_desc_memcpy->AddOutputDesc(desc);
    ge::AttrUtils::SetBool(op_desc_memcpy, ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, true);

    ge::OpDescPtr op_desc_a = std::make_shared<ge::OpDesc>("A", RESOURCEAPPLYMOMENTUM);
    op_desc_a->AddInputDesc(desc);
    op_desc_a->AddOutputDesc(desc);

    ge::OpDescPtr op_desc_gatherv2 = std::make_shared<ge::OpDesc>("GatherV2", GATHERV2);
    op_desc_gatherv2->AddInputDesc(desc);
    op_desc_gatherv2->AddOutputDesc(desc);

    ge::OpDescPtr op_desc_global_step = std::make_shared<ge::OpDesc>("global_step", VARIABLE);
    op_desc_global_step->AddOutputDesc(desc);

    ge::OpDescPtr op_desc_netout = std::make_shared<ge::OpDesc>("NetOutput", NETOUTPUT);
    ge::AttrUtils::SetInt(op_desc_netout, ATTR_NAME_TRUE_BRANCH_STREAM, TRUE_STREAM_ID);
    op_desc_netout->AddInputDesc(desc);
    op_desc_netout->AddInputDesc(desc);

    // add node
    ge::NodePtr get_next_node = graph->AddNode(op_desc_get_next);
    ge::NodePtr memcpy_node = graph->AddNode(op_desc_memcpy);
    ge::NodePtr node_a = graph->AddNode(op_desc_a);
    ge::NodePtr global_step = graph->AddNode(op_desc_global_step);
    ge::NodePtr gatherv2 = graph->AddNode(op_desc_gatherv2);
    ge::NodePtr netoutput = graph->AddNode(op_desc_netout);

    // add edge
    ge::GraphUtils::AddEdge(get_next_node->GetOutDataAnchor(0), memcpy_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(gatherv2->GetOutDataAnchor(0), netoutput->GetInDataAnchor(1));

    ge::GraphUtils::AddEdge(global_step->GetOutDataAnchor(0), gatherv2->GetInDataAnchor(0));
  }

  void AddSessionVariables(void) {
    static std::set<std::string> var_list = {
        NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
        NODE_NAME_FLOWCTRL_LOOP_COND,
        NODE_NAME_FLOWCTRL_LOOP_INCREMENT,
        NODE_NAME_FLOWCTRL_LOOP_RESETVALUE,
        NODE_NAME_GLOBAL_STEP,
    };

    uint8_t *dev_ptr = nullptr;
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
    for (std::string var_name : var_list) {
      EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM, nullptr));
    }
  }
};

TEST_F(UtestGraphPassesFlowCtrlPass, flow_ctrl_pass_success_test) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("FlowCtrlPassSuccess");
  graph->SetNeedIteration(true);

  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();

  AddSessionVariables();
  FlowCtrlPass flow_ctrl_pass;
  Status ret = flow_ctrl_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(19, graph->GetDirectNodesSize());

  int stream_switch_cnt = 0;
  int stream_activeCnt = 0;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == STREAMSWITCH) {
      stream_switch_cnt++;
    } else if (node->GetOpDesc()->GetType() == STREAMACTIVE) {
      stream_activeCnt++;
    }
  }
  EXPECT_EQ(stream_switch_cnt, 2);
  EXPECT_EQ(stream_activeCnt, 3);
}

TEST_F(UtestGraphPassesFlowCtrlPass, flow_ctrl_pass_success_var_node_add_before) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("FlowCtrlPassSuccess");
  graph->SetNeedIteration(true);

  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();

  AddSessionVariables();
  FlowCtrlPass flow_ctrl_pass;

  NodePtr loop_cond_node = flow_ctrl_pass.AddVariableNode(graph, NODE_NAME_FLOWCTRL_LOOP_COND);
  EXPECT_NE(loop_cond_node, nullptr);
  NodePtr loop_increment_node = flow_ctrl_pass.AddVariableNode(graph, NODE_NAME_FLOWCTRL_LOOP_INCREMENT);
  EXPECT_NE(loop_increment_node, nullptr);
  NodePtr loop_reset_node = flow_ctrl_pass.AddVariableNode(graph, NODE_NAME_FLOWCTRL_LOOP_RESETVALUE);
  EXPECT_NE(loop_reset_node, nullptr);
  NodePtr iter_per_loop_node = flow_ctrl_pass.AddVariableNode(graph, NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  EXPECT_NE(iter_per_loop_node, nullptr);
  Status ret = flow_ctrl_pass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGraphPassesFlowCtrlPass, flow_ctrl_pass_not_train) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("TestNotChange");
  graph->SetNeedIteration(false);

  FlowCtrlPass flow_ctrl_pass;
  Status ret = flow_ctrl_pass.Run(graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_fpbp_iterator_ctrl_without_var) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("TestNotChange");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();

  // must have NODE_NAME_FLOWCTRL_LOOP_PER_ITER
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(NODE_NAME_FLOWCTRL_LOOP_PER_ITER, tensor_desc,
                                                                   dev_ptr, RT_MEMORY_HBM, nullptr));
  // not add var
  FlowCtrlPass flow_ctrl_pass;
  Status ret = flow_ctrl_pass.Run(graph);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestGraphPassesFlowCtrlPass, run_add_special_node_iterator_ctrl_no_inanchor) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_PER_ITER");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();

  AddSessionVariables();
  FlowCtrlPass flow_ctrl_pass;
  NodePtr getnext_node = graph->FindNode("IteratorGetNext");
  NodePtr memcpy_node = graph->FindNode("MemcpyAsync");
  GraphUtils::RemoveEdge(getnext_node->GetOutDataAnchor(0), memcpy_node->GetInDataAnchor(0));
  Status ret = flow_ctrl_pass.Run(graph);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_fpbp_iterator_ctrl_without_loop_cond) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_COND");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();
  std::set<std::string> var_list = {
      NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
      NODE_NAME_FLOWCTRL_LOOP_INCREMENT,
      NODE_NAME_FLOWCTRL_LOOP_RESETVALUE,
      NODE_NAME_GLOBAL_STEP,
  };
  // must have NODE_NAME_FLOWCTRL_LOOP_PER_ITER
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  for (std::string var_name : var_list) {
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM, nullptr));
  }
  // not add var
  FlowCtrlPass flow_ctrl_pass;
  NodePtr pre_node = graph->FindNode("NetOutput");
  Status ret = flow_ctrl_pass.AddFpBpIteratorCtrl(graph, pre_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_fpbp_iterator_ctrl_without_loop_increment) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_INCREMENT");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();
  std::set<std::string> var_list = {
      NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
      NODE_NAME_FLOWCTRL_LOOP_COND,
      NODE_NAME_FLOWCTRL_LOOP_RESETVALUE,
      NODE_NAME_GLOBAL_STEP,
  };
  // must have NODE_NAME_FLOWCTRL_LOOP_PER_ITER
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  for (std::string var_name : var_list) {
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM, nullptr));
  }
  // not add var
  FlowCtrlPass flow_ctrl_pass;
  NodePtr pre_node = graph->FindNode("NetOutput");
  Status ret = flow_ctrl_pass.AddFpBpIteratorCtrl(graph, pre_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_fpbp_iterator_ctrl_without_loop_reset_value) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_RESETVALUE");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();
  std::set<std::string> var_list = {
      NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
      NODE_NAME_FLOWCTRL_LOOP_COND,
      NODE_NAME_FLOWCTRL_LOOP_INCREMENT,
      NODE_NAME_GLOBAL_STEP,
  };
  // must have NODE_NAME_FLOWCTRL_LOOP_PER_ITER
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  for (std::string var_name : var_list) {
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM, nullptr));
  }
  // not add var
  FlowCtrlPass flow_ctrl_pass;
  NodePtr pre_node = graph->FindNode("NetOutput");
  Status ret = flow_ctrl_pass.AddFpBpIteratorCtrl(graph, pre_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_fpbp_iterator_ctrl_without_loop_ref_iter) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_PER_ITER");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();
  std::set<std::string> var_list = {
      NODE_NAME_FLOWCTRL_LOOP_COND,
      NODE_NAME_FLOWCTRL_LOOP_INCREMENT,
      NODE_NAME_FLOWCTRL_LOOP_RESETVALUE,
      NODE_NAME_GLOBAL_STEP,
  };
  // must have NODE_NAME_FLOWCTRL_LOOP_PER_ITER
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  for (std::string var_name : var_list) {
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM,nullptr));
  }
  FlowCtrlPass flow_ctrl_pass;
  NodePtr pre_node = graph->FindNode("NetOutput");
  Status ret = flow_ctrl_pass.AddFpBpIteratorCtrl(graph, pre_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_special_node_iterator_ctrl_without_loop_cond) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_COND");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();
  std::set<std::string> var_list = {
      NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
      NODE_NAME_FLOWCTRL_LOOP_INCREMENT,
      NODE_NAME_FLOWCTRL_LOOP_RESETVALUE,
      NODE_NAME_GLOBAL_STEP,
  };
  // must have NODE_NAME_FLOWCTRL_LOOP_PER_ITER
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  for (std::string var_name : var_list) {
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM, nullptr));
  }

  FlowCtrlPass flow_ctrl_pass;
  NodePtr iter_per_loop_node = flow_ctrl_pass.AddVariableNode(graph, NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  EXPECT_NE(iter_per_loop_node, nullptr);
  NodePtr memcpy_node = graph->FindNode("MemcpyAsync");
  Status ret = flow_ctrl_pass.AddSpecialNodeIteratorCtrl(graph, memcpy_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_special_node_iterator_ctrl_without_loop_ref_iter) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_PER_ITER");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();
  std::set<std::string> var_list = {
      NODE_NAME_FLOWCTRL_LOOP_COND,
      NODE_NAME_FLOWCTRL_LOOP_INCREMENT,
      NODE_NAME_FLOWCTRL_LOOP_RESETVALUE,
      NODE_NAME_GLOBAL_STEP,
  };
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
  uint8_t *dev_ptr = nullptr;
  for (std::string var_name : var_list) {
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->SetVarAddr(var_name, tensor_desc, dev_ptr, RT_MEMORY_HBM, nullptr));
  }

  FlowCtrlPass flow_ctrl_pass;
  NodePtr loop_cond_node = flow_ctrl_pass.AddVariableNode(graph, NODE_NAME_FLOWCTRL_LOOP_COND);
  EXPECT_NE(loop_cond_node, nullptr);
  NodePtr memcpy_node = graph->FindNode("MemcpyAsync");
  Status ret = flow_ctrl_pass.AddSpecialNodeIteratorCtrl(graph, memcpy_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, add_special_node_iterator_ctrl_no_inchor) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_WITHOUT_LOOP_PER_ITER");
  graph->SetNeedIteration(true);
  // Create graph
  MakeGraph(graph);
  graph->TopologicalSorting();

  FlowCtrlPass flow_ctrl_pass;
  NodePtr getnext_node = graph->FindNode("IteratorGetNext");
  NodePtr memcpy_node = graph->FindNode("MemcpyAsync");
  GraphUtils::RemoveEdge(getnext_node->GetOutDataAnchor(0), memcpy_node->GetInDataAnchor(0));
  Status ret = flow_ctrl_pass.AddSpecialNodeIteratorCtrl(graph, memcpy_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, insert_assign_op_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_InsertAssignOp");

  FlowCtrlPass flow_ctrl_pass;
  GeTensorDesc tmp_geT_tensor_desc;
  NodePtr ref_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {tmp_geT_tensor_desc});
  NodePtr value_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {tmp_geT_tensor_desc});
  NodePtr add_node = flow_ctrl_pass.InsertAssignOp(graph, ASSIGNADD, "add_node", ref_node, value_node);
  EXPECT_NE(add_node, nullptr);
}

TEST_F(UtestGraphPassesFlowCtrlPass, insert_assign_op_ref_node_no_outanchor) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_InsertAssignOp");

  FlowCtrlPass flow_ctrl_pass;
  GeTensorDesc tmp_geT_tensor_desc;
  NodePtr ref_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {});
  NodePtr value_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {tmp_geT_tensor_desc});
  NodePtr add_node = flow_ctrl_pass.InsertAssignOp(graph, ASSIGNADD, "add_node", ref_node, value_node);
  EXPECT_EQ(add_node, nullptr);
}

TEST_F(UtestGraphPassesFlowCtrlPass, insert_assign_op_value_node_no_outanchor) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_InsertAssignOp");

  FlowCtrlPass flow_ctrl_pass;
  GeTensorDesc tmp_geT_tensor_desc;
  NodePtr ref_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {tmp_geT_tensor_desc});
  NodePtr value_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {});
  NodePtr add_node = flow_ctrl_pass.InsertAssignOp(graph, ASSIGNADD, "add_node", ref_node, value_node);
  EXPECT_EQ(add_node, nullptr);
}

TEST_F(UtestGraphPassesFlowCtrlPass, create_iter_ctrl_false_branch_insert_assign_op_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_CreateIterCtrlFalseBranch_InsertAssignOp_FAILED");

  FlowCtrlPass flow_ctrl_pass;
  GeTensorDesc tmp_geT_tensor_desc;
  NodePtr ref_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {tmp_geT_tensor_desc});
  NodePtr value_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {});
  NodePtr switch_node = flow_ctrl_pass.InsertOp(graph, STREAMSWITCH, "switch_node", {}, {});
  const OpDescPtr net_output = OpDescBuilder("NetOutput", NETOUTPUT).Build();
  auto net_output_node = graph->AddNode(net_output);
  EXPECT_NE(net_output_node, nullptr);
  Status ret = flow_ctrl_pass.CreateIterCtrlFalseBranch(graph, ref_node, value_node, net_output_node, switch_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, create_iter_ctrl_true_branch_insert_assign_op_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("CreateIterCtrlTrueBranch_InsertAssignOp_FAILED");

  FlowCtrlPass flow_ctrl_pass;
  GeTensorDesc tmp_geT_tensor_desc;
  NodePtr ref_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {tmp_geT_tensor_desc});
  NodePtr value_node = flow_ctrl_pass.InsertOp(graph, VARIABLE, "ref_node", {}, {});
  NodePtr switch_node = flow_ctrl_pass.InsertOp(graph, STREAMSWITCH, "switch_node", {}, {});
  Status ret = flow_ctrl_pass.CreateIterCtrlTrueBranch(graph, ref_node, value_node, switch_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesFlowCtrlPass, multi_iter_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Test_CreateIterCtrlFalseBranch_InsertModelExit_Succ");
  FlowCtrlPass flow_ctrl_pass;
  GeTensorDesc tensor_desc;
  NodePtr iterator1 = flow_ctrl_pass.InsertOp(graph, FRAMEWORKOP, "Iterator1", {}, {tensor_desc});
  NodePtr iterator2 = flow_ctrl_pass.InsertOp(graph, FRAMEWORKOP, "Iterator2", {}, {tensor_desc});
  NodePtr loop_cond = flow_ctrl_pass.InsertOp(graph, VARIABLE, "loop_cond", {}, {tensor_desc});
  NodePtr loop_reset = flow_ctrl_pass.InsertOp(graph, VARIABLE, "loop_reset", {}, {tensor_desc});
  NodePtr switch_node = flow_ctrl_pass.InsertOp(graph, STREAMSWITCH, "switch_node", {}, {});
  (void)AttrUtils::SetStr(iterator1->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
  (void)AttrUtils::SetStr(iterator2->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  const OpDescPtr net_output = OpDescBuilder("NetOutput", NETOUTPUT).Build();
  auto net_output_node = graph->AddNode(net_output);
  EXPECT_NE(net_output_node, nullptr);
  Status ret = flow_ctrl_pass.CreateIterCtrlFalseBranch(graph, loop_cond, loop_reset, net_output_node, switch_node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);
}
}  // namespace ge
