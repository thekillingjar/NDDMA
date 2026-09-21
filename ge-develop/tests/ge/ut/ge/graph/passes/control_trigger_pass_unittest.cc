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
#include "graph/passes/control_flow_and_stream/control_trigger_pass.h"
#include <string>

#include "common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/node_adapter.h"
#include "graph/node.h"
#include "common/context/local_context.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;

class UtestControlTriggerPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {
ComputeGraphPtr BuildNormalGraph() {
  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub1_data_0", sub1_data_0)->NODE("add", ADD)->NODE("less", LESS));
    CHAIN(NODE("sub1_data_1", sub1_data_1)->NODE("add", ADD));
    CHAIN(NODE("sub1_const_0", CONSTANTOP)->NODE("less", LESS)->NODE("netoutput", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub2_data_0", sub2_data_0)->NODE("add", ADD)->NODE("less", LESS));
    CHAIN(NODE("sub2_data_1", sub2_data_1)->NODE("add", ADD));
    CHAIN(NODE("sub2_const_0", CONSTANTOP)->NODE("less", LESS)->NODE("netoutput", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
  };

  sub_1.Layout();
  sub_2.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  return compute_graph;
}

void make_graph_no_control_trigger(ComputeGraphPtr &graph) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto add_desc = std::make_shared<OpDesc>("Add", ADD);
  add_desc->AddInputDesc(scalar_tensor);
  add_desc->AddInputDesc(scalar_tensor);
  add_desc->AddOutputDesc(scalar_tensor);
  auto add_node = graph->AddNode(add_desc);

  auto output_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  output_desc->AddInputDesc(scalar_tensor);
  output_desc->AddOutputDesc(scalar_tensor);
  auto output_node = graph->AddNode(output_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
}

void make_control_trigger_node(ComputeGraphPtr &graph, NodePtr &control_trigger_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto less_desc1 = std::make_shared<OpDesc>("Less1", LESS);
  less_desc1->AddInputDesc(scalar_tensor);
  less_desc1->AddInputDesc(scalar_tensor);
  less_desc1->AddOutputDesc(bool_tensor);
  auto less_node1 = graph->AddNode(less_desc1);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto switch_desc1 = std::make_shared<OpDesc>("Switch1", SWITCH);
  switch_desc1->AddInputDesc(scalar_tensor);
  switch_desc1->AddInputDesc(bool_tensor);
  switch_desc1->AddOutputDesc(scalar_tensor);
  switch_desc1->AddOutputDesc(scalar_tensor);
  auto switch_node1 = graph->AddNode(switch_desc1);

  auto identity_f_desc = std::make_shared<OpDesc>("IdentityF", IDENTITY);
  identity_f_desc->AddInputDesc(scalar_tensor);
  identity_f_desc->AddOutputDesc(scalar_tensor);
  auto identity_f_node = graph->AddNode(identity_f_desc);

  auto identity_t_desc = std::make_shared<OpDesc>("IdentityT", IDENTITY);
  identity_t_desc->AddInputDesc(scalar_tensor);
  identity_t_desc->AddOutputDesc(scalar_tensor);
  auto identity_t_node = graph->AddNode(identity_t_desc);

  auto identity_1_desc = std::make_shared<OpDesc>("Identity1", IDENTITY);
  identity_1_desc->AddInputDesc(scalar_tensor);
  identity_1_desc->AddOutputDesc(scalar_tensor);
  auto identity_1_node = graph->AddNode(identity_1_desc);

  auto identity_2_desc = std::make_shared<OpDesc>("Identity2", IDENTITY);
  identity_2_desc->AddInputDesc(scalar_tensor);
  identity_2_desc->AddOutputDesc(scalar_tensor);
  auto identity_2_node = graph->AddNode(identity_2_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node1->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node1->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), identity_f_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_t_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), identity_1_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_2_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), identity_f_node->GetInControlAnchor());

  (void)GraphUtils::AddEdge(switch_node1->GetOutControlAnchor(), identity_2_node->GetInControlAnchor());

  (void)GraphUtils::AddEdge(identity_f_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  (void)GraphUtils::AddEdge(identity_t_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  (void)GraphUtils::AddEdge(identity_1_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_cond_branch(ComputeGraphPtr &graph, bool branch_flag, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(branch_flag ? SWITCH_TRUE_OUTPUT : SWITCH_FALSE_OUTPUT),
                            identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_out_cond_branch_1(ComputeGraphPtr &graph, NodePtr &control_trigger_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto merge_desc = std::make_shared<OpDesc>("Merge", MERGE);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(int_tensor);
  auto merge_node = graph->AddNode(merge_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), merge_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), merge_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(merge_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_out_cond_branch_2(ComputeGraphPtr &graph, NodePtr &control_trigger_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto merge_desc = std::make_shared<OpDesc>("Merge", MERGE);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(int_tensor);
  auto merge_node = graph->AddNode(merge_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), merge_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), merge_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_no_valid_switch(ComputeGraphPtr &graph) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto pred_desc = std::make_shared<OpDesc>("LoopCond", LOOPCOND);
  pred_desc->AddInputDesc(bool_tensor);
  pred_desc->AddOutputDesc(bool_tensor);
  auto pred_node = graph->AddNode(pred_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto identity_t_desc = std::make_shared<OpDesc>("IdentityT", IDENTITY);
  identity_t_desc->AddInputDesc(scalar_tensor);
  identity_t_desc->AddOutputDesc(scalar_tensor);
  auto identity_t_node = graph->AddNode(identity_t_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  auto control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), pred_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(pred_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_t_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(identity_t_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_enter(ComputeGraphPtr &graph) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto enter_desc = std::make_shared<OpDesc>("Enter", ENTER);
  enter_desc->AddInputDesc(scalar_tensor);
  enter_desc->AddOutputDesc(scalar_tensor);
  AttrUtils::SetStr(enter_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto enter_node = graph->AddNode(enter_desc);

  auto pred_desc = std::make_shared<OpDesc>("LoopCond", LOOPCOND);
  pred_desc->AddInputDesc(bool_tensor);
  pred_desc->AddOutputDesc(bool_tensor);
  auto pred_node = graph->AddNode(pred_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  auto control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), enter_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(enter_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(pred_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_no_in_cond_anchor(ComputeGraphPtr &graph) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  auto control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_no_pred_cond_anchor(ComputeGraphPtr &graph) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  auto control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_switch_ctrl_edge(ComputeGraphPtr &graph, NodePtr &control_trigger_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  auto switch_node = graph->AddNode(switch_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_nested_cond_branch_1(ComputeGraphPtr &graph, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto in_switch_desc = std::make_shared<OpDesc>("InSwitch", SWITCH);
  in_switch_desc->AddInputDesc(scalar_tensor);
  in_switch_desc->AddInputDesc(bool_tensor);
  in_switch_desc->AddOutputDesc(scalar_tensor);
  in_switch_desc->AddOutputDesc(scalar_tensor);
  auto in_switch_node = graph->AddNode(in_switch_desc);

  auto merge_desc = std::make_shared<OpDesc>("Merge", MERGE);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(int_tensor);
  auto merge_node = graph->AddNode(merge_desc);

  auto out_switch_desc = std::make_shared<OpDesc>("OutSwitch", SWITCH);
  out_switch_desc->AddInputDesc(scalar_tensor);
  out_switch_desc->AddInputDesc(bool_tensor);
  out_switch_desc->AddOutputDesc(scalar_tensor);
  out_switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(out_switch_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), in_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), in_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(in_switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), merge_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(in_switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), merge_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(merge_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_nested_cond_branch_2(ComputeGraphPtr &graph, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto in_switch_desc = std::make_shared<OpDesc>("InSwitch", SWITCH);
  in_switch_desc->AddInputDesc(scalar_tensor);
  in_switch_desc->AddInputDesc(bool_tensor);
  in_switch_desc->AddOutputDesc(scalar_tensor);
  in_switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(in_switch_desc);

  auto out_switch_desc = std::make_shared<OpDesc>("OutSwitch", SWITCH);
  out_switch_desc->AddInputDesc(scalar_tensor);
  out_switch_desc->AddInputDesc(bool_tensor);
  out_switch_desc->AddOutputDesc(scalar_tensor);
  out_switch_desc->AddOutputDesc(scalar_tensor);
  auto out_switch_node = graph->AddNode(out_switch_desc);

  auto identity_t_desc = std::make_shared<OpDesc>("IdentityT", IDENTITY);
  identity_t_desc->AddInputDesc(scalar_tensor);
  identity_t_desc->AddOutputDesc(scalar_tensor);
  auto identity_t_node = graph->AddNode(identity_t_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), out_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), out_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(out_switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_t_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(identity_t_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_cascade_cond_branch(ComputeGraphPtr &graph, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto less_desc = std::make_shared<OpDesc>("Less", LESS);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddInputDesc(scalar_tensor);
  less_desc->AddOutputDesc(bool_tensor);
  auto less_node = graph->AddNode(less_desc);

  auto pre_switch_desc = std::make_shared<OpDesc>("InSwitch", SWITCH);
  pre_switch_desc->AddInputDesc(scalar_tensor);
  pre_switch_desc->AddInputDesc(bool_tensor);
  pre_switch_desc->AddOutputDesc(scalar_tensor);
  pre_switch_desc->AddOutputDesc(scalar_tensor);
  auto pre_switch_node = graph->AddNode(pre_switch_desc);

  auto merge_desc = std::make_shared<OpDesc>("Merge", MERGE);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(int_tensor);
  auto merge_node = graph->AddNode(merge_desc);

  auto out_switch_desc = std::make_shared<OpDesc>("OutSwitch", SWITCH);
  out_switch_desc->AddInputDesc(scalar_tensor);
  out_switch_desc->AddInputDesc(bool_tensor);
  out_switch_desc->AddOutputDesc(scalar_tensor);
  out_switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(out_switch_desc);

  auto identity_t_desc = std::make_shared<OpDesc>("IdentityT", IDENTITY);
  identity_t_desc->AddInputDesc(scalar_tensor);
  identity_t_desc->AddOutputDesc(scalar_tensor);
  auto identity_t_node = graph->AddNode(identity_t_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), pre_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), pre_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(pre_switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), merge_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pre_switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), merge_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_t_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(identity_t_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_loop_branch(ComputeGraphPtr &graph, bool branch_flag, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto pred_desc = std::make_shared<OpDesc>("LoopCond", LOOPCOND);
  pred_desc->AddInputDesc(bool_tensor);
  pred_desc->AddOutputDesc(bool_tensor);
  auto pred_node = graph->AddNode(pred_desc);

  auto merge_x_desc = std::make_shared<OpDesc>("MergeX", MERGE);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_x_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_x_node = graph->AddNode(merge_x_desc);

  auto merge_y_desc = std::make_shared<OpDesc>("MergeY", MERGE);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_y_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_y_node = graph->AddNode(merge_y_desc);

  auto switch_desc = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_desc->AddInputDesc(scalar_tensor);
  switch_desc->AddInputDesc(bool_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto exit_desc = std::make_shared<OpDesc>("Exit", EXIT);
  exit_desc->AddInputDesc(scalar_tensor);
  exit_desc->AddOutputDesc(scalar_tensor);
  auto exit_node = graph->AddNode(exit_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), merge_x_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_x_node->GetOutDataAnchor(0), pred_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pred_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), merge_y_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_y_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), exit_node->GetInDataAnchor(0));

  if (branch_flag) {
    (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  } else {
    (void)GraphUtils::AddEdge(exit_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  }
}

void make_graph_nested_loop_branch(ComputeGraphPtr &graph, bool branch_flag, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto out_pred_desc = std::make_shared<OpDesc>("OutLoopCond", LOOPCOND);
  out_pred_desc->AddInputDesc(bool_tensor);
  out_pred_desc->AddOutputDesc(bool_tensor);
  auto out_pred_node = graph->AddNode(out_pred_desc);

  auto out_merge_x_desc = std::make_shared<OpDesc>("OutMergeX", MERGE);
  out_merge_x_desc->AddInputDesc(scalar_tensor);
  out_merge_x_desc->AddInputDesc(scalar_tensor);
  out_merge_x_desc->AddOutputDesc(scalar_tensor);
  out_merge_x_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(out_merge_x_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto out_merge_x_node = graph->AddNode(out_merge_x_desc);

  auto out_merge_y_desc = std::make_shared<OpDesc>("OutMergeY", MERGE);
  out_merge_y_desc->AddInputDesc(scalar_tensor);
  out_merge_y_desc->AddInputDesc(scalar_tensor);
  out_merge_y_desc->AddOutputDesc(scalar_tensor);
  out_merge_y_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(out_merge_y_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto out_merge_y_node = graph->AddNode(out_merge_y_desc);

  auto out_switch_x_desc = std::make_shared<OpDesc>("OutSwitchX", SWITCH);
  out_switch_x_desc->AddInputDesc(scalar_tensor);
  out_switch_x_desc->AddInputDesc(bool_tensor);
  out_switch_x_desc->AddOutputDesc(scalar_tensor);
  out_switch_x_desc->AddOutputDesc(scalar_tensor);
  auto out_switch_x_node = graph->AddNode(out_switch_x_desc);

  auto out_switch_y_desc = std::make_shared<OpDesc>("OutSwitchY", SWITCH);
  out_switch_y_desc->AddInputDesc(scalar_tensor);
  out_switch_y_desc->AddInputDesc(bool_tensor);
  out_switch_y_desc->AddOutputDesc(scalar_tensor);
  out_switch_y_desc->AddOutputDesc(scalar_tensor);
  auto out_switch_y_node = graph->AddNode(out_switch_y_desc);

  auto in_pred_desc = std::make_shared<OpDesc>("InLoopCond", LOOPCOND);
  in_pred_desc->AddInputDesc(bool_tensor);
  in_pred_desc->AddOutputDesc(bool_tensor);
  auto in_pred_node = graph->AddNode(in_pred_desc);

  auto in_merge_x_desc = std::make_shared<OpDesc>("InMergeX", MERGE);
  in_merge_x_desc->AddInputDesc(scalar_tensor);
  in_merge_x_desc->AddInputDesc(scalar_tensor);
  in_merge_x_desc->AddOutputDesc(scalar_tensor);
  in_merge_x_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(in_merge_x_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto in_merge_x_node = graph->AddNode(in_merge_x_desc);

  auto in_merge_y_desc = std::make_shared<OpDesc>("InMergeY", MERGE);
  in_merge_y_desc->AddInputDesc(scalar_tensor);
  in_merge_y_desc->AddInputDesc(scalar_tensor);
  in_merge_y_desc->AddOutputDesc(scalar_tensor);
  in_merge_y_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(in_merge_y_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto in_merge_y_node = graph->AddNode(in_merge_y_desc);

  auto in_switch_desc = std::make_shared<OpDesc>("InSwitch", SWITCH);
  in_switch_desc->AddInputDesc(scalar_tensor);
  in_switch_desc->AddInputDesc(bool_tensor);
  in_switch_desc->AddOutputDesc(scalar_tensor);
  in_switch_desc->AddOutputDesc(scalar_tensor);
  auto in_switch_node = graph->AddNode(in_switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto exit_desc = std::make_shared<OpDesc>("Exit", EXIT);
  exit_desc->AddInputDesc(scalar_tensor);
  exit_desc->AddOutputDesc(scalar_tensor);
  auto exit_node = graph->AddNode(exit_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), out_merge_x_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(out_merge_x_node->GetOutDataAnchor(0), out_pred_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(out_pred_node->GetOutDataAnchor(0), out_switch_x_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(out_merge_x_node->GetOutDataAnchor(0), out_switch_x_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), out_merge_y_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(out_pred_node->GetOutDataAnchor(0), out_switch_y_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(out_merge_y_node->GetOutDataAnchor(0), out_switch_y_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(out_switch_x_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), in_merge_x_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(out_switch_y_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), in_merge_y_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(in_merge_x_node->GetOutDataAnchor(0), in_pred_node->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(in_pred_node->GetOutDataAnchor(0), in_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(in_merge_y_node->GetOutDataAnchor(0), in_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(in_switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(in_switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), exit_node->GetInDataAnchor(0));

  if (branch_flag) {
    (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  } else {
    (void)GraphUtils::AddEdge(exit_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  }
}

void make_graph_loop_cond_out(ComputeGraphPtr &graph, NodePtr &control_trigger_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto pred_desc = std::make_shared<OpDesc>("LoopCond", LOOPCOND);
  pred_desc->AddInputDesc(bool_tensor);
  pred_desc->AddOutputDesc(bool_tensor);
  auto pred_node = graph->AddNode(pred_desc);

  auto merge_x_desc = std::make_shared<OpDesc>("MergeX", MERGE);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_x_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_x_node = graph->AddNode(merge_x_desc);

  auto merge_y_desc = std::make_shared<OpDesc>("MergeY", MERGE);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_y_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_y_node = graph->AddNode(merge_y_desc);

  auto loop_switch_desc = std::make_shared<OpDesc>("LoopSwitch", SWITCH);
  loop_switch_desc->AddInputDesc(scalar_tensor);
  loop_switch_desc->AddInputDesc(bool_tensor);
  loop_switch_desc->AddOutputDesc(scalar_tensor);
  loop_switch_desc->AddOutputDesc(scalar_tensor);
  auto loop_switch_node = graph->AddNode(loop_switch_desc);

  auto cond_switch_desc = std::make_shared<OpDesc>("CondSwitch", SWITCH);
  cond_switch_desc->AddInputDesc(scalar_tensor);
  cond_switch_desc->AddInputDesc(bool_tensor);
  cond_switch_desc->AddOutputDesc(scalar_tensor);
  cond_switch_desc->AddOutputDesc(scalar_tensor);
  auto cond_switch_node = graph->AddNode(cond_switch_desc);

  auto merge_desc = std::make_shared<OpDesc>("CondMerge", MERGE);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddInputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(scalar_tensor);
  merge_desc->AddOutputDesc(int_tensor);
  auto merge_node = graph->AddNode(merge_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), merge_x_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_x_node->GetOutDataAnchor(0), pred_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pred_node->GetOutDataAnchor(0), loop_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), merge_y_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_y_node->GetOutDataAnchor(0), loop_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), cond_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(loop_switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), cond_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(cond_switch_node->GetOutDataAnchor(SWITCH_FALSE_OUTPUT), merge_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(cond_switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), merge_node->GetInDataAnchor(1));

  (void)GraphUtils::AddEdge(merge_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_loop_cond(ComputeGraphPtr &graph, bool branch_flag, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto pred_desc = std::make_shared<OpDesc>("LoopCond", LOOPCOND);
  pred_desc->AddInputDesc(bool_tensor);
  pred_desc->AddOutputDesc(bool_tensor);
  auto pred_node = graph->AddNode(pred_desc);

  auto merge_x_desc = std::make_shared<OpDesc>("MergeX", MERGE);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_x_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_x_node = graph->AddNode(merge_x_desc);

  auto merge_y_desc = std::make_shared<OpDesc>("MergeY", MERGE);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_y_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_y_node = graph->AddNode(merge_y_desc);

  auto loop_switch_desc = std::make_shared<OpDesc>("LoopSwitch", SWITCH);
  loop_switch_desc->AddInputDesc(scalar_tensor);
  loop_switch_desc->AddInputDesc(bool_tensor);
  loop_switch_desc->AddOutputDesc(scalar_tensor);
  loop_switch_desc->AddOutputDesc(scalar_tensor);
  auto loop_switch_node = graph->AddNode(loop_switch_desc);

  auto cond_switch_desc = std::make_shared<OpDesc>("CondSwitch", SWITCH);
  cond_switch_desc->AddInputDesc(scalar_tensor);
  cond_switch_desc->AddInputDesc(bool_tensor);
  cond_switch_desc->AddOutputDesc(scalar_tensor);
  cond_switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(cond_switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), merge_x_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_x_node->GetOutDataAnchor(0), pred_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pred_node->GetOutDataAnchor(0), loop_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), merge_y_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_y_node->GetOutDataAnchor(0), loop_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(loop_switch_node->GetOutDataAnchor(branch_flag ? SWITCH_TRUE_OUTPUT : SWITCH_FALSE_OUTPUT),
                            switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_cond_loop(ComputeGraphPtr &graph, bool branch_flag, NodePtr &control_trigger_node, NodePtr &switch_node) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc int_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto cond_switch_desc = std::make_shared<OpDesc>("CondSwitch", SWITCH);
  cond_switch_desc->AddInputDesc(scalar_tensor);
  cond_switch_desc->AddInputDesc(bool_tensor);
  cond_switch_desc->AddOutputDesc(scalar_tensor);
  cond_switch_desc->AddOutputDesc(scalar_tensor);
  switch_node = graph->AddNode(cond_switch_desc);

  auto pred_desc = std::make_shared<OpDesc>("LoopCond", LOOPCOND);
  pred_desc->AddInputDesc(bool_tensor);
  pred_desc->AddOutputDesc(bool_tensor);
  auto pred_node = graph->AddNode(pred_desc);

  auto merge_x_desc = std::make_shared<OpDesc>("MergeX", MERGE);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddInputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(scalar_tensor);
  merge_x_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_x_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_x_node = graph->AddNode(merge_x_desc);

  auto merge_y_desc = std::make_shared<OpDesc>("MergeY", MERGE);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddInputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(scalar_tensor);
  merge_y_desc->AddOutputDesc(int_tensor);
  AttrUtils::SetStr(merge_y_desc, ATTR_NAME_NEXT_ITERATION, "");
  auto merge_y_node = graph->AddNode(merge_y_desc);

  auto loop_switch_desc = std::make_shared<OpDesc>("LoopSwitch", SWITCH);
  loop_switch_desc->AddInputDesc(scalar_tensor);
  loop_switch_desc->AddInputDesc(bool_tensor);
  loop_switch_desc->AddOutputDesc(scalar_tensor);
  loop_switch_desc->AddOutputDesc(scalar_tensor);
  auto loop_switch_node = graph->AddNode(loop_switch_desc);

  auto identity_desc = std::make_shared<OpDesc>("Identity", IDENTITY);
  identity_desc->AddInputDesc(scalar_tensor);
  identity_desc->AddOutputDesc(scalar_tensor);
  auto identity_node = graph->AddNode(identity_desc);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), merge_x_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_x_node->GetOutDataAnchor(0), pred_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pred_node->GetOutDataAnchor(0), loop_switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));

  (void)GraphUtils::AddEdge(switch_node->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), merge_y_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(merge_y_node->GetOutDataAnchor(0), loop_switch_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(loop_switch_node->GetOutDataAnchor(branch_flag ? SWITCH_TRUE_OUTPUT : SWITCH_FALSE_OUTPUT),
                            identity_node->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}

void make_graph_multi_branch(ComputeGraphPtr &graph) {
  GeTensorDesc bool_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  x_desc->AddOutputDesc(bool_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(bool_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto switch_desc1 = std::make_shared<OpDesc>("Switch1", SWITCH);
  switch_desc1->AddInputDesc(scalar_tensor);
  switch_desc1->AddInputDesc(bool_tensor);
  switch_desc1->AddOutputDesc(scalar_tensor);
  switch_desc1->AddOutputDesc(scalar_tensor);
  auto switch_node1 = graph->AddNode(switch_desc1);

  auto switch_desc2 = std::make_shared<OpDesc>("Switch2", SWITCH);
  switch_desc2->AddInputDesc(scalar_tensor);
  switch_desc2->AddInputDesc(bool_tensor);
  switch_desc2->AddOutputDesc(scalar_tensor);
  switch_desc2->AddOutputDesc(scalar_tensor);
  auto switch_node2 = graph->AddNode(switch_desc2);

  auto identity_desc1 = std::make_shared<OpDesc>("Identity1", IDENTITY);
  identity_desc1->AddInputDesc(scalar_tensor);
  identity_desc1->AddOutputDesc(scalar_tensor);
  auto identity_node1 = graph->AddNode(identity_desc1);

  auto identity_desc2 = std::make_shared<OpDesc>("Identity2", IDENTITY);
  identity_desc2->AddInputDesc(scalar_tensor);
  identity_desc2->AddOutputDesc(scalar_tensor);
  auto identity_node2 = graph->AddNode(identity_desc2);

  auto control_trigger_desc = std::make_shared<OpDesc>("ControlTrigger", CONTROLTRIGGER);
  auto control_trigger_node = graph->AddNode(control_trigger_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(SWITCH_PRED_INPUT));
  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(SWITCH_DATA_INPUT));

  (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(switch_node2->GetOutDataAnchor(SWITCH_TRUE_OUTPUT), identity_node2->GetInDataAnchor(0));

  (void)GraphUtils::AddEdge(identity_node1->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
  (void)GraphUtils::AddEdge(identity_node2->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor());
}
}




TEST_F(UtestControlTriggerPass, Run0) {
  Status retStatus;
  ControlTriggerPass ctrlTriggerPass;

  DEF_GRAPH(g1) {
    CHAIN(NODE("sgt/arg_0", DATA)->NODE("sgt/conv2d", CONTROLTRIGGER)->NODE("sgt/Output", NETOUTPUT));
    CHAIN(NODE("sgt/arg_1", DATA)->NODE("sgt/conv2d"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);

  retStatus = ctrlTriggerPass.Run(graph);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestControlTriggerPass, HandleDynamicCtrlEdges) {
  GeTensorDesc scalar_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ControlTriggerPass ctrlTriggerPass;
  Status retStatus;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("node");

  const OpDescPtr xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
  xOpDef->AddInputDesc(scalar_tensor_desc);
  xOpDef->AddOutputDesc(scalar_tensor_desc);
  NodePtr xNode = graph->AddNode(xOpDef);

  const OpDescPtr yOpDef = std::make_shared<OpDesc>("y", SWITCH);
  yOpDef->AddInputDesc(scalar_tensor_desc);
  yOpDef->AddOutputDesc(scalar_tensor_desc);
  NodePtr yNode = graph->AddNode(yOpDef);

  const OpDescPtr zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
  zOpDef->AddOutputDesc(scalar_tensor_desc);
  zOpDef->AddInputDesc(scalar_tensor_desc);
  NodePtr zNode = graph->AddNode(zOpDef);
  GraphUtils::AddEdge(xNode->GetOutControlAnchor(), yNode->GetInControlAnchor());

  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_nested_cond_branch_1(graph, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
    EXPECT_EQ(ctrlTriggerPass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  // bool branch_flag = false;
  retStatus = ctrlTriggerPass.InsertOppositeBranch(graph, control_trigger_node, in_ctrl_node, find_switch_node, false);
  // EXPECT_EQ(retStatus, FAILED);
  retStatus = ctrlTriggerPass.InsertOppositeBranch(graph, control_trigger_node, in_ctrl_node, find_switch_node, true);

  retStatus = ctrlTriggerPass.HandleDynamicCtrlEdges(graph, xNode, yNode);
  EXPECT_EQ(retStatus, SUCCESS);

  ctrlTriggerPass.InsertMergeNode(graph, xNode, yNode, scalar_tensor_desc, zNode);
  // EXPECT_EQ(ret, nullptr);

  ctrlTriggerPass.InsertMergeNode(graph, yNode, xNode, scalar_tensor_desc, zNode);

  auto ret = ctrlTriggerPass.InsertConstNode(graph, zNode, scalar_tensor_desc, true);
  EXPECT_EQ(ret, nullptr);

  ret = ctrlTriggerPass.InsertConstNode(graph, zNode, scalar_tensor_desc, false);
  // EXPECT_EQ(ret, nullptr);

  ret = ctrlTriggerPass.InsertIdentityNode(graph, "name", scalar_tensor_desc, find_switch_node);
  EXPECT_NE(ret, nullptr);

  ctrlTriggerPass.Run(graph);
}

/**
 *   data1  data2
 *     \   /
 *      add
 *       |
 *   netoutput
 */
ComputeGraphPtr BuildWrongGraph1() {
  auto builder = ut::GraphBuilder("g5");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UtestControlTriggerPass, FindPredInput) {
  ControlTriggerPass ctrlTriggerPass;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("node");
  OpDescPtr sub_graph_while_op_desc = std::make_shared<OpDesc>("while", WHILE);
  sub_graph_while_op_desc->AddInputDesc(GeTensorDesc());
  sub_graph_while_op_desc->AddOutputDesc(GeTensorDesc());
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  NodePtr sub_graph_while_node = sub_graph->AddNode(sub_graph_while_op_desc);
  graph->SetParentNode(sub_graph_while_node);
  Status retStatus = ctrlTriggerPass.FindPredInput(sub_graph_while_node);
  EXPECT_NE(retStatus, SUCCESS);

  NodePtr switch_node1 = nullptr;
  retStatus = ctrlTriggerPass.FindPredInput(switch_node1);
  EXPECT_NE(retStatus, SUCCESS);

  GeTensorDesc int64_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
  auto switch_n_desc = std::make_shared<OpDesc>("switch_n", SWITCHN);
  switch_n_desc->AddInputDesc(int64_tensor);
  switch_n_desc->AddInputDesc(int64_tensor);
  auto switch_n_node = graph->AddNode(switch_n_desc); 

  OpDescPtr dst_op = std::make_shared<OpDesc>("B1", "B");
  dst_op->AddInputDesc(int64_tensor);
  dst_op->AddOutputDesc(int64_tensor);
  auto dst_node1 = graph->AddNode(dst_op);

  GraphUtils::AddEdge(switch_n_node->GetOutDataAnchor(0), dst_node1->GetInDataAnchor(0));

  retStatus = ctrlTriggerPass.FindPredInput(switch_n_node);

}

TEST_F(UtestControlTriggerPass, GetInNodes) {
  ControlTriggerPass ctrlTriggerPass;
  ge::ComputeGraphPtr graph = BuildNormalGraph();
  const auto case_node = graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  // EXPECT_EQ(case_node->GetAllInDataAnchorsSize(), 2);

  std::set<std::pair<NodePtr, uint32_t>> in_nodes;
  ctrlTriggerPass.GetInNodes(case_node, in_nodes);
  // EXPECT_EQ(in_nodes.empty(), true);
}

TEST_F(UtestControlTriggerPass, TransferNodeType) {
  ControlTriggerPass ctrlTriggerPass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("node");
  OpDescPtr switch_op_desc = std::make_shared<OpDesc>("switch", SWITCH);
  NodePtr switch_op_node = graph->AddNode(switch_op_desc);

  EXPECT_NO_THROW(ctrlTriggerPass.TransferNodeType(switch_op_node, SWITCH_INPUT_NUM));
  // EXPECT_EQ(in_nodes.empty(), true);

  EXPECT_NO_THROW(ctrlTriggerPass.TransferNodeType(switch_op_node, SWITCH_TRUE_OUTPUT));
}

TEST_F(UtestControlTriggerPass, ClearStatus) {
  ControlTriggerPass ctrlTriggerPass;
  Status retStatus = ctrlTriggerPass.ClearStatus();
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestControlTriggerPass, no_control_trigger_run_success) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_no_control_trigger(graph);

  ControlTriggerPass control_trigger_pass;
  EXPECT_EQ(control_trigger_pass.Run(graph), domi::SUCCESS);
}

TEST_F(UtestControlTriggerPass, no_valid_switch_run_success) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_no_valid_switch(graph);

  ControlTriggerPass control_trigger_pass;
  EXPECT_EQ(control_trigger_pass.Run(graph), domi::SUCCESS);

  uint32_t switch_num = 0;
  uint32_t identity_num = 0;
  uint32_t const_num = 0;
  uint32_t merge_num = 0;

  std::string node_type;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    OpDescPtr tmp_desc = node->GetOpDesc();
    node_type = tmp_desc->GetType();
    if (node_type == SWITCH) {
      switch_num++;
    } else if (node_type == IDENTITY) {
      identity_num++;
    } else if (node_type == CONSTANT) {
      const_num++;
    } else if (node_type == MERGE) {
      merge_num++;
    }
  }

  EXPECT_EQ(switch_num, 1);
  EXPECT_EQ(identity_num, 1);
  EXPECT_TRUE(const_num == 0);
  EXPECT_TRUE(merge_num == 0);
}

TEST_F(UtestControlTriggerPass, enter) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_enter(graph);

  ControlTriggerPass control_trigger_pass;
  EXPECT_EQ(control_trigger_pass.Run(graph), domi::SUCCESS);
}

TEST_F(UtestControlTriggerPass, FindPredInput_fail_no_in_cond_anchor) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_no_in_cond_anchor(graph);

  ControlTriggerPass control_trigger_pass;
  EXPECT_NE(control_trigger_pass.Run(graph), domi::SUCCESS);
}

TEST_F(UtestControlTriggerPass, FindPredInput_fail_no_pred_cond_anchor) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_no_pred_cond_anchor(graph);

  ControlTriggerPass control_trigger_pass;
  EXPECT_NE(control_trigger_pass.Run(graph), domi::SUCCESS);
}

TEST_F(UtestControlTriggerPass, HandleDynamicCtrlEdges_same_pred) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  make_control_trigger_node(graph, control_trigger_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 3);
  auto in_ctrl_node_f = control_trigger_node->GetInControlNodes().at(0);
  auto in_ctrl_node_t = control_trigger_node->GetInControlNodes().at(1);
  auto in_ctrl_node_1 = control_trigger_node->GetInControlNodes().at(2);

  ControlTriggerPass control_trigger_pass;
  EXPECT_EQ(control_trigger_pass.HandleDynamicCtrlEdges(graph, control_trigger_node, in_ctrl_node_f), domi::SUCCESS);
  EXPECT_EQ(control_trigger_pass.HandleDynamicCtrlEdges(graph, control_trigger_node, in_ctrl_node_1), domi::SUCCESS);
  uint32_t switch_num = 0;
  uint32_t identity_num = 0;
  uint32_t const_num = 0;
  uint32_t merge_num = 0;

  std::string node_type;
  std::string merge_node_name;
  bool is_same_node = false;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    OpDescPtr tmp_desc = node->GetOpDesc();
    node_type = tmp_desc->GetType();
    if (node_type == SWITCH) {
      switch_num++;
    } else if (node_type == IDENTITY) {
      identity_num++;
    } else if (node_type == CONSTANT) {
      const_num++;
    } else if (node_type == MERGE) {
      if (merge_node_name.empty()) {
        merge_node_name = node->GetName();
      } else {
        if (merge_node_name == node->GetName()) {
          is_same_node = true;
        }
      }
      std::cout << node->GetName() << std::endl;
      merge_num++;
    }
  }
  EXPECT_EQ(is_same_node, false);
  EXPECT_EQ(switch_num, 2);
  EXPECT_EQ(identity_num, 6);
  EXPECT_EQ(const_num, 4);
  EXPECT_EQ(merge_num, 2);

  EXPECT_EQ(control_trigger_pass.HandleDynamicCtrlEdges(graph, control_trigger_node, in_ctrl_node_t), domi::SUCCESS);
  EXPECT_EQ(in_ctrl_node_t->GetOutControlNodes().size(), 1);
  EXPECT_EQ(in_ctrl_node_t->GetOutControlNodes().at(0)->GetType(), CONSTANT);
  EXPECT_NE(GraphUtils::RemoveEdge(in_ctrl_node_t->GetOutControlAnchor(), control_trigger_node->GetInControlAnchor()),
            GRAPH_SUCCESS);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_cond_branch_f) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_cond_branch(graph, false, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_cond_branch_t) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_cond_branch(graph, true, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_out_cond_branch_1) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  make_graph_out_cond_branch_1(graph, control_trigger_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_out_cond_branch_2) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  make_graph_out_cond_branch_2(graph, control_trigger_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_switch_ctrl_edge) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  make_graph_switch_ctrl_edge(graph, control_trigger_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_nested_cond_branch_1) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_nested_cond_branch_1(graph, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_nested_cond_branch_2) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_nested_cond_branch_2(graph, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_cascade_cond_branch) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_cascade_cond_branch(graph, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_loop_branch_t) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_loop_branch(graph, true, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_loop_branch_f) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_loop_branch(graph, false, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_nested_loop_branch_t) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_nested_loop_branch(graph, true, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_nested_loop_branch_f) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_nested_loop_branch(graph, false, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_loop_cond_out) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  make_graph_loop_cond_out(graph, control_trigger_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}


TEST_F(UtestControlTriggerPass, FindSwitchNode_loop_t_cond) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_loop_cond(graph, true, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_loop_f_cond) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_loop_cond(graph, false, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_cond_loop_f) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_cond_loop(graph, false, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == switch_node, true);
  EXPECT_EQ(branch_flag, true);
}

TEST_F(UtestControlTriggerPass, FindSwitchNode_cond_loop_t) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr control_trigger_node = nullptr;
  NodePtr switch_node = nullptr;
  make_graph_cond_loop(graph, true, control_trigger_node, switch_node);

  EXPECT_EQ(control_trigger_node->GetInControlNodes().size(), 1);
  auto in_ctrl_node = control_trigger_node->GetInControlNodes().at(0);

  ControlTriggerPass control_trigger_pass;
  NodePtr find_switch_node = nullptr;
  bool branch_flag = false;
  EXPECT_EQ(control_trigger_pass.FindSwitchNode(in_ctrl_node, find_switch_node, branch_flag), domi::SUCCESS);

  EXPECT_EQ(find_switch_node == nullptr, true);
  EXPECT_TRUE(branch_flag == false);
}

TEST_F(UtestControlTriggerPass, multi_branch_succ) {
  GetLocalOmgContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_multi_branch(graph);

  ControlTriggerPass control_trigger_pass;
  EXPECT_EQ(control_trigger_pass.Run(graph), domi::SUCCESS);

  uint32_t switch_num = 0;
  uint32_t identity_num = 0;
  uint32_t const_num = 0;
  uint32_t merge_num = 0;

  std::string node_type;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    OpDescPtr tmp_desc = node->GetOpDesc();
    node_type = tmp_desc->GetType();
    if (node_type == SWITCH) {
      switch_num++;
    } else if (node_type == IDENTITY) {
      identity_num++;
    } else if (node_type == CONSTANT) {
      const_num++;
    } else if (node_type == MERGE) {
      merge_num++;
    } else if (node_type == CONTROLTRIGGER) {
      for (auto &in_ctrl_node : node->GetInControlNodes()) {
        EXPECT_EQ(in_ctrl_node->GetType(), MERGE);
      }
    }
  }

  EXPECT_EQ(switch_num, 2);
  EXPECT_EQ(identity_num, 4);
  EXPECT_EQ(const_num, 4);
  EXPECT_EQ(merge_num, 2);
}

TEST_F(UtestControlTriggerPass, FindPredInput_fail_nullptr) {
  ControlTriggerPass control_trigger_pass;
  EXPECT_NE(control_trigger_pass.FindPredInput(nullptr), domi::SUCCESS);
}
