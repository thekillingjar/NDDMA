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
#include <nlohmann/json.hpp>
#include <iostream>
#include <list>

#define private public
#define protected public

#include "ops_kernel_builder/task_builder/task_builder.h"
#include "ops_kernel_builder/task_builder/cmo_task_builder.h"
#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_task_base.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/ge_attr_value.h"
#include "proto/task.pb.h"

using namespace fe;
using namespace ge;
class CMOTaskBuilderSTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_prefetch) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim = {16, 16, 16, 16};
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 262176);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoPrefetch, {{mul1_node, CmoTypeObject::INPUT, 1}}}};
  add1->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*add1_node, task_defs, context, true), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_invalid) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(add1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr mul1_cmo_ext_attr = {{kCmoInvalid, {{add1_node, CmoTypeObject::INPUT, 1}}}};
  mul1->SetExtAttr("cmo_", mul1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul1_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_invalid2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(add1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr mul1_cmo_ext_attr = {{kCmoInvalid, {{add1_node, CmoTypeObject::INPUT, 1}}}};
  mul1->SetExtAttr("cmo_", mul1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  (void)ge::AttrUtils::SetInt(add1->MutableInputDesc(1), "_complex_cmo_id", 10000);
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul1_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_invalid_output) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(add1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr mul1_cmo_ext_attr = {{kCmoInvalid, {{add1_node, CmoTypeObject::OUTPUT, 0}}}};
  mul1->SetExtAttr("cmo_", mul1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul1_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_invalid_output2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(add1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr mul1_cmo_ext_attr = {{kCmoInvalid, {{add1_node, CmoTypeObject::OUTPUT, 0}}}};
  mul1->SetExtAttr("cmo_", mul1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  (void)ge::AttrUtils::SetInt(add1->MutableOutputDesc(0), "_complex_cmo_id", 10000);
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul1_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_invalid_workspace) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(add1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr mul1_cmo_ext_attr = {{kCmoInvalid, {{add1_node, CmoTypeObject::WORKSPACE, 0}}}};
  mul1->SetExtAttr("cmo_", mul1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul1_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_invalid_workspace2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(add1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(add1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr mul1_cmo_ext_attr = {{kCmoInvalid, {{add1_node, CmoTypeObject::WORKSPACE, 0}}}};
  mul1->SetExtAttr("cmo_", mul1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  (void)ge::AttrUtils::SetInt(add1, "_worksapce_0_complex_cmo_id", 10000);
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul1_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_barrier) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoBarrier, {{mul1_node, CmoTypeObject::INPUT, 1}}}};
  mul2->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul2_node, task_defs, context, true), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_barrier_output) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoBarrier, {{mul1_node, CmoTypeObject::OUTPUT, 0}}}};
  mul2->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul2_node, task_defs, context, true), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_barrier_workspace) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);

  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoBarrier, {{mul1_node, CmoTypeObject::WORKSPACE, 0}}}};
  mul2->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048 * 1000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul2_node, task_defs, context, true), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_writeback) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoWriteback, {{mul1_node, CmoTypeObject::INPUT, 1}}}};
  mul2->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 20480000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul2_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_writeback_output) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoWriteback, {{mul1_node, CmoTypeObject::OUTPUT, 0}}}};
  mul2->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 20480000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul2_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_writeback_workspace) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoWriteback, {{mul1_node, CmoTypeObject::WORKSPACE, 0}}}};
  mul2->SetExtAttr("cmo_", add1_cmo_ext_attr);
  mul1->SetInputOffset({256, 512});
  mul1->SetOutputOffset({1024});
  mul1->SetWorkspaceBytes({256});
  mul1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 20480000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*mul2_node, task_defs, context, false), SUCCESS);
}

TEST_F(CMOTaskBuilderSTest, cmo_task_builder_no_writeback) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data1", "Data");
  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr add1 = std::make_shared<ge::OpDesc>("add1", "Add");
  ge::OpDescPtr const2 = std::make_shared<ge::OpDesc>("const2", "Const");
  ge::OpDescPtr mul1 = std::make_shared<ge::OpDesc>("mul1", "Mul");
  ge::OpDescPtr const3 = std::make_shared<ge::OpDesc>("const3", "Const");
  ge::OpDescPtr add2 = std::make_shared<ge::OpDesc>("add2", "Add");
  ge::OpDescPtr const4 = std::make_shared<ge::OpDesc>("const4", "Const");
  ge::OpDescPtr mul2 = std::make_shared<ge::OpDesc>("mul2", "Mul");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");
  std::vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  ge::TensorUtils::SetSize(out_desc, 1056);
  data->AddOutputDesc(out_desc);
  const1->AddOutputDesc(out_desc);
  const2->AddOutputDesc(out_desc);
  const3->AddOutputDesc(out_desc);
  const4->AddOutputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddInputDesc(out_desc);
  add1->AddOutputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddInputDesc(out_desc);
  mul1->AddOutputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddInputDesc(out_desc);
  add2->AddOutputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddInputDesc(out_desc);
  mul2->AddOutputDesc(out_desc);
  netoutput->AddInputDesc(out_desc);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr const1_node = graph->AddNode(const1);
  ge::NodePtr const2_node = graph->AddNode(const2);
  ge::NodePtr const3_node = graph->AddNode(const3);
  ge::NodePtr const4_node = graph->AddNode(const4);
  ge::NodePtr add1_node = graph->AddNode(add1);
  ge::NodePtr mul1_node = graph->AddNode(mul1);
  ge::NodePtr add2_node = graph->AddNode(add2);
  ge::NodePtr mul2_node = graph->AddNode(mul2);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  (void)ge::AttrUtils::SetStr(mul1, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(0), ge::ANCHOR_DATA);
  ge::AnchorUtils::SetStatus(mul1_node->GetInDataAnchor(1), ge::ANCHOR_DATA);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  (void)ge::GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  CmoExtraAttr add1_cmo_ext_attr = {{kCmoWriteback, {{mul1_node, CmoTypeObject::INPUT, 1}}}};
  add1->SetExtAttr("cmo_", add1_cmo_ext_attr);
  add1->SetInputOffset({256, 512});
  add1->SetOutputOffset({1024});
  add1->SetWorkspaceBytes({256});
  add1->SetWorkspace({2048});
  std::vector<domi::TaskDef> task_defs;
  TaskBuilderContext context;
  context.dataMemSize = 2048000;
  CMOTaskBuilderPtr cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>();

  EXPECT_EQ(cmo_task_builder_ptr->GenerateCMOTask(*add1_node, task_defs, context, true), SUCCESS);
}
