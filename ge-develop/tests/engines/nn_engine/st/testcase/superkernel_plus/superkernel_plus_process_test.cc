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
#include "fe_llt_utils.h"
#include "common/fe_op_info_common.h"
#include "common/scope_allocator.h"
#include "ops_kernel_builder/aicore_ops_kernel_builder.h"
#include "itf_handler/itf_handler.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected
#include "itf_handler/itf_handler.h"

using namespace ge;
namespace fe {
class SuperkernelPlusProcessTest: public testing::Test {
 protected:
  static void SetUpTestCase() {
    auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
    cout << "SuperkernelPlusProcessTest SetUp" << endl;
    InitWithSocVersion("Ascend910B1", "force_fp16");
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    map<string, string> options;
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
    Configuration::Instance(AI_CORE_NAME).content_map_["superkernel_plus.enable"] = "true";
  }
  static void TearDownTestCase() {
    cout << "SuperkernelPlusProcessTest TearDown" << endl;
    Finalize();
  }

  /**
   *    Const \
   * Data -> Conv2D -> Relu -> Add -> SoftmaxV2 -> StridedSliceD
   *          Data -> Sigmoid -/
   *
   */
  ge::ComputeGraphPtr CreateGraphWithType() {
    vector<int64_t> dims = {3,4,5,6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "PlaceHolder");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    OpDescPtr softmax_op = std::make_shared<OpDesc>("softmax", "SoftmaxV2");
    OpDescPtr add_op = std::make_shared<OpDesc>("add", "Add");
    OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
    OpDescPtr slice_op = std::make_shared<OpDesc>("strided_sliced", "StridedSliceD");

    data1_op->AddOutputDesc(tensor_desc);
    data2_op->AddOutputDesc(tensor_desc);
    const_op->AddOutputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddOutputDesc(tensor_desc);
    relu_op->AddInputDesc(tensor_desc);
    relu_op->AddOutputDesc(tensor_desc);
    sigmoid_op->AddInputDesc(tensor_desc);
    sigmoid_op->AddOutputDesc(tensor_desc);
    add_op->AddInputDesc(tensor_desc);
    add_op->AddInputDesc(tensor_desc);
    add_op->AddOutputDesc(tensor_desc);
    softmax_op->AddInputDesc(tensor_desc);
    softmax_op->AddOutputDesc(tensor_desc);
    slice_op->AddInputDesc(tensor_desc);
    slice_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(sigmoid_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(add_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(softmax_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(slice_op, "_fe_imply_type", 6);

    AttrUtils::SetListInt(conv_op, "strides", {1,1,1,1});
    AttrUtils::SetListInt(conv_op, "pads", {1,1,1,1});
    AttrUtils::SetListInt(conv_op, "dilations", {1,1,1,1});

    AttrUtils::SetListInt(slice_op, "begin", {1,1,1,1});
    AttrUtils::SetListInt(slice_op, "end", {1,1,1,1});
    AttrUtils::SetListInt(slice_op, "strides", {1,1,1,1});
    AttrUtils::SetInt(slice_op, "begin_mask", 1);
    AttrUtils::SetInt(slice_op, "end_mask", 1);
    AttrUtils::SetInt(slice_op, "ellipsis_mask", 1);
    AttrUtils::SetInt(slice_op, "new_axis_mask", 1);
    AttrUtils::SetInt(slice_op, "shrink_axis_mask", 1);

    AttrUtils::SetBool(conv_op, ATTR_NAME_IS_FIRST_NODE, true);
    AttrUtils::SetBool(slice_op, ATTR_NAME_IS_LAST_NODE, true);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr const_node = graph->AddNode(const_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
    NodePtr add_node = graph->AddNode(add_op);
    NodePtr softmax_node = graph->AddNode(softmax_op);
    NodePtr slice_node = graph->AddNode(slice_op);
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0), slice_node->GetInDataAnchor(0));

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");

    return graph;
  }
};
/*
TEST_F(SuperkernelPlusProcessTest, superkernel_plus_case1) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType();
  Status ret = graph_optimizer->OptimizeOriginalGraphJudgeInsert(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  int64_t scope_id = -1;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (!IsTbeOp(node->GetOpDesc())) {
      continue;
    }
    int64_t tmp_scope_id = -1;
    bool has_scope = ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), tmp_scope_id);
    EXPECT_EQ(has_scope, true);
    if (scope_id >= 0) {
      EXPECT_EQ(tmp_scope_id, scope_id);
    } else {
      scope_id = tmp_scope_id;
    }
  }

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  context.stream = reinterpret_cast<void *>(&base);
  std::vector<domi::TaskDef> tasks;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 3);
}
*/
TEST_F(SuperkernelPlusProcessTest, superkernel_plus_case2) {
  Configuration::Instance(AI_CORE_NAME).content_map_["superkernel_plus.enable"] = "false";
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType();
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);

  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (!IsTbeOp(node->GetOpDesc()) || IsNoTaskOp(node)) {
      continue;
    }
    FillNodeParaType(node);
    int64_t tmp_scope_id = -1;
    bool has_scope = ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), tmp_scope_id);
    EXPECT_EQ(has_scope, false);
  }

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      // EXPECT_EQ(ret, SUCCESS);
    }
  }
  // EXPECT_EQ(tasks.size(), 8);
  Configuration::Instance(AI_CORE_NAME).content_map_["superkernel_plus.enable"] = "true";
}

TEST_F(SuperkernelPlusProcessTest, superkernel_plus_case3) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType();
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  int64_t scope_id = -1;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (!IsTbeOp(node->GetOpDesc())) {
      continue;
    }
    FillNodeParaType(node);
    int64_t tmp_scope_id = -1;
    bool has_scope = ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), tmp_scope_id);
    // EXPECT_EQ(has_scope, true);
    if (scope_id >= 0) {
      EXPECT_EQ(tmp_scope_id, scope_id);
      bool no_task = false;
      AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, true);
    } else {
      scope_id = tmp_scope_id;
    }
    if (node->GetType() == "Add") {
      node->GetOpDesc()->DelAttr(ge::ATTR_NAME_NOTASK);
      node->GetOpDesc()->DelAttr("_skp_fusion_scope");
    }
  }

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      // EXPECT_EQ(ret, SUCCESS);
    }
  }
  // EXPECT_EQ(tasks.size(), 5);
}

TEST_F(SuperkernelPlusProcessTest, superkernel_plus_case4) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType();
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  int64_t scope_id = -1;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (!IsTbeOp(node->GetOpDesc())) {
      continue;
    }
    FillNodeParaType(node);
    int64_t tmp_scope_id = -1;
    bool has_scope = ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), tmp_scope_id);
    // EXPECT_EQ(has_scope, true);
    if (scope_id >= 0) {
      EXPECT_EQ(tmp_scope_id, scope_id);
    } else {
      scope_id = tmp_scope_id;
    }
    if (node->GetType() == "Add") {
      ScopeAllocator::SetSkpScopeAttr(node->GetOpDesc(), ScopeAllocator::Instance().AllocateSkpScopeId());
      node->GetOpDesc()->DelAttr(ge::ATTR_NAME_NOTASK);
    }
  }

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      // EXPECT_EQ(ret, SUCCESS);
    }
  }
  // EXPECT_EQ(tasks.size(), 5);
}

TEST_F(SuperkernelPlusProcessTest, superkernel_plus_case5) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType();
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  int64_t scope_id = -1;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (!IsTbeOp(node->GetOpDesc())) {
      continue;
    }
    FillNodeParaType(node);
    int64_t tmp_scope_id = -1;
    bool has_scope = ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), tmp_scope_id);
    // EXPECT_EQ(has_scope, true);
    if (scope_id >= 0) {
      EXPECT_EQ(tmp_scope_id, scope_id);
    } else {
      scope_id = tmp_scope_id;
    }
    if (node->GetType() == "Add") {
      OpDescPtr memset_op = std::make_shared<OpDesc>("memset", "MemSet");
      memset_op->SetWorkspaceBytes({1,2,3,4});
      memset_op->SetWorkspace({1,2,3,4});
      AttrUtils::SetInt(memset_op, "_fe_imply_type", 6);
      AttrUtils::SetStr(memset_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
      ComputeGraphPtr graph_ptr2 = std::make_shared<ComputeGraph>("test2");
      NodePtr memset_node = graph_ptr2->AddNode(memset_op);
      node->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
    }
  }

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      // EXPECT_EQ(ret, SUCCESS);
    }
  }
  // EXPECT_EQ(tasks.size(), 6);
}
}
