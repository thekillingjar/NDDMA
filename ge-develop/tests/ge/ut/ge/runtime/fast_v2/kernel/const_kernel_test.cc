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
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "compute_graph.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph/utils/execute_graph_utils.h"


namespace gert {
namespace kernel {
ge::graphStatus DefaultFunc(KernelContext *context);
struct ConstKernelTest : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

ge::ExecuteGraphPtr ConstGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  return ge::ToExecuteGraph(g1);
}

TEST_F(ConstKernelTest, test_const_for_uint64_success) {
  auto graph = ConstGraph();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "const1");
  auto run_context = BuildKernelRunContext(0, 1);

  uint64_t data = 100;
  node->GetOpDescBarePtr()->SetAttr("is_string", ge::AnyValue::CreateFrom(false));
  ge::AttrUtils::SetZeroCopyBytes(node->GetOpDescBarePtr(), kConstValue,
                                  ge::Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(&data), sizeof(data)));

  auto kern_context = run_context.GetContext<KernelContext>();
  auto constFuncs = registry.FindKernelFuncs("Const");
  ASSERT_EQ(constFuncs->outputs_creator(node, kern_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(*kern_context->GetOutputPointer<uint64_t>(0), 100);
  ASSERT_EQ(kernel::DefaultFunc(kern_context), ge::GRAPH_SUCCESS);
}

struct TestData {
  uint64_t a;
  uint64_t b;
};

TEST_F(ConstKernelTest, test_const_for_struct_size_then_64_success) {
  auto graph = ConstGraph();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "const1");
  auto run_context = BuildKernelRunContext(0, 1);

  TestData data{10, 10};

  node->GetOpDescBarePtr()->SetAttr("is_string", ge::AnyValue::CreateFrom(false));
  ge::AttrUtils::SetZeroCopyBytes(node->GetOpDescBarePtr(), kConstValue,
                                  ge::Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(&data), sizeof(data)));

  auto kern_context = run_context.GetContext<KernelContext>();
  auto constFuncs = registry.FindKernelFuncs("Const");
  ASSERT_EQ(constFuncs->outputs_creator(node, kern_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kern_context->GetOutputPointer<TestData>(0)->a, 10);
  ASSERT_EQ(constFuncs->run_func(kern_context), ge::GRAPH_SUCCESS);
  run_context.FreeValue(0);
}

TEST_F(ConstKernelTest, test_const_for_small_string_success) {
  auto graph = ConstGraph();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "const1");
  auto run_context = BuildKernelRunContext(0, 1);

  std::string data = "hello";

  node->GetOpDescBarePtr()->SetAttr("is_string", ge::AnyValue::CreateFrom(true));
  ge::AttrUtils::SetZeroCopyBytes(
      node->GetOpDescBarePtr(), kConstValue,
      ge::Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(data.c_str()), data.size() + 1));

  auto kern_context = run_context.GetContext<KernelContext>();
  auto constFuncs = registry.FindKernelFuncs("Const");
  ASSERT_EQ(constFuncs->outputs_creator(node, kern_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(strlen(*kern_context->GetOutputPointer<char *>(0)), 5);
  ASSERT_EQ(constFuncs->run_func(kern_context), ge::GRAPH_SUCCESS);
  run_context.FreeValue(0);
}

TEST_F(ConstKernelTest, test_const_for_large_string_success) {
  auto graph = ConstGraph();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "const1");
  auto run_context = BuildKernelRunContext(0, 1);

  std::string data = "hello world";

  node->GetOpDescBarePtr()->SetAttr("is_string", ge::AnyValue::CreateFrom(true));
  ge::AttrUtils::SetZeroCopyBytes(
      node->GetOpDescBarePtr(), kConstValue,
      ge::Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(data.c_str()), data.size() + 1));

  auto kern_context = run_context.GetContext<KernelContext>();
  auto constFuncs = registry.FindKernelFuncs("Const");
  ASSERT_EQ(constFuncs->outputs_creator(node, kern_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(strlen(*kern_context->GetOutputPointer<char *>(0)), 11);
  ASSERT_EQ(constFuncs->run_func(kern_context), ge::GRAPH_SUCCESS);
  run_context.FreeValue(0);
}

TEST_F(ConstKernelTest, test_const_create_for_output_is_empty) {
  auto graph = ConstGraph();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "const1");
  auto run_context = BuildKernelRunContext(0, 0);

  std::string data = "hello world";

  node->GetOpDescBarePtr()->SetAttr("is_string", ge::AnyValue::CreateFrom(true));
  ge::AttrUtils::SetZeroCopyBytes(
      node->GetOpDescBarePtr(), kConstValue,
      ge::Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(data.c_str()), data.size() + 1));

  auto kern_context = run_context.GetContext<KernelContext>();
  auto constFuncs = registry.FindKernelFuncs("Const");
  ASSERT_EQ(constFuncs->outputs_creator(node, kern_context), ge::GRAPH_PARAM_INVALID);
}

TEST_F(ConstKernelTest, test_const_create_for_node_const_value_empty) {
  auto graph = ConstGraph();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "const1");
  auto run_context = BuildKernelRunContext(0, 0);
  auto constFuncs = registry.FindKernelFuncs("Const");
  ASSERT_EQ(constFuncs->outputs_creator(node, run_context), ge::GRAPH_FAILED);
}
}
}