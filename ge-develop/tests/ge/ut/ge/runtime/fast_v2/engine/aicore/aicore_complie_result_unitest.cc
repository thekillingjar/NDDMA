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
#include "common/share_graph.h"
#include "engine/aicore/converter/aicore_compile_results.h"
#include "faker/global_data_faker.h"
#include "graph/op_kernel_bin.h"
#include "graph/debug/ge_attr_define.h"
#include "common/bg_test.h"

using namespace ge;
namespace gert {

struct AicoreCompileResultTest :  public bg::BgTestAutoCreateFrame {};

TEST_F(AicoreCompileResultTest, test_sink_atomic_bin_failed){
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto node = graph->FindNode("data1");
  ASSERT_EQ(SinkAtomicBin(node, global_data.FindCompiledResult(node)), nullptr);
}

void StubKernelBin(NodePtr & node, const std::string &kernel_key, const std::string &magic_key){
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(node->GetName(), std::move(buffer));
  node->GetOpDesc()->SetExtAttr(kernel_key, tbe_kernel_ptr);
  ge::AttrUtils::SetStr(node->GetOpDesc(), magic_key, "RT_DEV_BINARY_MAGIC_ELF_AICPU");
}

TEST_F(AicoreCompileResultTest, test_sink_bin_for_aicore_success){
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto node = graph->FindNode("data1");
  auto global_data =
      GlobalDataFaker(root_model).AddTaskDef(node->GetName(), AiCoreTaskDefFaker("stub_func")).Build();
  StubKernelBin(node, ge::OP_EXTATTR_NAME_TBE_KERNEL, ge::TVM_ATTR_NAME_MAGIC);
  ASSERT_NE(SinkBinForAicore(node, global_data.FindCompiledResult(node)), nullptr);
}

TEST_F(AicoreCompileResultTest, test_sink_aicore_bin_failed){
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto node = graph->FindNode("data1");
  ASSERT_EQ(SinkBinForAicore(node, global_data.FindCompiledResult(node)), nullptr);
}

TEST_F(AicoreCompileResultTest, test_sink_atomic_bin_success){
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto node = graph->FindNode("data1");
  auto global_data =
      GlobalDataFaker(root_model).AddTaskDef(node->GetName(), AiCoreTaskDefFaker("stub_func").AtomicStubNum("atomic_func")).Build();
  StubKernelBin(node, ge::EXT_ATTR_ATOMIC_TBE_KERNEL, ge::ATOMIC_ATTR_TVM_MAGIC);
  ASSERT_NE(SinkAtomicBin(node, global_data.FindCompiledResult(node)), nullptr);
}
}
