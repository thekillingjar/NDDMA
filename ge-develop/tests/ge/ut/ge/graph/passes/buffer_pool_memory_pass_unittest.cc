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
#include "graph/utils/tensor_utils.h"
#include "graph/passes/pass_manager.h"
#include "graph_builder_utils.h"
#include "../utils/buffer_pool_graph_builder.h"
#include "graph/passes/feature/buffer_pool_memory_pass.h"
#include "framework/memory/memory_assigner.h"


namespace ge {
class UtestBufferPoolMemoryPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  Status CheckAttr(const ComputeGraphPtr &compute_graph) {
    for (const auto &node : compute_graph->GetAllNodes()) {
      const auto &op_desc = node->GetOpDesc();
      if (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE)) {
        std::vector<int64_t> memory_size_and_offset;
        AttrUtils::GetListInt(op_desc, ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET, memory_size_and_offset);
        if (memory_size_and_offset.size() != 2U) {
          GELOGE(FAILED, "ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET not set, node = %s", op_desc->GetName().c_str());
          return FAILED;
        }
      }
    }
    return SUCCESS;
  }
};

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_normal_success_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;3");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add2;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;0");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;1");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 7680);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_normal_graph_with_multi_buffer_pool_success_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraphWithMultiBufferPool");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraphWithMultiBufferPool();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;2");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;4");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_contain_one_node_success_test) {
  ut::BufferPoolGraphBuilder builder("SerialGraph");
  ge::ComputeGraphPtr graph = builder.BuildSerialGraph();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add1;2");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add2;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;0");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;1");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;1");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add4;2");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add4");
  }
}

TEST_F(UtestBufferPoolMemoryPass, calc_node_with_multi_buffer_pool_input_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiPrefetch();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 0);
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 0);
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add1;2");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add2;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add2");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_in_different_subgraph_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithSubgraph");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithSubgraph();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;3");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 0);
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;4");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 1);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch4");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_in_different_subgraph_with_inner_dependency_success_test) {
  ut::BufferPoolGraphBuilder builder("SubgraphWithInnerDependency");
  ge::ComputeGraphPtr graph = builder.BuildSubgraphWithInnerDependency();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;3");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 1);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;4");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_with_batch_label_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiBatch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiBatch();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add1;4");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add2;5");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add3;6");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add4;7");
    EXPECT_EQ(event_info.at(1), "RecvFrom;batch_label_256/add2;4");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "batch_label_256/add2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add5;4");
    EXPECT_EQ(event_info.at(1), "RecvFrom;batch_label_256/add3;5");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "batch_label_256/add3");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_node_has_multi_output_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiOutputPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiOutputPrefetch();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch1_memcpy_async;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch2_memcpy_async;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch3_memcpy_async;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch4_memcpy_async;3");
    EXPECT_EQ(event_info.at(1), "RecvFrom;prefetch2_memcpy_async;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch2_memcpy_async");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;0");
    EXPECT_EQ(event_info.at(1), "RecvFrom;prefetch3_memcpy_async;1");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch3_memcpy_async");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_has_different_size_fail_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  const int64_t dummy_size = 256;
  auto prefetch = graph->FindNode("prefetch3");
  EXPECT_NE(prefetch, nullptr);
  (void) AttrUtils::SetInt(prefetch->GetOpDesc(), "_buffer_pool_size", dummy_size);

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_size_is_not_enough_fail_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;
  auto prefetch = graph->FindNode("prefetch3");
  EXPECT_NE(prefetch, nullptr);
  builder.SetPrefetchNodeInfo(prefetch, buffer_pool_id, buffer_pool_size, {buffer_pool_size + 512});

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_size_is_not_enough_for_multi_fail_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiPrefetch();
  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;
  auto prefetch = graph->FindNode("prefetch3");
  EXPECT_NE(prefetch, nullptr);
  builder.SetPrefetchNodeInfo(prefetch, buffer_pool_id, buffer_pool_size, {buffer_pool_size});

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_node_has_multi_input_output_fail_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiInputOutputPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiInputOutputPrefetch();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

///
/// Normal graph
///
///             w1         w2         w3
///              \          \          \.
///          prefetch1  prefetch3  prefetch5
///               \          \          \.
///            prefetch2  prefetch4  prefetch6
///                \          \          \.
/// const1  ----- add1 ----- add2 ----- add3 ----- net_output
///
///
///  Memory distribution:
///
///      |___p1__|__p2__|_________|
///
///      |_____p3_____|_____p4____|
///
///      |_____p5_____|_____p6____|
TEST_F(UtestBufferPoolMemoryPass, buffer_pool_normal_success_test_multi_prefetch) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildMultiPrefetchGraph();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CheckAttr(graph), SUCCESS);

  {
    std::vector<std::string> evt_info_list;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", evt_info_list);
    EXPECT_EQ(evt_info_list.size(), 1);
    EXPECT_EQ(evt_info_list.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> evt_info_list;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", evt_info_list);
    EXPECT_EQ(evt_info_list.size(), 1);
    EXPECT_EQ(evt_info_list.at(0), "RecvFrom;add1;1");
  }

  {
    std::vector<std::string> evt_info_list;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", evt_info_list);
    EXPECT_EQ(evt_info_list.size(), 1);
    EXPECT_EQ(evt_info_list.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> evt_info_list;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", evt_info_list);
    EXPECT_EQ(evt_info_list.size(), 1);
    EXPECT_EQ(evt_info_list.at(0), "RecvFrom;add2;0");
  }

  {
    std::vector<std::string> evt_info_list;
    auto prefetch = graph->FindNode("prefetch6");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", evt_info_list);
    EXPECT_EQ(evt_info_list.size(), 1);
    EXPECT_EQ(evt_info_list.at(0), "SendTo;add3;0");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_ref_io_success_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  auto prefetch2_node = graph->FindNode("prefetch2");
  auto add1_node = graph->FindNode("add1");
  auto add2_node = graph->FindNode("add2");

  auto split_op_desc = std::make_shared<OpDesc>(SPLIT, SPLIT);
  split_op_desc->AddInputDesc(GeTensorDesc());
  split_op_desc->AddOutputDesc(GeTensorDesc());
  split_op_desc->AddOutputDesc(GeTensorDesc());
  auto concat_op_desc = std::make_shared<OpDesc>(CONCAT, CONCAT);
  concat_op_desc->AddInputDesc(GeTensorDesc());
  concat_op_desc->AddInputDesc(GeTensorDesc());
  concat_op_desc->AddOutputDesc(GeTensorDesc());
  auto split_node = graph->AddNode(split_op_desc);
  auto concat_node = graph->AddNode(concat_op_desc);
  add1_node->GetOutControlAnchor()->LinkTo(split_node->GetInControlAnchor());
  prefetch2_node->GetOutDataAnchor(0)->UnlinkAll();
  prefetch2_node->GetOutDataAnchor(0)->LinkTo(split_node->GetInDataAnchor(0));
  split_node->GetOutDataAnchor(0)->LinkTo(concat_node->GetInDataAnchor(0));
  split_node->GetOutDataAnchor(1)->LinkTo(concat_node->GetInDataAnchor(1));
  concat_node->GetOutDataAnchor(0)->LinkTo(add2_node->GetInDataAnchor(1));

  BufferPoolMemoryPass buffer_pool_mem_pass;
  EXPECT_EQ(buffer_pool_mem_pass.Run(graph), SUCCESS);

  auto prefetch4_node = graph->FindNode("prefetch4");
  EXPECT_TRUE(concat_node->GetOutControlAnchor()->IsLinkedWith(prefetch4_node->GetInControlAnchor()));
  EXPECT_FALSE(split_node->GetOutControlAnchor()->IsLinkedWith(prefetch4_node->GetInControlAnchor()));
}
}  // namespace ge
