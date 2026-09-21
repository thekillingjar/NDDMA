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

#include "runtime/rt.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "macro_utils/dt_public_scope.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/model/node_item.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "hybrid/executor/node_state.h"
#include "single_op/single_op.h"
#include "single_op/single_op_manager.h"
#include "single_op/task/build_task_utils.h"
#include "common/dump/dump_manager.h"
#include "common/utils/executor_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace ge;
using namespace hybrid;

namespace {
struct FakeGraphItem : GraphItem {
  FakeGraphItem(NodePtr node) {
    NodeItem::Create(node, node_item);
    node_item->num_inputs = 2;
    node_item->input_start = 0;
    node_item->output_start = 0;
    node_items_.emplace_back(node_item.get());
    total_inputs_ = 2;
    total_outputs_ = 1;
  }

  NodeItem *GetNodeItem() {
    return node_item.get();
  }

 private:
  std::unique_ptr<NodeItem> node_item;
};
}  // namespace
class UtestExecutorUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST(UtestExecutorUtils, test_if_has_host_mem_input) {
  DEF_GRAPH(single_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("data1");

    auto transdata1 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("transdata1");

    CHAIN(NODE(op_ptr)->NODE(transdata1));
  };

  auto graph = ToComputeGraph(single_op);
  auto node = graph->FindNode("transdata1");
  for (size_t i = 0U; i < node->GetOpDesc()->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr &input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_desc == nullptr) {
      continue;
    }
    (void)ge::AttrUtils::SetInt(*input_desc, ge::ATTR_NAME_PLACEMENT, 2);
  }
  EXPECT_TRUE(ExecutorUtils::HasHostMemInput(node->GetOpDesc()));
}

TEST(UtestExecutorUtils, test_update_host_mem_input_args_util_for_hybrid) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc tensor_desc;
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  FrameState frame_state;
  FakeGraphItem graphItem(node);
  auto node_item = graphItem.GetNodeItem();
  SubgraphContext subgraph_context(nullptr, nullptr);
  uint8_t host_mem[8];
  TensorValue tensorValue(&host_mem, 8);
  subgraph_context.all_inputs_.emplace_back(tensorValue);
  subgraph_context.all_inputs_.emplace_back(tensorValue);
  NodeState node_state(*node_item, &subgraph_context, frame_state);
  TaskContext context(nullptr, &node_state, &subgraph_context);

  size_t io_addrs_size = 3 * sizeof(void *) + kMaxHostMemInputLen;
  uint64_t io_addrs[io_addrs_size];
  size_t host_mem_input_data_offset_in_args = 3 * sizeof(void *);

  vector<rtHostInputInfo_t> host_inputs;
  auto ret = ExecutorUtils::UpdateHostMemInputArgs(context, io_addrs, io_addrs_size,
                                                   host_mem_input_data_offset_in_args, host_inputs, false);
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestExecutorUtils, test_assemble_reuse_binary_args) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc tensor_desc;
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  optiling::utils::OpRunInfo run_info;
  rtArgsEx_t args_ex;
  auto ret = ExecutorUtils::AssembleReuseBinaryArgs(op_desc, run_info, args_ex);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(args_ex.argsSize, 16408);
  AttrUtils::SetInt(op_desc, ATTR_NAME_MAX_TILING_SIZE, 2);
  args_ex.argsSize = 32;
  ret = ExecutorUtils::AssembleReuseBinaryArgs(op_desc, run_info, args_ex);
  EXPECT_EQ(ret, SUCCESS);
}
