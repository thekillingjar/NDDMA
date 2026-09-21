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

#include <cstdint>
#include <iostream>
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "ge/ge_api.h"
#include "ge_attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/control_flow_and_stream/mark_agnostic_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/passes/pass_manager.h"
#include "api/gelib/gelib.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/passes/feature/iterator_op_pass.h"
#include "graph/passes/control_flow_and_stream/flow_ctrl_pass.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "macro_utils/dt_public_unscope.h"


using namespace domi;

namespace ge {
namespace {
class IteratorOpPassTest : public testing::Test {
 protected:
  void SetUp() {
    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    EXPECT_EQ(SUCCESS, ge::VarManager::Instance(0)->Init(session_version, session_id, device_id, job_id));
  }
  void TearDown() { VarManagerPool::Instance().Destory(); }

  void MakeGraph(ge::ComputeGraphPtr &graph) {
    auto desc_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    ge::OpDescPtr op_desc_get_next = std::make_shared<ge::OpDesc>("IteratorGetNext", FRAMEWORKOP);
    ge::AttrUtils::SetStr(op_desc_get_next, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, FRAMEWORKOP);
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

  void MakeGraph2(ge::ComputeGraphPtr &graph) {
    auto desc_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    ge::OpDescPtr op_desc_get_next = std::make_shared<ge::OpDesc>("IteratorGetNext", FRAMEWORKOP);
    ge::AttrUtils::SetStr(op_desc_get_next, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, FRAMEWORKOP);
    op_desc_get_next->AddOutputDesc(desc);

    ge::OpDescPtr op_desc_memcpy = std::make_shared<ge::OpDesc>("MemcpyAsync", MEMCPYASYNC);
    op_desc_memcpy->AddInputDesc(desc);
    op_desc_memcpy->AddOutputDesc(desc);
    ge::AttrUtils::SetBool(op_desc_memcpy, ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, true);

    ge::OpDescPtr op_desc_a = std::make_shared<ge::OpDesc>("A", RESOURCEAPPLYMOMENTUM);
    op_desc_a->AddInputDesc(desc);
    op_desc_a->AddOutputDesc(desc);

    ge::OpDescPtr op_desc_gatherv2 = std::make_shared<ge::OpDesc>("IteratorV2", ITERATORV2);
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
}  // namespace

TEST_F(IteratorOpPassTest, iterator_op_pass_run_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("IteratorOpPassSuccess");
  graph->SetNeedIteration(true);

  MakeGraph(graph);
  graph->TopologicalSorting();

  AddSessionVariables();
  IteratorOpPass iterator_op_pass;

  Status ret = iterator_op_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(IteratorOpPassTest, iterator_op_pass_run_success2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("IteratorOpPassSuccess");
  graph->SetNeedIteration(true);

  MakeGraph2(graph);
  graph->TopologicalSorting();

  AddSessionVariables();
  IteratorOpPass iterator_op_pass;

  Status ret = iterator_op_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = iterator_op_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

} // namespace ge
