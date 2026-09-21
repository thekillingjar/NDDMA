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
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "hybrid/node_executor/node_executor.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/attr_value.h"
#include "api/gelib/gelib.h"
#include "ge/ge_api_types.h"
#include "common/sgt_slice_type.h"
#include "omg/ge_init.h"
#include "stub/gert_runtime_stub.h"
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "compiler/graph/build/memory/checker/special_node_checker.h"
#include "ge/st/stubs/utils/synchronizer.h"
#include "graph/manager/mem_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace ge;
using namespace gert;
namespace {

static void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    ge::OpKernelBinPtr tbeKernelPtr = std::make_shared<ge::OpKernelBin>("test_tvm", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
}

}
class MemoryAssignTest : public testing::Test {
  void SetUp() {
    const std::vector<rtMemType_t> mem_type{RT_MEMORY_HBM, RT_MEMORY_P2P_DDR};
    (void) ge::MemManager::Instance().Initialize(mem_type);
    MockGenerateTask();
  }
  void TearDown() {
    hybrid::NpuMemoryAllocator::Finalize();
    ge::MemManager::Instance().Finalize();
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
  }
};

/**
 * 用例描述：连续输入内存异常场景覆盖
 *
 * 预置条件：
 * 1.构造计算图
 * 2.调用内存分配，修改offset注入错误
 *
 *    data  data
 *     \    /
 *      hcom
 *       |
 *     netoutput
 *
 * 测试步骤
 * 1.构造单个计算图1
 * 2.调用内存分配，修改offset注入错误
 * 3.调用内存检查
 * 预期结果
 * 1. 检查报错
 */
TEST_F(MemoryAssignTest, ContinuousInputCheck_Failed) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildHcomGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_NE(SpecialNodeChecker::Check(compute_graph), SUCCESS);
}

/**
 * 用例描述：连续输入内存异常场景覆盖
 *
 * 预置条件：
 * 1.构造计算图
 * 2.调用内存分配，修改offset注入错误
 *
 *        data
 *         \
 *         hcom
 *         | \
 *         a  b
 *
 * 测试步骤
 * 1.构造单个计算图1
 * 2.调用内存分配，修改offset注入错误
 * 3.调用内存检查
 * 预期结果
 * 1. 检查报错
 */
TEST_F(MemoryAssignTest, ContinuousOutputCheck_Failed) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildHcomGraphWithTwoOutputs();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto hcom = compute_graph->FindNode("hcom_1");
  hcom->GetOpDescBarePtr()->SetOutputOffset({0, 2024});
  EXPECT_NE(SpecialNodeChecker::Check(compute_graph), SUCCESS);
}

/**
 * 用例描述：NoPadding连续输入内存异常场景覆盖
 *
 * 预置条件：
 * 1.构造计算图
 * 2.调用内存分配，修改offset注入错误
 *
            (0,0)
  ┌────────────────────┐
  │                    ∨
┌─────────┐  (0,1)   ┌───────┐  (0,0)   ┌─────┐  (0,0)   ┌───────────┐     ┌────────┐
│ const_1 │ ───────> │ add_1 │ ───────> │ pc1 │ ───────> │ NetOutput │ <·· │ data_1 │
└─────────┘          └───────┘          └─────┘          └───────────┘     └────────┘
            (0,1)                         ∧
  ┌────────────────────┐                  │
  │                    ∨                  │
┌─────────┐  (0,0)   ┌───────┐  (0,1)     │
│ const_2 │ ───────> │ add_2 │ ───────────┘
└─────────┘          └───────┘

 *
 * 测试步骤
 * 1.构造单个计算图1
 * 2.调用内存分配，修改offset注入错误
 * 3.调用内存检查
 * 预期结果
 * 1. 检查报错
 */
TEST_F(MemoryAssignTest, NoPaddingContinuousInputCheck_Failed) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::NetoutputNotSupportZeroCopy();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto pc1 = compute_graph->FindNode("pc1");
  pc1->GetInDataNodes().at(1)->GetOpDescBarePtr()->SetOutputOffset({1024});
  EXPECT_NE(SpecialNodeChecker::Check(compute_graph), SUCCESS);
}

/**
 * 用例描述：NoPadding连续输出内存异常场景覆盖
 *
 * 预置条件：
 * 1.构造计算图
 * 2.调用内存分配，修改offset注入错误
 *
 *   Data    Data   var
 *     \      /      |
 *      add1       split
 *         \        / \
 *          \      add2
 *            \      |
 *             \    |
 *              NetOutput
 *
 * 测试步骤
 * 1.构造单个计算图1
 * 2.调用内存分配，修改offset注入错误
 * 3.调用内存检查
 * 预期结果
 * 1. 检查报错
 */
TEST_F(MemoryAssignTest, NoPaddingContinuousOutputCheck_Failed) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  gert::GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(gert::ShareGraph::BuildVarConnectToSplit());

  uint32_t graph_id = 1;
  EXPECT_EQ(SUCCESS, session.AddGraph(graph_id, graph));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(ge::Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 0);
  TensorDesc desc_2(ge::Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_2);

  runtime_stub.Clear();
  Synchronizer sync;
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  sync.WaitFor(2);
  runtime_stub.Clear();

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto split = compute_graph->FindNode("split");
  split->GetOpDescBarePtr()->SetOutputOffset({0, 1024});
  EXPECT_NE(SpecialNodeChecker::Check(compute_graph), SUCCESS);
}
