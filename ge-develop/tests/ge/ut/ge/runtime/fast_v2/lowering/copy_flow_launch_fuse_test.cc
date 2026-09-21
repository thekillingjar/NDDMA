/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/copy_flow_launch_fuse.h"
#include <gtest/gtest.h>
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/execute_graph_adapter.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "common/bg_test.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "kernel/common_kernel_impl/copy_flow_launch.h"
#include "lowering/pass_changed_kernels_info.h"

namespace gert {
namespace bg {
LowerResult LoweringNetOutput(ge::FastNode *const node, const LowerInput &lower_input);

class CopyFlowLaunchFuseUT : public BgTestAutoCreateFrame {};
namespace {
bool IsFreeMemoryNode(const ge::FastNode *node) {
  return strcmp(node->GetTypePtr(), "FreeMemory") == 0;
}
void CheckFreeMemoryInControlEdge(const ge::ExecuteGraph *graph, const std::string &launch_type, size_t free_node_size) {
  auto free_nodes = graph->GetAllNodes(IsFreeMemoryNode);
  EXPECT_EQ(free_nodes.size(), free_node_size);
  for (const auto node : free_nodes) {
    EXPECT_EQ(node->GetAllInControlEdgesSize(), 1);
    FastNodeTopoChecker topo_checker3(node);
    EXPECT_EQ(topo_checker3.StrictConnectFrom({{"CopyFlowLaunch"}, {launch_type, -1}}), "success");
  }
}
std::vector<ValueHolderPtr> CreateTiling() {
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
  };
  return ValueHolder::CreateDataOutput("Tiling", inputs, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

std::vector<ValueHolderPtr> CreateLaunchKernelCommonInputs(const vector<ValueHolderPtr> &tiling_out) {
  std::vector<ValueHolderPtr> common_inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      tiling_out[TilingContext::kOutputBlockDim],
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      tiling_out[TilingContext::kOutputScheduleMode],
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      tiling_out[static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)],
      ValueHolder::CreateFeed(0),
  };

  return common_inputs;
}

std::vector<ValueHolderPtr> CreateLaunchKernelWithFlagCommonInputs() {
  auto tiling_out = CreateTiling();
  auto inputs = CreateLaunchKernelCommonInputs(tiling_out);
  inputs.emplace_back(ValueHolder::CreateSingleDataOutput("InnerData", {}));
  return inputs;
}

std::vector<ValueHolderPtr> CreateTilingMemCheck() {
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
  };
  auto tiling_ret =
      ValueHolder::CreateDataOutput("Tiling", inputs, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  int64_t known_worspace = 5;
  tiling_ret[TilingContext::kOutputWorkspace] = ValueHolder::CreateSingleDataOutput(
      "TilingAppendWorkspace",
      {tiling_ret[TilingContext::kOutputWorkspace], ValueHolder::CreateConst(&known_worspace, sizeof(known_worspace))});
  std::vector<ValueHolderPtr> input_vec;
  input_vec.emplace_back(tiling_ret[TilingContext::kOutputTilingData]);
  input_vec.emplace_back(tiling_ret[TilingContext::kOutputWorkspace]);
  tiling_ret[TilingContext::kOutputWorkspace] = ValueHolder::CreateSingleDataOutput("TilingAppendDfxInfo", input_vec);
  return tiling_ret;
};

ValueHolderPtr CreateAllocBatchHbm() {
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
  };
  return ValueHolder::CreateSingleDataOutput("AllocBatchHbm", inputs);
}
std::vector<ValueHolderPtr> CreateLaunchKernelWithHandleCommonInputs() {
  const auto tiling_out = CreateTiling();
  auto inputs = CreateLaunchKernelCommonInputs(tiling_out);
  inputs.emplace_back(tiling_out[TilingContext::kOutputTilingKey]);
  inputs.emplace_back(ValueHolder::CreateSingleDataOutput("InnerData", {}));
  inputs.emplace_back(ValueHolder::CreateSingleDataOutput("InnerData", {}));
  return inputs;
}

std::vector<ValueHolderPtr> CreateLaunchKernelWithHandleCommonInputsMemCheck() {
  const auto tiling_out = CreateTilingMemCheck();
  auto inputs = CreateLaunchKernelCommonInputs(tiling_out);
  inputs.emplace_back(tiling_out[TilingContext::kOutputTilingKey]);
  inputs.emplace_back(ValueHolder::CreateSingleDataOutput("InnerData", {}));
  inputs.emplace_back(ValueHolder::CreateSingleDataOutput("InnerData", {}));
  return inputs;
}

ValueHolderPtr CreateMakeSureTensorAtDevice() {
  auto data = ValueHolder::CreateFeed(0);
  auto dt_holder = ValueHolder::CreateConst("Hello", 5, true);
  auto calc_tensor_size_from_storage =
      ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {dt_holder, data});

  auto split_outputs = ValueHolder::CreateDataOutput("SplitTensor", {}, 2);
  std::vector<ValueHolderPtr> copy_inputs = {
      ValueHolder::CreateFeed(0), ValueHolder::CreateSingleDataOutput("InnerData", {}),
      split_outputs[1],           calc_tensor_size_from_storage,
      split_outputs[0],           ValueHolder::CreateSingleDataOutput("InnerData", {})};
  auto make_sure_tensor_at_device = ValueHolder::CreateSingleDataOutput("MakeSureTensorAtDevice", copy_inputs);

  ValueHolder::CreateVoidGuarder("FreeMemory", make_sure_tensor_at_device, {});

  return make_sure_tensor_at_device;
}

ValueHolderPtr CreateCopyH2D() {
  auto data = ValueHolder::CreateFeed(0);
  auto dt_holder = ValueHolder::CreateConst("Hello", 5, true);
  auto calc_tensor_size_from_storage =
      ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {dt_holder, data});

  auto split_outputs = ValueHolder::CreateDataOutput("SplitTensor", {}, 2);
  std::vector<ValueHolderPtr> copy_inputs = {
      ValueHolder::CreateFeed(0), ValueHolder::CreateSingleDataOutput("InnerData", {}),
      split_outputs[1],           calc_tensor_size_from_storage,
      split_outputs[0],           ValueHolder::CreateSingleDataOutput("InnerData", {})};
  auto copy_h2d = ValueHolder::CreateSingleDataOutput("CopyH2D", copy_inputs);

  ValueHolder::CreateVoidGuarder("FreeMemory", copy_h2d, {});

  return copy_h2d;
}
}  // namespace

/*
 *before HostInputsProcFuse pass:
 *                 netoutput
 *                    |
 *           LaunchKernelWithFlag
 *           /                  \
 *  MakeSureTensorAtDevice     data2
 *        |
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 *
 * after HostInputsProcFuse pass:
 *                  netoutput
 *                      |
 *            LaunchKernelWithFlag
 *                 /           \
 *         CopyFlowLaunch      data2
 *              /
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 */

TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithFlag_1H1D_MakeSureTensorAtDevice) {
  auto make_sure_tensor_at_device = CreateMakeSureTensorAtDevice();
  auto launch_kernel_inputs = CreateLaunchKernelWithFlagCommonInputs();
  launch_kernel_inputs.emplace_back(make_sure_tensor_at_device);
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", launch_kernel_inputs);

  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 1},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"Data", 9},
                                        {"Tiling", 1},
                                        {"FreeMemory", 1},
                                        {"InnerData", 12},
                                        {"LaunchKernelWithFlag", 1},
                                        {"SplitTensor", 1},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 3},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 9},
                                        {"Tiling", 1},
                                        {"FreeMemory", 1},
                                        {"InnerData", 12},
                                        {"LaunchKernelWithFlag", 1},
                                        {"SplitTensor", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
    EXPECT_EQ(in_edge->src_output, static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg));
  }
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithFlag", 1);
}

/*
 *                 netoutput
 *                    |
 *           LaunchKernelWithFlag
 *           /                  \
 * MakeSureTensorAtDevice  MakeSureTensorAtDevice
 *        |                             |
 * CalTensorSizeFromStorage      CalTensorSizeFromStorage
 *        |                             |
 *    SplitTensor                    SplitTensor
 *        |                             |
 *      data1                         data2
 *
 *
 * after HostInputsProcFuse pass:
 *                  netoutput
 *                      |
 *            LaunchKernelWithFlag
 *                      |
 *                CopyFlowLaunch
 *              /                \
 * CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *        |                             |
 *    SplitTensor                    SplitTensor
 *        |                             |
 *      data1                         data2
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithFlag_2H1D) {
  auto launch_kernel_inputs = CreateLaunchKernelWithFlagCommonInputs();
  launch_kernel_inputs.emplace_back(CreateMakeSureTensorAtDevice());
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());
  launch_kernel_inputs.emplace_back(CreateMakeSureTensorAtDevice());
  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", launch_kernel_inputs);

  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 2},
                                        {"MakeSureTensorAtDevice", 2},
                                        {"Data", 11},
                                        {"Tiling", 1},
                                        {"FreeMemory", 2},
                                        {"InnerData", 14},
                                        {"LaunchKernelWithFlag", 1},
                                        {"SplitTensor", 2},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 4},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 11},
                                        {"Tiling", 1},
                                        {"FreeMemory", 2},
                                        {"InnerData", 14},
                                        {"LaunchKernelWithFlag", 1},
                                        {"SplitTensor", 2},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
    EXPECT_EQ(in_edge->src_output, static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg));
  }
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithFlag", 2);
}

/*
 *before HostInputsProcFuse pass:
 *                 netoutput
 *                    |
 *           LaunchKernelWithHandle
 *           /                  \
 *  MakeSureTensorAtDevice     data2
 *        |
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 *
 * after HostInputsProcFuse pass:
 *                  netoutput
 *                      |
 *            LaunchKernelWithHandle
 *                   /        \
 *          CopyFlowLaunch   data2
 *              /
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_1H1D) {
  auto make_sure_tensor_at_device = CreateMakeSureTensorAtDevice();
  auto launch_kernel_inputs = CreateLaunchKernelWithHandleCommonInputs();
  launch_kernel_inputs.emplace_back(make_sure_tensor_at_device);
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs);
  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 1},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"Data", 9},
                                        {"Tiling", 1},
                                        {"FreeMemory", 1},
                                        {"InnerData", 13},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 1},
                                        {"Tiling", 1},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 3},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 9},
                                        {"Tiling", 1},
                                        {"FreeMemory", 1},
                                        {"InnerData", 13},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 1},
                                        {"Tiling", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
  }
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 1);
}

/*
 *                netoutput
 *                    |
 *           LaunchKernelWithHandle
 *           /                 \
 * MakeSureTensorAtDevice  MakeSureTensorAtDevice
 *        |                             |
 * CalTensorSizeFromStorage      CalTensorSizeFromStorage
 *        |                             |
 *    SplitTensor                    SplitTensor
 *        |                             |
 *      data1                         data2
 *
 *  after HostInputsProcFuse pass:
 *                netoutput
 *                    |
 *           LaunchKernelWithHandle
 *                    |
 *               CopyFlowLaunch
 *             /               \
 * CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *          |                      |
 *       SplitTensor        SplitTensor
 *          |                      |
 *         data1                 data2
 *
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_2H1D) {
  auto launch_kernel_inputs = CreateLaunchKernelWithHandleCommonInputs();
  launch_kernel_inputs.emplace_back(CreateMakeSureTensorAtDevice());
  launch_kernel_inputs.emplace_back(CreateMakeSureTensorAtDevice());
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());

  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs);

  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 2},
                                        {"MakeSureTensorAtDevice", 2},
                                        {"Data", 11},
                                        {"FreeMemory", 2},
                                        {"InnerData", 15},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 2},
                                        {"Tiling", 1},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 4},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 11},
                                        {"FreeMemory", 2},
                                        {"InnerData", 15},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 2},
                                        {"Tiling", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
    FastNodeTopoChecker topo_checker(node);
    // RefFrom整改后, 数据边替代了原先的控制边, Tiling到CopyFlowLaunch之间没有控制边
    EXPECT_NE(topo_checker.InChecker().CtrlFromByType("Tiling").Result(), "success");
  }
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 2);
}
/*
 * before HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *            \            ...  ...
 *   FreeMem   \          LaunchKernelWithHandle1         FreeMem
 *         \    \          /                \             /
 *          MakeSureTensorAtDevice  MakeSureTensorAtDevice
 *                      |                          |
 *         CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 *  after HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *           \                        ...  ...
 *  FreeMem   \               FreeMem  LaunchKernel1   FreeMem
 *      \      \                  \      |  |        /
 *       CopyFlowLaunch              CopyFlowLaunch
 *                \                /            |
 *           CalTensorSizeFromStorage   CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_2LaunchKernel) {
  auto launch_kernel_inputs1 = CreateLaunchKernelWithHandleCommonInputs();
  auto make_sure_tensor_at_device1 = CreateMakeSureTensorAtDevice();
  launch_kernel_inputs1.emplace_back(make_sure_tensor_at_device1);
  launch_kernel_inputs1.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel1 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs1);

  auto launch_kernel_inputs2 = CreateLaunchKernelWithHandleCommonInputs();
  launch_kernel_inputs2.emplace_back(CreateMakeSureTensorAtDevice());
  launch_kernel_inputs2.emplace_back(CreateAllocBatchHbm());
  launch_kernel_inputs2.emplace_back(make_sure_tensor_at_device1);
  launch_kernel_inputs2.emplace_back(launch_kernel1);
  auto launch_kernel2 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs2);

  auto frame = ValueHolder::PopGraphFrame({launch_kernel2}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  ge::DumpGraph(graph.get(), "testzxx");
  EXPECT_EQ(graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 2},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 2},
                                        {"MakeSureTensorAtDevice", 2},
                                        {"Data", 18},
                                        {"FreeMemory", 2},
                                        {"InnerData", 26},
                                        {"LaunchKernelWithHandle", 2},
                                        {"SplitTensor", 2},
                                        {"Tiling", 2},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  ge::DumpGraph(graph.get(), "AFTER TOPO");
  EXPECT_EQ(graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 2},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 6},
                                        {"CopyFlowLaunch", 2},
                                        {"Data", 18},
                                        {"FreeMemory", 3},
                                        {"InnerData", 26},
                                        {"LaunchKernelWithHandle", 2},
                                        {"SplitTensor", 2},
                                        {"Tiling", 2},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
  }

  FastNodeTopoChecker topo_checker1(launch_kernel1);
  EXPECT_EQ(topo_checker1.ConnectFromByType({"MakeSureTensorAtDevice"}), false);
  EXPECT_EQ(topo_checker1.StrictConnectFrom({
                {"Data", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputBlockDim},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputScheduleMode},
                {"InnerData", 0},
                {"Tiling", static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)},
                {"Data", 0},
                {"Tiling", TilingContext::kOutputTilingKey},
                {"InnerData", 0},
                {"InnerData", 0},
                {"CopyFlowLaunch", 0},  // copy flow
                {"AllocBatchHbm", 0},
            }),
            "success");

  FastNodeTopoChecker topo_checker2(launch_kernel2);
  EXPECT_EQ(topo_checker2.StrictConnectFrom({
                {"Data", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputBlockDim},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputScheduleMode},
                {"InnerData", 0},
                {"Tiling", static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)},
                {"Data", 0},
                {"Tiling", TilingContext::kOutputTilingKey},
                {"InnerData", 0},
                {"InnerData", 0},
                {"CopyFlowLaunch", 0},  // copy flow
                {"AllocBatchHbm", 0},
                {"CopyFlowLaunch", 1},  // copy flow
                {"LaunchKernelWithHandle", 0},
            }),
            "success");
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 3);
}

/*
 * before HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle1                   LaunchKernelWithHandle3
 *            \            ...  ...                /
 *   FreeMem   \     LaunchKernelWithHandle2      /
 *         \    \          /                     /
 *                MakeSureTensorAtDevice
 *                      |
 *            CalTensorSizeFromStorage
 *                      |
 *                 SplitTensor
 *                      |
 *                    data1
 *
 *  after HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle1
 *           \                        ...  ...
 *  FreeMem   \          FreeMem  LaunchKernel2      FreeMem  LaunchKernel3
 *      \      \             \      |                \      |
 *       CopyFlowLaunch      CopyFlowLaunch       CopyFlowLaunch
 *                     \           /             /
 *                      CalTensorSizeFromStorage
 *                          |
 *                 SplitTensor
 *                      |
 *                    data1
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_3LaunchKernel) {
  auto launch_kernel_inputs1 = CreateLaunchKernelWithHandleCommonInputs();
  auto make_sure_tensor_at_device1 = CreateMakeSureTensorAtDevice();
  launch_kernel_inputs1.emplace_back(make_sure_tensor_at_device1);
  launch_kernel_inputs1.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel1 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs1);

  auto launch_kernel_inputs2 = CreateLaunchKernelWithHandleCommonInputs();
  launch_kernel_inputs2.emplace_back(make_sure_tensor_at_device1);
  launch_kernel_inputs2.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel2 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs2);

  auto launch_kernel_inputs3 = CreateLaunchKernelWithHandleCommonInputs();
  launch_kernel_inputs3.emplace_back(make_sure_tensor_at_device1);
  launch_kernel_inputs3.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel3 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs3);

  auto frame = ValueHolder::PopGraphFrame({launch_kernel3}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 3},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 1},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"Data", 23},
                                        {"FreeMemory", 1},
                                        {"InnerData", 35},
                                        {"LaunchKernelWithHandle", 3},
                                        {"SplitTensor", 1},
                                        {"Tiling", 3},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = back_options;
  new_options["ge.topoSortingMode"] = "0";  // BFS
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  EXPECT_EQ(graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 3},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 7},
                                        {"CopyFlowLaunch", 3},
                                        {"Data", 23},
                                        {"FreeMemory", 3},
                                        {"InnerData", 35},
                                        {"LaunchKernelWithHandle", 3},
                                        {"SplitTensor", 1},
                                        {"Tiling", 3},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
  }

  FastNodeTopoChecker topo_checker1(launch_kernel1);
  EXPECT_EQ(topo_checker1.ConnectFromByType({"MakeSureTensorAtDevice"}), false);
  EXPECT_EQ(topo_checker1.StrictConnectFrom({
                {"Data", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputBlockDim},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputScheduleMode},
                {"InnerData", 0},
                {"Tiling", static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)},
                {"Data", 0},
                {"Tiling", TilingContext::kOutputTilingKey},
                {"InnerData", 0},
                {"InnerData", 0},
                {"CopyFlowLaunch", 0},  // copy flow
                {"AllocBatchHbm", 0},
            }),
            "success");

  FastNodeTopoChecker topo_checker2(launch_kernel2);
  EXPECT_EQ(topo_checker2.StrictConnectFrom({{"Data", 0},
                                             {"InnerData", 0},
                                             {"Tiling", TilingContext::kOutputBlockDim},
                                             {"InnerData", 0},
                                             {"InnerData", 0},
                                             {"InnerData", 0},
                                             {"InnerData", 0},
                                             {"Tiling", TilingContext::kOutputScheduleMode},
                                             {"InnerData", 0},
                                             {"Tiling", static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)},
                                             {"Data", 0},
                                             {"Tiling", TilingContext::kOutputTilingKey},
                                             {"InnerData", 0},
                                             {"InnerData", 0},
                                             {"CopyFlowLaunch", 0},  // copy flow
                                             {"AllocBatchHbm", 0}}),
            "success");
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 3);
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}

/*
 *               netoutput
 *                    |
 *          LaunchKernelWithHandle
 *                   \/
 *          MakeSureTensorAtDevice
 *                   |
 *          CalTensorSizeFromStorage
 *                   |
 *               SplitTensor
 *                   |
 *                 data1
 *
 *  after HostInputsProcFuse pass:
 *               netoutput
 *                   |
 *          LaunchKernelWithHandle
 *                   \/
 *            CopyFlowLaunch
 *                    |
 *           CalTensorSizeFromStorage
 *                    |
 *                SplitTensor
 *                    |
 *                  data1
 *
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_2H1D_2) {
  auto launch_kernel_inputs = CreateLaunchKernelWithHandleCommonInputs();
  auto make_sure_tensor_at_device = CreateMakeSureTensorAtDevice();
  launch_kernel_inputs.emplace_back(make_sure_tensor_at_device);
  launch_kernel_inputs.emplace_back(make_sure_tensor_at_device);
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());

  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs);
  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 1},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"Data", 9},
                                        {"FreeMemory", 1},
                                        {"InnerData", 13},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 1},
                                        {"Tiling", 1},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 3},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 9},
                                        {"FreeMemory", 1},
                                        {"InnerData", 13},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 1},
                                        {"Tiling", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
  }
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 1);
}

/*
 *before HostInputsProcFuse pass:
 *                 netoutput
 *                    |
 *           LaunchKernelWithFlag
 *           /                  \
 *     CopyH2D                  data2
 *        |
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 *
 * after HostInputsProcFuse pass:
 *                  netoutput
 *                      |
 *            LaunchKernelWithFlag
 *                 /           \
 *         CopyFlowLaunch      data2
 *              /
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithFlag_1H1D_H2D) {
  auto copy_h2d = CreateCopyH2D();
  auto launch_kernel_inputs = CreateLaunchKernelWithFlagCommonInputs();
  launch_kernel_inputs.emplace_back(copy_h2d);
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", launch_kernel_inputs);
  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  auto original_copyh2d_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D");
  ASSERT_EQ(original_copyh2d_nodes.size(), 1UL);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 1},
                                        {"CopyH2D", 1},
                                        {"Data", 9},
                                        {"Tiling", 1},
                                        {"FreeMemory", 1},
                                        {"InnerData", 12},
                                        {"LaunchKernelWithFlag", 1},
                                        {"SplitTensor", 1},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 3},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 9},
                                        {"Tiling", 1},
                                        {"FreeMemory", 1},
                                        {"InnerData", 12},
                                        {"LaunchKernelWithFlag", 1},
                                        {"SplitTensor", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  ASSERT_EQ(copy_flow_launch_nodes.size(), 1UL);
  const auto in_edge = copy_flow_launch_nodes[0]->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(in_edge, nullptr);
  const auto src_node = in_edge->src;
  ASSERT_NE(src_node, nullptr);
  EXPECT_EQ(src_node->GetType(), "Tiling");
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithFlag", 1);
  // 校验PassChangedInfo是否成功设置
  auto pass_changed_info =
      original_copyh2d_nodes[0]->GetOpDescBarePtr()->TryGetExtAttr(kPassChangedInfo, PassChangedKernels{});
  ASSERT_EQ(pass_changed_info.pass_changed_kernels.size(), 1UL);
  std::pair<KernelNameAndIdx, KernelNameAndIdx> expect_changed_info{{original_copyh2d_nodes[0]->GetName(), 0UL},
                                                                    {copy_flow_launch_nodes[0]->GetName(), 0UL}};
  EXPECT_EQ(pass_changed_info.pass_changed_kernels[0], expect_changed_info);
}

/*
 * before HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *            \            ...  ...
 *   FreeMem   \        LaunchKernelWithHandle1         FreeMem
 *         \    \       /                \             /
 *              CopyH2D             MakeSureTensorAtDevice
 *                 \                               |
 *         CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 *  after HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *           \                        ...  ...
 *  FreeMem   \               FreeMem  LaunchKernel1   FreeMem
 *      \      \                  \      |  |        /
 *       CopyFlowLaunch              CopyFlowLaunch
 *                \                /            |
 *           CalTensorSizeFromStorage   CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_2LaunchKernel_H2D) {
  auto launch_kernel_inputs1 = CreateLaunchKernelWithHandleCommonInputs();
  auto copy_h2d = CreateCopyH2D();
  launch_kernel_inputs1.emplace_back(copy_h2d);
  launch_kernel_inputs1.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel1 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs1);

  auto launch_kernel_inputs2 = CreateLaunchKernelWithHandleCommonInputs();
  launch_kernel_inputs2.emplace_back(CreateMakeSureTensorAtDevice());
  launch_kernel_inputs2.emplace_back(CreateAllocBatchHbm());
  launch_kernel_inputs2.emplace_back(copy_h2d);
  launch_kernel_inputs2.emplace_back(launch_kernel1);
  auto launch_kernel2 = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs2);

  auto frame = ValueHolder::PopGraphFrame({launch_kernel2}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 2},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 2},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"CopyH2D", 1},
                                        {"Data", 18},
                                        {"FreeMemory", 2},
                                        {"InnerData", 26},
                                        {"LaunchKernelWithHandle", 2},
                                        {"SplitTensor", 2},
                                        {"Tiling", 2},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 2},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Const", 6},
                                        {"CopyFlowLaunch", 2},
                                        {"Data", 18},
                                        {"FreeMemory", 3},
                                        {"InnerData", 26},
                                        {"LaunchKernelWithHandle", 2},
                                        {"SplitTensor", 2},
                                        {"Tiling", 2},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
  }

  FastNodeTopoChecker topo_checker1(launch_kernel1);
  EXPECT_EQ(topo_checker1.ConnectFromByType({"MakeSureTensorAtDevice"}), false);
  EXPECT_EQ(topo_checker1.StrictConnectFrom({
                {"Data", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputBlockDim},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputScheduleMode},
                {"InnerData", 0},
                {"Tiling", static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)},
                {"Data", 0},
                {"Tiling", TilingContext::kOutputTilingKey},
                {"InnerData", 0},
                {"InnerData", 0},
                {"CopyFlowLaunch", 0},  // copy flow
                {"AllocBatchHbm", 0},
            }),
            "success");

  FastNodeTopoChecker topo_checker2(launch_kernel2);
  EXPECT_EQ(topo_checker2.StrictConnectFrom({
                {"Data", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputBlockDim},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"InnerData", 0},
                {"Tiling", TilingContext::kOutputScheduleMode},
                {"InnerData", 0},
                {"Tiling", static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)},
                {"Data", 0},
                {"Tiling", TilingContext::kOutputTilingKey},
                {"InnerData", 0},
                {"InnerData", 0},
                {"CopyFlowLaunch", 0},  // copy flow
                {"AllocBatchHbm", 0},
                {"CopyFlowLaunch", 1},  // copy flow
                {"LaunchKernelWithHandle", 0},
            }),
            "success");
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 3);
}
/*
 *before HostInputsProcFuse pass:
 *                 netoutput
 *                    |
 *           LaunchKernelWithHandle
 *           /                  \
 *  MakeSureTensorAtDevice     data2
 *        |
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 *
 * after HostInputsProcFuse pass:
 *                  netoutput
 *                      |
 *            LaunchKernelWithHandle
 *                   /        \
 *          CopyFlowLaunch   data2
 *              /
 * CalTensorSizeFromStorage
 *        |
 *    SplitTensor
 *        |
 *      data1
 *
 */
TEST_F(CopyFlowLaunchFuseUT, TestLaunchKernelWithHandle_1H1D_MEMCHECK) {
  auto make_sure_tensor_at_device = CreateMakeSureTensorAtDevice();
  auto launch_kernel_inputs = CreateLaunchKernelWithHandleCommonInputsMemCheck();
  launch_kernel_inputs.emplace_back(make_sure_tensor_at_device);
  launch_kernel_inputs.emplace_back(CreateAllocBatchHbm());
  auto launch_kernel = ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", launch_kernel_inputs);
  auto frame = ValueHolder::PopGraphFrame({launch_kernel}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 2},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"Data", 9},
                                        {"FreeMemory", 1},
                                        {"InnerData", 13},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 1},
                                        {"Tiling", 1},
                                        {"TilingAppendDfxInfo", 1},
                                        {"TilingAppendWorkspace", 1},
                                        {"NetOutput", 1}}),
            "success");

  bool changed = false;
  EXPECT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_EQ(changed, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(graph.get())
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"Const", 4},
                                        {"CopyFlowLaunch", 1},
                                        {"Data", 9},
                                        {"FreeMemory", 1},
                                        {"InnerData", 13},
                                        {"LaunchKernelWithHandle", 1},
                                        {"SplitTensor", 1},
                                        {"Tiling", 1},
                                        {"TilingAppendDfxInfo", 1},
                                        {"TilingAppendWorkspace", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto copy_flow_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch");
  for (auto &node : copy_flow_launch_nodes) {
    const auto in_edge = node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
    ASSERT_NE(in_edge, nullptr);
    const auto src_node = in_edge->src;
    ASSERT_NE(src_node, nullptr);
    EXPECT_EQ(src_node->GetType(), "Tiling");
    // TilingAppendDfxInfo仍然有一体控制边到CopyFlowLaunch
    FastNodeTopoChecker topo_checker(node);
    EXPECT_EQ(topo_checker.InChecker().CtrlFromByType("TilingAppendDfxInfo").Result(), "success");
  }
  CheckFreeMemoryInControlEdge(graph.get(), "LaunchKernelWithHandle", 1);
}
}  // namespace bg
}  // namespace gert