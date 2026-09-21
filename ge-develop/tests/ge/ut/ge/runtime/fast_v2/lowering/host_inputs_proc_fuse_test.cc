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
#include "faker/global_data_faker.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "lowering/graph_converter.h"
#include "lowering/model_converter.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "faker/model_desc_holder_faker.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
LowerResult LoweringNetOutput(const ge::NodePtr &node, const LowerInput &lower_input);
class HostInputsProcFuseUT : public testing::Test {};

/*
 ****** ComputeGraph ******
 *       netoutput
 *          |
 *         add1
 *         /  \
 *      data1 data2
 *
 ****** ExecuteGraph ******
 * before HostInputsProcFuse pass:
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
 *            OptimizeHostInputs
 *             /               \
 * CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *          |                      |
 *       SplitTensor        SplitTensor
 *          |                      |
 *         data1                 data2
 *
 *
 */


TEST_F(HostInputsProcFuseUT, TestSingleAicoreGraph) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  ge::DumpGraph(exe_graph.get(), "HostInputsProcFuse_single_node_graph");
  ge::DumpGraph(exe_graph.get(), "HostInputsProcFuse_SingleAicoreNode");
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ge::DumpGraph(exe_graph.get(), "single_node_main_graph");
  ge::DumpGraph(exe_graph.get(), "ModelOutZeroCopy_SingleAicoreNode_main");
  ASSERT_NE(main_graph, nullptr);
#if 0
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"CalcTensorSizeFromStorage", 3},
                                        {"Data", 6},
                                        {"FreeBatchHbm", 1},
                                        {"FreeMemory", 3},
                                        {"CompatibleInferShape", 1},
                                        {"InnerData", 32},
                                        {"LaunchKernelWithHandle", 1},
                                        {"CopyFlowLaunch", 1},
                                        {"SplitRtStreams", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"SelectL1Allocator", 1},
                                        {"SelectL2Allocator", 1},
                                        {"SplitDataTensor", 2},
                                        {"CacheableTiling", 1},
                                        {"TilingAppendWorkspace", 1},
                                        {"TilingAppendDfxInfo", 1}}),
            "success");
#endif

  auto netoutput = ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph, "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData"}, {"EnsureTensorAtOutMemory"}}), "success");
  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  for (auto &node : output_data_node->GetOutDataNodes()) {
    if (node->GetType() == "NetOutput") {
      continue;
    }
    if (node->GetType() == "EnsureTensorAtOutMemory") {
      continue;
    }
    ASSERT_EQ(node->GetType(), "AllocModelOutTensor");
    EXPECT_EQ(
        FastNodeTopoChecker(node).StrictConnectFrom({{"SelectL2Allocator"}, {"CalcTensorSizeFromShape"}, {"OutputData"}}),
        "success");
  }
}

/*
 ****** ComputeGraph ******
 *            NetOutput
 *                |
 *              add2
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 *
 ****** ExecuteGraph ******
 * before HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *           \            ...  ...
 *            \          LaunchKernelWithHandle1
 *             \          /                \
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
 *           \            ...  ...
 *     CopyFlowLaunch     LaunchKernelWithHandle1
 *             \          /                \
 *              \   CopyFlowLaunch        OptimizeHostInputs
 *               \       |                          |
 *           CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 */

TEST_F(HostInputsProcFuseUT, TestTwoAicoreGraph) {
  auto compute_graph = ShareGraph::BuildTwoAddNodeGraph();
  compute_graph->TopologicalSorting();
  ge::GraphUtils::DumpGEGraph(compute_graph, "", true, "./HostInputsProcFuse_TwoAddNodeComputeGraph.txt");
  ge::GraphUtils::DumpGEGraphToOnnx(*compute_graph, "HostInputsProcFuse_TwoAddNodeComputeGraph");
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  ge::DumpGraph(exe_graph.get(), "HostInputsProcFuse_TwoAddNodeExeGraph");
  ge::DumpGraph(exe_graph.get(), "HostInputsProcFuse_TwoAddNodeExeGraph");
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ge::DumpGraph(main_graph, "HostInputsProcFuse_TwoAddNodeExeGraph2");
  ge::DumpGraph(main_graph, "HostInputsProcFuse_TwoAddNodeExeGraph2");
  ASSERT_NE(main_graph, nullptr);
#if 0
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"AllocBatchHbm", 2},
                                        {"AllocMemHbm", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"CalcTensorSizeFromShape", 2},
                                        {"CalcTensorSizeFromStorage", 4},
                                        {"Data", 6},
                                        {"FreeBatchHbm", 2},
                                        {"FreeMemory", 5},
                                        {"CompatibleInferShape", 2},
                                        {"InnerData", 51},
                                        {"LaunchKernelWithHandle", 2},
                                        {"SplitRtStreams", 1},
                                        {"CopyFlowLaunch", 2},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"SelectL1Allocator", 1},
                                        {"SelectL2Allocator", 1},
                                        {"SplitDataTensor", 2},
                                        {"CacheableTiling", 2},
                                        {"TilingAppendWorkspace", 2},
                                        {"TilingAppendDfxInfo", 2}}),
            "success");
#endif

  auto netoutput = ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph, "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData"}, {"EnsureTensorAtOutMemory"}}), "success");
  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  for (auto &node : output_data_node->GetOutDataNodes()) {
    if (node->GetType() == "NetOutput") {
      continue;
    }
    if (node->GetType() == "EnsureTensorAtOutMemory") {
      continue;
    }
    ASSERT_EQ(node->GetType(), "AllocModelOutTensor");
    EXPECT_EQ(
        FastNodeTopoChecker(node).StrictConnectFrom({{"SelectL2Allocator"}, {"CalcTensorSizeFromShape"}, {"OutputData"}}),
        "success");
  }
}

/*
 ******* ComputeGraph ******
 *            NetOutput
 *                |
 *            reshape1
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 *
 ****** ExecuteGraph ******
 *before HostInputsProcFuse pass:
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
 *              OptimizeHostInputs
 *              /                \
 * CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *        |                             |
 *    SplitTensor                    SplitTensor
 *        |                             |
 *      data1                         data2
 *
 */

TEST_F(HostInputsProcFuseUT, TestAicoreStaticGraph) {
  auto compute_graph = ShareGraph::AicoreStaticGraph();
  compute_graph->TopologicalSorting();
  ge::GraphUtils::DumpGEGraphToOnnx(*compute_graph, "HostInputsProcFuse_AicoreStaticComputeGraph");
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  GeModelBuilder builder(compute_graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  ASSERT_EQ(exe_graph->GetDirectNodesSize(), 3);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ge::DumpGraph(main_graph, "HostInputsProcFuse_AicoreStaticExeGraph");
  ge::DumpGraph(main_graph, "HostInputsProcFuse_AicoreStaticExeGraph");
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"AllocBatchHbm", 1},
                                        {"AllocMemHbm", 1},
                                        {"BuildTensor", 1},
                                        {"InferShape", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Data", 6},
                                        {"CopyD2H", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"FreeBatchHbm", 1},
                                        {"FreeMemory", 4},
                                        {"FreeTensorMemory", 1},
                                        {"InnerData", 28},
                                        {"SplitRtStreams", 1},
                                        {"LaunchKernelWithFlag", 1},
                                        {"CopyFlowLaunch", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"SelectL1Allocator", 2},
                                        {"SelectL2Allocator", 1},
                                        {"CreateHostL2Allocator", 1},
                                        {"SplitDataTensor", 2},
                                        {"SyncStream", 1}}),
            "success");
}
}  // namespace gert
