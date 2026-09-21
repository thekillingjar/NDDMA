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
#include "graph/utils/graph_utils.h"
#include "lowering/graph_converter.h"
#include "lowering/model_converter.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "faker/aicpu_taskdef_faker.h"

namespace gert {
class AicpuHostInputsFusePassUT: public testing::Test {};

/*
 ****** ComputeGraph ******
 *       netoutput
 *          |
 *         add1
 *         /  \
 *      data1 data2
 *
 ****** ExecuteGraph ******
 * before AicpuHostInputsFusePass:
 *                netoutput
 *                    |
 *          AicpuLaunchTfKernel
 *                    |
 *            UpdateAicpuIoAddr
 *           /                 \
 * MakeSureTensorAtDevice  MakeSureTensorAtDevice
 *        |                             |
 * CalTensorSizeFromStorage      CalTensorSizeFromStorage
 *        |                             |
 *    SplitTensor                    SplitTensor
 *        |                             |
 *      data1                         data2
 *
 *  after AicpuHostInputsFusePass:
 *                netoutput
 *                    |
 *            AicpuLaunchTfKernel
 *                    |
 *             UpdateAicpuIoAddr
 *                    |
 *               AicpuFuseHost
 *             /               \
 * CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *          |                      |
 *       SplitTensor        SplitTensor
 *          |                      |
 *        data1                  data2
 *
 */
TEST_F(AicpuHostInputsFusePassUT, TestSingleAicpuGraph) {
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  auto graph = ShareGraph::BuildSingleNodeGraph();
  (void)ge::AttrUtils::SetBool(graph, "_single_op_scene", true);
  graph->TopologicalSorting();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"AllocModelOutTensor", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Data", 6},
                                        {"FreeMemory", 3},
                                        {"InferShape", 1},
                                        {"InnerData", 18},
                                        {"SplitRtStreams", 1},
                                        {"UpdateAicpuIoAddr", 1},
                                        {"UpdateExtShape", 1},
                                        {"AicpuLaunchTfKernel", 1},
                                        {"AicpuFuseHost", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"SelectL1Allocator", 1},
                                        {"SelectL2Allocator", 1},
                                        {"SplitDataTensor", 2}}),
            "success");
}
}  // namespace gert