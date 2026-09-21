/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_CORE_MODEL_V2_EXECUTOR_UNITTEST_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_CORE_MODEL_V2_EXECUTOR_UNITTEST_H_

#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "stub/gert_runtime_stub.h"
#include "runtime/model_v2_executor.h"
// for test const data
#include "runtime/rt_session.h"
#include "register/node_converter_registry.h"
#include "register/kernel_registry.h"
#include "graph_builder/bg_rt_session.h"
#include "graph/debug/ge_attr_define.h"
#include "faker/model_desc_holder_faker.h"
#include "kernel/memory/host_mem_allocator.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
namespace {
memory::HostMemAllocator host_mem_allocator;
memory::HostGertMemAllocator host_gert_mem_allocator{&host_mem_allocator};
struct ExecutorWithExeGraph {
  std::unique_ptr<ModelV2Executor> executor;
  ge::ExecuteGraphPtr exe_graph;
};
struct ExeGraphWithRootModel {
  ge::ExecuteGraphPtr exe_graph;
  ge::GeRootModelPtr root_model;
};
} // namespace
inline ExeGraphWithRootModel BuildExeGraphFromSingleNode() {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  return {exe_graph, root_model};
}

inline ExecutorWithExeGraph BuildExecutorFromSingleNode() {
  auto exe_graph_2_root_model = BuildExeGraphFromSingleNode();
  auto exe_graph = exe_graph_2_root_model.exe_graph;
  if (exe_graph == nullptr) {
    return {};
  }
  static RtSession session;
  return {ModelV2Executor::Create(exe_graph, exe_graph_2_root_model.root_model, &session), exe_graph};
}

inline ge::ExecuteGraphPtr BuildExeGraphFromSingleNodeWithShapeAndRange(
    std::vector<std::initializer_list<int64_t>> shape, std::vector<std::initializer_list<int64_t>> min_shape,
    std::vector<std::initializer_list<int64_t>> max_shape) {
  auto compute_graph =
      ShareGraph::BuildSingleNodeGraph("Add", std::move(shape), std::move(min_shape), std::move(max_shape));
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.MutableModelDesc().SetReusableStreamNum(1U);
  model_desc_holder.MutableModelDesc().SetReusableEventNum(0U);
  model_desc_holder.MutableModelDesc().SetReusableNotifyNum(2);
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  return exe_graph;
}

inline std::unique_ptr<ModelV2Executor> BuildExecutorFromSingleNodeForDump(bool workspace_flag = false) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto node = compute_graph->FindFirstNodeMatchType("Add");
  ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_SINGLE_OP_SCENE, true);
  if (workspace_flag) {
    ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_AICPU_WORKSPACE_TYPE,
                              {ge::AicpuWorkSpaceType::CUST_LOG});
  }
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  root_model->SetModelName("test_model");
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();

  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  ge::ModelData model_data{};
  model_data.om_name = "test";
  static RtSession session;
  return ModelV2Executor::Create(exe_graph, model_data, root_model, &session);
}

inline std::unique_ptr<ModelV2Executor> BuildExecutorTraningTrace() {
  auto compute_graph = ShareGraph::BuildTwoAddNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
         .SetModelDescHolder(&model_desc_holder)
         .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  ge::ModelData model_data{};
  model_data.om_name = "test";
  static RtSession session;
  return ModelV2Executor::Create(exe_graph, model_data, root_model, &session);
}
} // namespace gert
#endif  // AIR_CXX_TESTS_UT_GE_RUNTIME_V2_CORE_MODEL_V2_EXECUTOR_UNITTEST_H_
