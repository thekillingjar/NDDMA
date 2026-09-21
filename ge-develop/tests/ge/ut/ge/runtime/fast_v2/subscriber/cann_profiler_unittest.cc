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
#include "base/registry/op_impl_space_registry_v2.h"
#include <memory>
#include "common/bg_test.h"
#include "kernel/common_kernel_impl/infer_shape.h"
#include "core/executor_error_code.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "engine/node_converter_utils.h"
#include "faker/fake_value.h"
#include "macro_utils/dt_public_scope.h"
#include "common/global_variables/diagnose_switch.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/model_v2_executor_test_helper.h"
#include "runtime/subscriber/global_profiler.h"
#include "common/fake_node_helper.h"
#include "core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "runtime/model_v2_executor.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"
#include "lowering/model_converter.h"
#include "subscriber/profiler/cann_host_profiler.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "subscriber/profiler/cann_memory_profiler.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "common/fake_node_helper.h"
#include "subscriber/profiler/base_executor_profiler.h"
#include "core/utils/tensor_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "common/sgt_slice_type.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph_builder/bg_tiling.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph/operator_factory_impl.h"
#include "lowering/exe_graph_serializer.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/utils/graph_dump_utils.h"
#include "core/executor/executor_base_def.h"
#include "register/op_compile_info_base.h"
#include "faker/space_registry_faker.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
class CannProfilerUT : public bg::BgTest {
 protected:
  void SetUp() {
    GlobalProfilingWrapper::GetInstance()->Free();
    ge::diagnoseSwitch::DisableProfiling();
    std::hash<std::string> hs;
    auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
      auto report_data = reinterpret_cast<ReporterData *>(data);
      std::string tag_name(report_data->tag);
      if (type == static_cast<uint32_t>(MSPROF_REPORTER_HASH)) {
        auto *hash_data = reinterpret_cast<MsprofHashData *>(data);
        std::string name((char *)hash_data->data, hash_data->dataLen);
        hash_data->hashId = hs(name);
      }
      return 0;
    };
    ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  }
  void TearDown() {
    GlobalProfilingWrapper::GetInstance()->Free();
    ge::diagnoseSwitch::DisableProfiling();
  }
};
namespace {
Node *FindExeNodeByType(const ExecutionData *execution_data, const std::string &type) {
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    auto kernel_extend_info =
        reinterpret_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[i]->context.kernel_extend_info);
    if (kernel_extend_info->GetKernelType() == type) {
      return execution_data->base_ed.nodes[i];
    }
  }
  return nullptr;
}

void TestReportNodeBasicInfo() {
  auto execution_data_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(ExecutionData)]);
  auto execution_data = reinterpret_cast<ExecutionData *>(execution_data_holder.get());
  auto infer_shape_node = FakeNodeHelper::FakeNode("test1", "InferShape", 1);

  // build fake ai core launch node
  auto block_dims = 32;
  auto context = KernelRunContextFaker()
                     .NodeName("test1")
                     .NodeType("test1")
                     .KernelType("LaunchKernelWithHandle")
                     .KernelName("LaunchKernelWithHandle")
                     .KernelIONum(3, 1)
                     .Build();

  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 4;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = 0;
  context.value_holder[2].Set(reinterpret_cast<void *>(block_dims), nullptr);
  memcpy(&launch_node->context, context.context, sizeof(KernelRunContext) + 4 * sizeof(AsyncAnyValue *));
  // build fake execution data with 2 exe nodes
  execution_data->base_ed.node_num = 2;
  std::vector<Node *> nodes{&infer_shape_node.node, launch_node};
  execution_data->base_ed.nodes = nodes.data();

  auto model_executor = BuildExecutorFromSingleNodeForDump();
  ModelV2ExecutorTestHelper::SetExecutionData(execution_data, kMainExeGraph, model_executor.get());
  auto profiler = model_executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(
      BuiltInSubscriberType::kCannProfilerV2);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime, ProfilingType::kDevice});
  profiler->Init();
  CannProfilerV2::OnExecuteEvent(kMainExeGraph, profiler, kExecuteStart, launch_node, 0);
  CannProfilerV2::OnExecuteEvent(kMainExeGraph, profiler, kExecuteEnd, launch_node, 0);
  free(launch_node);
}

void TestReportTaskMemory() {
  auto execution_data_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(ExecutionData)]);
  auto execution_data = reinterpret_cast<ExecutionData *>(execution_data_holder.get());
  auto infer_shape_node = FakeNodeHelper::FakeNode("test1", "InferShape", 1);

  // build fake AllocMemHbm node
  auto context = KernelRunContextFaker()
                     .NodeName("test1")
                     .NodeType("test1")
                     .KernelType("FreeMemHbm")
                     .KernelName("FreeMemHbm")
                     .KernelIONum(2, 0)
                     .Build();

  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *free_hbm_node = (Node *)malloc(size);
  auto model_executor = BuildExecutorFromSingleNodeForDump();
  free_hbm_node->node_id = 0;
  auto profiler = model_executor->GetSubscribers().MutableBuiltInSubscriber<CannMemoryProfiler>(
      BuiltInSubscriberType::kMemoryProfiler);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kMemory});
  auto allocator = memory::CachingMemAllocator::GetAllocator(0UL);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(allocator.get());
  single_stream_l2_allocator.SetL1Allocator(allocator.get());
  size_t size1 = 128UL;
  size_t size2 = 64UL;
  auto tensor_data1 = single_stream_l2_allocator.MallocTensorData(size1);
  auto tensor_data2 = single_stream_l2_allocator.MallocTensorData(size2);
  context.value_holder[0].Set(reinterpret_cast<void *>(&tensor_data1), nullptr);
  context.value_holder[1].Set(reinterpret_cast<void *>(&tensor_data2), nullptr);

  memcpy(&free_hbm_node->context, context.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  // build fake execution data with 2 exe nodes
  execution_data->base_ed.node_num = 2;
  std::vector<Node *> nodes{&infer_shape_node.node, free_hbm_node};
  execution_data->base_ed.nodes = nodes.data();

  ModelV2ExecutorTestHelper::SetExecutionData(execution_data, kMainExeGraph, model_executor.get());
  profiler->Init();
  tensor_data1.Free();
  tensor_data2.Free();
  CannMemoryProfiler::OnExecuteEvent(0, profiler, kModelStart, free_hbm_node, 0);
  CannMemoryProfiler::OnExecuteEvent(0, profiler, kExecuteEnd, free_hbm_node, 0);
  // execution_data is nullptr
  ModelV2ExecutorTestHelper::SetExecutionData(nullptr, kMainExeGraph, model_executor.get());
  profiler->is_device_prof_inited_ = false;
  profiler->Init();
  free(free_hbm_node);
}
}  // namespace

TEST_F(CannProfilerUT, InitCannHost_Ok) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost});
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  auto profiler =
      executor->GetSubscribers().MutableBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler);
  EXPECT_TRUE(profiler->is_host_prof_inited_);
  EXPECT_FALSE(profiler->host_sch_infos_.empty());
}

// todo: 此处是覆盖profiling_v1中代码，后续需要伴随全部切换profiling_v2后删除
TEST_F(CannProfilerUT, InitCannDevice_Ok) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  ASSERT_EQ(executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                              outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(executor->UnLoad(), ge::GRAPH_SUCCESS);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  rtStreamDestroy(stream);
}

TEST_F(CannProfilerUT, NotInitCannHost_Ok_DisableProfiling) {
  ge::diagnoseSwitch::DisableProfiling();
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(executor, nullptr);
  auto profiler =
      executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  EXPECT_FALSE(profiler->is_device_prof_inited_);
}

TEST_F(CannProfilerUT, FillShapeInfo_Ok_WithEmptyChain) {
  TensorInfoWrapper wrapper{};
  wrapper.tensor_num = 5;
  auto any_value = new AsyncAnyValue();
  StorageShape storage_shape{{1, 2, 3, 4}, {1, 2, 3, 4}};
  reinterpret_cast<Chain *>(any_value)->Set(&storage_shape, nullptr);
  wrapper.shapes.resize(5, nullptr);
  wrapper.shapes[1] = reinterpret_cast<Chain *>(any_value);
  auto empty_any_value = new AsyncAnyValue();
  wrapper.shapes[2] = reinterpret_cast<Chain *>(empty_any_value);
  wrapper.FillShapeInfo();
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&wrapper.tensor_info.data[68]), 1);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&wrapper.tensor_info.data[72]), 2);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&wrapper.tensor_info.data[76]), 3);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&wrapper.tensor_info.data[80]), 4);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&wrapper.tensor_info.data[82]), 0);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&wrapper.tensor_info.data[112]),
            0);
  delete any_value;
  delete empty_any_value;
}

TEST_F(CannProfilerUT, InitCannProfiler_InitTensorInfoOk_WithMultipleInputNode) {
  auto compute_graph = ShareGraph::BuildFakeNodeGraphWithMultipleInput();
  compute_graph->SetGraphUnknownFlag(true);
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Fake", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice, ProfilingType::kTaskTime});
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  auto exe_node = FindExeNodeByType(
      static_cast<const ExecutionData *>(model_executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData()),
      "CompatibleInferShape");
  const auto &node_name = static_cast<const ComputeNodeInfo *>(exe_node->context.compute_node_info)->GetNodeName();
  auto profilerV2 = model_executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(
      BuiltInSubscriberType::kCannProfilerV2);
  auto name_hash = MsprofGetHashId(node_name, strlen(node_name));
  auto iter = profilerV2->GetNamesToAddInfos().find(name_hash);
  EXPECT_NE(iter, profilerV2->GetNamesToAddInfos().cend());
  EXPECT_EQ(iter->second.tensor_info_wrappers.size(), 2UL);
  EXPECT_EQ(iter->second.tensor_info_wrappers[0].shapes.size(), 5UL);
  EXPECT_EQ(iter->second.tensor_info_wrappers[1].shapes.size(), 2UL);
  auto context = reinterpret_cast<KernelContext *>(&exe_node->context);
  auto input_start_pos = static_cast<size_t>(kernel::CompatibleInferShapeOrRangeInputIndex::kInputNum);
  for (size_t i = 0UL; i<5UL; ++i) {
    EXPECT_EQ((void *)context->MutableInput(input_start_pos + i),
              iter->second.tensor_info_wrappers[0].shapes[i]);
  }

  EXPECT_EQ((void *)context->MutableInput(input_start_pos + 5),
            iter->second.tensor_info_wrappers[1].shapes[0]);
  EXPECT_EQ((void *)context->GetOutput(0), iter->second.tensor_info_wrappers[1].shapes[1]);
}

TEST_F(CannProfilerUT, ReportTaskInfo_Ok_OnExecutorWithFakeExecutionDataV2) {
  size_t api_count = 0;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    auto report_data = reinterpret_cast<ReporterData *>(data);
    std::string tag_name(report_data->tag);
    if (type == static_cast<uint32_t>(MSPROF_REPORTER_HASH)) {
      auto *hash_data = reinterpret_cast<MsprofHashData *>(data);
      std::string name((char *)hash_data->data, hash_data->dataLen);
      hash_data->hashId = 1;
    }
    if (type == ge::InfoType::kCompactInfo) {
      auto node_basic_info = reinterpret_cast<MsprofCompactInfo *>(data);
      EXPECT_EQ(MSPROF_REPORT_NODE_LEVEL, node_basic_info->level);
      EXPECT_NE(1, *reinterpret_cast<uint32_t *>(&node_basic_info->data.info[0]));
      auto &prof_node_basic_info = node_basic_info->data.nodeBasicInfo;
      EXPECT_EQ(MSPROF_GE_TASK_TYPE_AI_CORE, prof_node_basic_info.taskType);
      EXPECT_EQ(32, prof_node_basic_info.blockDim);
    }

    if (type == ge::InfoType::kApi) {
      ++api_count;
    }
    return 0;
  };
  ge::AutoProfilingTestWithExpectedFunc(TestReportNodeBasicInfo, check_func);
  EXPECT_EQ(api_count, 1);
}

TEST_F(CannProfilerUT, ReportMemoryInfo_Ok_OnExecutorWithFakeExecutionDataV2) {
  size_t cur_info_count = 0;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == ge::InfoType::kInfo) {
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_NE(info->dataLen, 0);
      if (info->type == MSPROF_REPORT_NODE_TASK_MEMORY_TYPE) {
        ++cur_info_count;
        auto memory_info_data = reinterpret_cast<MsprofMemoryInfo *>(info->data);
        std::cout << "Report memory info: node_id: " << memory_info_data->nodeId
                  << ", addr: " << memory_info_data->addr
                  << ", size: " << memory_info_data->size
                  << ", total allocate size: " << memory_info_data->totalAllocateMemory
                  << ", total reserve size: " << memory_info_data->totalReserveMemory << std::endl;
      }
    }
    return 0;
  };
  ge::AutoProfilingTestWithExpectedFunc(TestReportTaskMemory, check_func);
  EXPECT_EQ(cur_info_count, 4);
}

TEST_F(CannProfilerUT, InitProfiler_InitTaskAndTensorInfoOk_WithEmptyComputeNodeInfo) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto execution_data = reinterpret_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  for (size_t i=0UL; i<execution_data->base_ed.node_num; ++i) {
    std::string kernel_type = reinterpret_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[i]->context.kernel_extend_info)->GetKernelType();
    if (kernel_type == "LaunchKernelWithHandle") {
      const_cast<KernelRunContext *>(&execution_data->base_ed.nodes[i]->context)->compute_node_info = nullptr;
    }
  }
  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  EXPECT_EQ(profiler->InitForCannDevice(execution_data), ge::SUCCESS);
}

TEST_F(CannProfilerUT, InitProfiler_DefaultValueUin32Max_SetEngineType) {
  auto node1 = FakeNodeHelper::FakeNode("test1", "LaunchHcomKernel", 1);
  CannProfilerV2 profilerV2{nullptr};
  profilerV2.prof_extend_infos_.resize(10, ProfExtendInfo{});
  auto kernel_extend_info1 = reinterpret_cast<const KernelExtendInfo *>(
      reinterpret_cast<KernelContext *>(node1.context.context)->GetKernelExtend());
  profilerV2.SetEngineType(node1.node.node_id, kernel_extend_info1->GetKernelType(), nullptr);
  EXPECT_EQ(profilerV2.prof_extend_infos_[node1.node.node_id].engine_type,
            static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_HCCL));

  auto node2 = FakeNodeHelper::FakeNode("test1", "LaunchFFTSPlusTask", 2);
  auto kernel_extend_info2 = reinterpret_cast<const KernelExtendInfo *>(
      reinterpret_cast<KernelContext *>(node2.context.context)->GetKernelExtend());
  profilerV2.SetEngineType(node2.node.node_id, kernel_extend_info2->GetKernelType(), nullptr);
  EXPECT_EQ(profilerV2.prof_extend_infos_[node2.node.node_id].engine_type, std::numeric_limits<uint32_t>::max());

  auto node3 = FakeNodeHelper::FakeNode("test1", "AicpuLaunchTfKernel", 3);
  auto kernel_extend_info3 = reinterpret_cast<const KernelExtendInfo *>(
      reinterpret_cast<KernelContext *>(node3.context.context)->GetKernelExtend());
  profilerV2.SetEngineType(node3.node.node_id, kernel_extend_info3->GetKernelType(), nullptr);
  EXPECT_EQ(profilerV2.prof_extend_infos_[node3.node.node_id].engine_type,
            static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CPU));
}

TEST_F(CannProfilerUT, InitContextInfo) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  auto node = compute_graph->FindNode("add1");
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV");
  // add cmo context info
  for (size_t i = 0; i < CMO_IDX_KEY.size(); i++) {
    const std::string ctx_id_attr = "_data_prof_ctx_id_v" + CMO_IDX_KEY[i];
    const std::string name_attr = "_data_prof_name_v" + CMO_IDX_KEY[i];
    const std::string type_attr = "_data_prof_type" + CMO_IDX_KEY[i];
    std::vector<uint32_t> cmo_context_ids{1, 2, 3};
    std::stringstream cmo_context_name;
    cmo_context_name << "write_back_" << i;
    uint32_t cmo_context_type = RT_CTX_TYPE_FLUSH_DATA + i;
    (void)ge::AttrUtils::SetListInt(op_desc, ctx_id_attr, cmo_context_ids);
    (void)ge::AttrUtils::SetStr(op_desc, name_attr, cmo_context_name.str());
    (void)ge::AttrUtils::SetInt(op_desc, type_attr, cmo_context_type);
  }

  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto execution_data = reinterpret_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  profiler->prof_extend_infos_.resize(execution_data->base_ed.node_num, ProfExtendInfo{});
  profiler->node_addition_infos_.resize(execution_data->base_ed.node_num);
  profiler->exe_node_id_to_profiling_filler_.resize(execution_data->base_ed.node_num, nullptr);
  profiler->node_id_to_profiler_node_id_.resize(execution_data->base_ed.node_num);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    const auto kernel_extend_info =
        static_cast<const KernelExtendInfo *>(execute_node->context.kernel_extend_info);
    if (kernel_extend_info == nullptr) {
      continue;
    }
    const auto compute_node_info =
        static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[i]->context.compute_node_info);
    uint64_t name_hash_id = 0;
    uint64_t node_type_hash_id = 0;
    if (compute_node_info == nullptr) {
      continue;
    } else {
      const auto &node_type = compute_node_info->GetNodeType();
      node_type_hash_id = MsprofGetHashId(node_type, strlen(node_type));
      const auto &node_name = compute_node_info->GetNodeName();
      name_hash_id = MsprofGetHashId(node_name, strlen(node_name));
    }
    if (!ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_extend_info->GetKernelType())) {
      continue;
    }
    ProfNodeAdditionInfo addition_info{};
    profiler->name_hashes_to_node_addition_infos_[name_hash_id] = std::move(addition_info);
    profiler->node_addition_infos_[execute_node->node_id][0] = &(profiler->name_hashes_to_node_addition_infos_[name_hash_id]);
    profiler->prof_extend_infos_[execute_node->node_id].node_name_idx = name_hash_id;
    profiler->prof_extend_infos_[execute_node->node_id].node_type_idx = node_type_hash_id;
    profiler->name_hash_to_node_id_.insert({name_hash_id, execute_node->node_id});
  }

  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  const std::vector<uint32_t> input_dims {1, 2, 3, 4};
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", {});
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", input_dims);
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    EXPECT_EQ(profiler->DoProf(execute_node), ge::SUCCESS);
  }
}

TEST_F(CannProfilerUT, InitContextInfo_MixVector) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  auto node = compute_graph->FindNode("add1");
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_VECTOR_CORE");
  // add cmo context info
  for (size_t i =0UL; i < CMO_IDX_KEY.size(); i++) {
    const std::string ctx_id_attr = "_data_prof_ctx_id_v" + CMO_IDX_KEY[i];
    const std::string name_attr = "_data_prof_name_v" + CMO_IDX_KEY[i];
    const std::string type_attr = "_data_prof_type" + CMO_IDX_KEY[i];
    std::vector<uint32_t> cmo_context_ids{1, 2, 3};
    std::stringstream cmo_context_name;
    cmo_context_name << "write_back_" << i;
    uint32_t cmo_context_type = RT_CTX_TYPE_FLUSH_DATA + i;
    (void)ge::AttrUtils::SetListInt(op_desc, ctx_id_attr, cmo_context_ids);
    (void)ge::AttrUtils::SetStr(op_desc, name_attr, cmo_context_name.str());
    (void)ge::AttrUtils::SetInt(op_desc, type_attr, cmo_context_type);
  }

  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto execution_data = reinterpret_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  profiler->prof_extend_infos_.resize(execution_data->base_ed.node_num, ProfExtendInfo{});
  profiler->node_addition_infos_.resize(execution_data->base_ed.node_num);
  profiler->exe_node_id_to_profiling_filler_.resize(execution_data->base_ed.node_num, nullptr);
  profiler->node_id_to_profiler_node_id_.resize(execution_data->base_ed.node_num);
  profiler->exe_node_id_to_profiling_wrappers_.resize(execution_data->base_ed.node_num);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    const auto kernel_extend_info =
        static_cast<const KernelExtendInfo *>(execute_node->context.kernel_extend_info);
    if (kernel_extend_info == nullptr) {
      continue;
    }
    const auto compute_node_info =
        static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[i]->context.compute_node_info);
    uint64_t name_hash_id = 0;
    uint64_t node_type_hash_id = 0;
    if (compute_node_info == nullptr) {
      continue;
    } else {
      const auto &node_type = compute_node_info->GetNodeType();
      node_type_hash_id = MsprofGetHashId(node_type, strlen(node_type));
      const auto &node_name = compute_node_info->GetNodeName();
      name_hash_id = MsprofGetHashId(node_name, strlen(node_name));
    }
    if (!ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_extend_info->GetKernelType())) {
      continue;
    }
    ProfNodeAdditionInfo addition_info{};
    profiler->name_hashes_to_node_addition_infos_[name_hash_id] = std::move(addition_info);
    profiler->node_addition_infos_[execute_node->node_id][0] = &(profiler->name_hashes_to_node_addition_infos_[name_hash_id]);
    profiler->node_addition_infos_[execute_node->node_id][0]->mix_launch_enable = true;
    profiler->prof_extend_infos_[execute_node->node_id].node_name_idx = name_hash_id;
    profiler->prof_extend_infos_[execute_node->node_id].node_type_idx = node_type_hash_id;
    profiler->name_hash_to_node_id_.insert({name_hash_id, execute_node->node_id});
  }

  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  const std::vector<uint32_t> input_dims {1, 2, 3, 4};
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", {});
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", input_dims);
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    EXPECT_EQ(profiler->DoProf(execute_node), ge::SUCCESS);
  }
}

TEST_F(CannProfilerUT, InitContextInfoByAttr) {
  const std::vector<uint32_t> input_dims {1, 2, 3, 4};
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  auto node = compute_graph->FindNode("add1");
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", input_dims);
  ge::AttrUtils::SetInt(op_desc, "_ffts_prof_ctx_type", RT_CTX_TYPE_AICORE);
  // add cmo context info
  for (size_t i = 0; i < CMO_IDX_KEY.size(); i++) {
    const std::string ctx_id_attr = "_data_prof_ctx_id_v" + CMO_IDX_KEY[i];
    const std::string name_attr = "_data_prof_name_v" + CMO_IDX_KEY[i];
    const std::string type_attr = "_data_prof_type" + CMO_IDX_KEY[i];
    std::vector<uint32_t> cmo_context_ids{1, 2, 3};
    std::string cmo_context_name = "write_back_123";
    uint32_t cmo_context_type = RT_CTX_TYPE_FLUSH_DATA + i;
    (void)ge::AttrUtils::SetListInt(op_desc, ctx_id_attr, cmo_context_ids);
    (void)ge::AttrUtils::SetStr(op_desc, name_attr, cmo_context_name);
    (void)ge::AttrUtils::SetInt(op_desc, type_attr, cmo_context_type);
  }

  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto execution_data = reinterpret_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());

  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  profiler->prof_extend_infos_.resize(execution_data->base_ed.node_num, ProfExtendInfo{});
  profiler->node_addition_infos_.resize(execution_data->base_ed.node_num);
  profiler->exe_node_id_to_profiling_filler_.resize(execution_data->base_ed.node_num);
  profiler->node_id_to_profiler_node_id_.resize(execution_data->base_ed.node_num);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    const auto kernel_extend_info =
        static_cast<const KernelExtendInfo *>(execute_node->context.kernel_extend_info);
    if (kernel_extend_info == nullptr) {
      continue;
    }
    const auto compute_node_info =
        static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[i]->context.compute_node_info);
    uint64_t name_hash_id = 0;
    uint64_t node_type_hash_id = 0;
    if (compute_node_info == nullptr) {
      continue;
    } else {
      const auto &node_type = compute_node_info->GetNodeType();
      node_type_hash_id = MsprofGetHashId(node_type, strlen(node_type));
      const auto &node_name = compute_node_info->GetNodeName();
      name_hash_id = MsprofGetHashId(node_name, strlen(node_name));
    }
    if (!ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_extend_info->GetKernelType())) {
      continue;
    }
    ProfNodeAdditionInfo addition_info{};
    profiler->name_hashes_to_node_addition_infos_[name_hash_id] = std::move(addition_info);
    profiler->node_addition_infos_[execute_node->node_id][0] = &(profiler->name_hashes_to_node_addition_infos_[name_hash_id]);
    profiler->prof_extend_infos_[execute_node->node_id].node_name_idx = name_hash_id;
    profiler->prof_extend_infos_[execute_node->node_id].node_type_idx = node_type_hash_id;
    profiler->name_hash_to_node_id_.insert({name_hash_id, execute_node->node_id});
  }

  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    EXPECT_EQ(profiler->DoProf(execute_node), ge::SUCCESS);
  }
}


TEST_F(CannProfilerUT, InitContextInfo_aic_ffts_plus_task) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  auto node = compute_graph->FindNode("add1");
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  ge::AttrUtils::SetBool(op_desc, "_is_fftsplus_task", true);

  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto execution_data = reinterpret_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  profiler->prof_extend_infos_.resize(execution_data->base_ed.node_num, ProfExtendInfo{});
  profiler->node_addition_infos_.resize(execution_data->base_ed.node_num);
  profiler->exe_node_id_to_profiling_filler_.resize(execution_data->base_ed.node_num, nullptr);
  profiler->node_id_to_profiler_node_id_.resize(execution_data->base_ed.node_num);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    const auto kernel_extend_info =
        static_cast<const KernelExtendInfo *>(execute_node->context.kernel_extend_info);
    if (kernel_extend_info == nullptr) {
      continue;
    }
    const auto compute_node_info =
        static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[i]->context.compute_node_info);
    uint64_t name_hash_id = 0;
    uint64_t node_type_hash_id = 0;
    if (compute_node_info == nullptr) {
      continue;
    } else {
      const auto &node_type = compute_node_info->GetNodeType();
      node_type_hash_id = MsprofGetHashId(node_type, strlen(node_type));
      const auto &node_name = compute_node_info->GetNodeName();
      name_hash_id = MsprofGetHashId(node_name, strlen(node_name));
    }
    if (!ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_extend_info->GetKernelType())) {
      continue;
    }
    ProfNodeAdditionInfo addition_info{};
    profiler->name_hashes_to_node_addition_infos_[name_hash_id] = std::move(addition_info);
    profiler->node_addition_infos_[execute_node->node_id][0] = &(profiler->name_hashes_to_node_addition_infos_[name_hash_id]);
    profiler->prof_extend_infos_[execute_node->node_id].node_name_idx = name_hash_id;
    profiler->prof_extend_infos_[execute_node->node_id].node_type_idx = node_type_hash_id;
    profiler->name_hash_to_node_id_.insert({name_hash_id, execute_node->node_id});
  }

  const uint64_t name_hash = MsprofGetHashId(op_desc->GetNamePtr(), op_desc->GetName().length());
  auto node_iter = profiler->name_hash_to_node_id_.find(name_hash);
  EXPECT_TRUE(node_iter != profiler->name_hash_to_node_id_.end());
  const auto node_id = node_iter->second;

  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  EXPECT_EQ(profiler->prof_extend_infos_[node_id].engine_type, MSPROF_GE_TASK_TYPE_MIX_AIC);
  auto context_info = &(profiler->node_addition_infos_[node_id][0U]->context_info);
  auto context_data = reinterpret_cast<MsprofContextIdInfo *>(context_info->data);
  EXPECT_EQ(context_data->ctxIdNum, 1);
  EXPECT_EQ(context_data->ctxIds[0], 0);

  ge::AttrUtils::SetBool(op_desc, "_mix_is_aiv", true);
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
  EXPECT_EQ(profiler->prof_extend_infos_[node_id].engine_type, MSPROF_GE_TASK_TYPE_MIX_AIV);

  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    EXPECT_EQ(profiler->DoProf(execute_node), ge::SUCCESS);
  }
}

TEST_F(CannProfilerUT, davici_model_report) {
  (void)bg::ValueHolder::PushGraphFrame();
  auto graph = ShareGraph::SimpleStaticPartitionedCallGraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("StaticFoo", false).Build();
  ge::DavinciModel ge_model(0, nullptr);
  graph->TopologicalSorting();
  bg::ValueHolder::PopGraphFrame();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto execution_data = reinterpret_cast<const ExecutionData *>(executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());

  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);

  std::unordered_map<uint64_t, DfxExtendInfo *> node_name_to_dfx_extend_info = {};
  profiler->InitNameAndTypeWithHash(*execution_data);
  EXPECT_EQ(profiler->InitBasicInfoAndTensorInfo(*execution_data, graph, node_name_to_dfx_extend_info), ge::SUCCESS);
  auto context = KernelRunContextFaker()
                    .NodeName("test1")
                    .NodeType("test1")
                    .KernelType("DavinciModelExecute")
                    .KernelName("DavinciModelExecute")
                    .KernelIONum(1, 0)
                    .Build();
  context.value_holder[0].Set(reinterpret_cast<void *>(&ge_model), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *);
  Node *davinci_model_node = (Node *)malloc(size);
  memcpy(&davinci_model_node->context, context.context, sizeof(KernelRunContext) + sizeof(AsyncAnyValue *));
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    const auto kernel_context = reinterpret_cast<KernelContext *>(&execute_node->context);
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
    if (kernel_extend_info == nullptr) {
      continue;
    }
    auto kernel_type = kernel_extend_info->GetKernelType();
    if (std::string(kernel_type) == std::string("DavinciModelExecute")) {
      davinci_model_node->node_id = execute_node->node_id;
      break;
    }
  }
  GlobalProfilingWrapper::GetInstance()->IncreaseProfCount();
  EXPECT_EQ(profiler->DoProf(davinci_model_node), ge::SUCCESS);
  free(davinci_model_node);
}

TEST_F(CannProfilerUT, InitContextInfo_no_namehash) {
  const std::vector<uint32_t> input_dims {1, 2, 3, 4};
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  auto node = compute_graph->FindNode("add1");
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", input_dims);
  // add cmo context info
  for (size_t i = 0; i < CMO_IDX_KEY.size(); i++) {
    const std::string ctx_id_attr = "_data_prof_ctx_id_v" + CMO_IDX_KEY[i];
    const std::string name_attr = "_data_prof_name_v" + CMO_IDX_KEY[i];
    const std::string type_attr = "_data_prof_type" + CMO_IDX_KEY[i];
    std::vector<uint32_t> cmo_context_ids{1, 2, 3};
    std::string cmo_context_name = "write_back_123";
    uint32_t cmo_context_type = RT_CTX_TYPE_FLUSH_DATA + i;
    (void)ge::AttrUtils::SetListInt(op_desc, ctx_id_attr, cmo_context_ids);
    (void)ge::AttrUtils::SetStr(op_desc, name_attr, cmo_context_name);
    (void)ge::AttrUtils::SetInt(op_desc, type_attr, cmo_context_type);
  }

  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
}

TEST_F(CannProfilerUT, InitContextInfo_invalid_ctx_num) {
  const std::vector<uint32_t> input_dims {1, 2, 3, 4};
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  auto node = compute_graph->FindNode("add1");
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetListInt(op_desc, "_context_id_list", input_dims);
  // add cmo context info
  for (size_t i = 0; i < CMO_IDX_KEY.size(); i++) {
    const std::string ctx_id_attr = "_data_prof_ctx_id_v" + CMO_IDX_KEY[i];
    const std::string name_attr = "_data_prof_name_v" + CMO_IDX_KEY[i];
    const std::string type_attr = "_data_prof_type" + CMO_IDX_KEY[i];
    std::vector<uint32_t> cmo_context_ids(100, 1);
    std::string cmo_context_name = "write_back";
    uint32_t cmo_context_type = RT_CTX_TYPE_FLUSH_DATA + i;
    (void)ge::AttrUtils::SetListInt(op_desc, ctx_id_attr, cmo_context_ids);
    (void)ge::AttrUtils::SetStr(op_desc, name_attr, cmo_context_name);
    (void)ge::AttrUtils::SetInt(op_desc, type_attr, cmo_context_type);
  }

  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(executor, nullptr);
  auto profiler = executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  EXPECT_EQ(profiler->InitProfInfoByGraphNode(node), ge::SUCCESS);
}

static void BuildMixL2NodeGraph(ge::ComputeGraphPtr &root_graph, ge::NodePtr &node, bool single) {
  DEF_GRAPH(fused_graph) {
                           auto data_0 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
                           auto data_1 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
                           auto ret_val_0 = OP_CFG(ge::FRAMEWORKOP).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                               .Attr(ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal");

                           auto conv = OP_CFG("CONV2D_T")
                               .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                               .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                               .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
                           auto sqrt = OP_CFG("SQRT_T")
                               .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                               .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                               .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
                           CHAIN(NODE("_arg_in_0", data_0)
                                     ->EDGE(0, 0)
                                     ->NODE("conv2d", conv)
                                     ->EDGE(0, 0)
                                     ->NODE("sqrt", sqrt)
                                     ->EDGE(0, 0)
                                     ->NODE("retVal", ret_val_0));
                           CHAIN(NODE("_arg_in_1", data_1)
                                     ->EDGE(0, 1)
                                     ->NODE("conv2d", conv));
                         };
  auto origin_fused_graph = ge::ToComputeGraph(fused_graph);

  DEF_GRAPH(g1) {
                  auto data_0 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
                  auto data_1 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
                  auto fused_conv = OP_CFG("CONV2D_T")
                      .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                      .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
                  CHAIN(NODE("_arg_0", data_0)
                            ->EDGE(0, 0)
                            ->NODE("Conv2D_Sqrt", fused_conv)
                            ->EDGE(0, 0)
                            ->NODE("Node_Output", ge::NETOUTPUT));
                  CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("Conv2D_Sqrt", fused_conv));
                };
  uint32_t mem_offset = 0U;
  root_graph = ge::ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  ge::AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
  node = root_graph->FindNode("Conv2D_Sqrt");
  SetGraphOutShapeRange(root_graph);
  auto conv2d_desc = node->GetOpDesc();
  ge::AttrUtils::SetGraph(conv2d_desc, "_original_fusion_graph", origin_fused_graph);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrLowingFunc, ge::kFFTSMixL2LowerFunc);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrCalcArgsSizeFunc, ge::kFFTSMixL2CalcFunc);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);

  vector<int64_t> workspace_bytes = { 200, 300, 400};
  conv2d_desc->SetWorkspaceBytes(workspace_bytes);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = ge::MakeShared<ge::OpKernelBin>("s_mix_aictbeKernel", std::move(test_bin));
  conv2d_desc->SetExtAttr(std::string("_mix_aic") + ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  conv2d_desc->SetExtAttr(std::string("_mix_aiv") + ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  conv2d_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  std::vector<int64_t> clean_output_indexes = {0};
  (void)ge::AttrUtils::SetListInt(conv2d_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, clean_output_indexes);
  int64_t atom_max_size = 32;
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxAtomicCleanTilingSize, atom_max_size);
  std::string op_compile_info_json = R"({"_workspace_size_list":[32], "vars":{"ub_size": 12, "core_num": 2}})";
  ge::AttrUtils::SetStr(conv2d_desc, optiling::ATOMIC_COMPILE_INFO_JSON, op_compile_info_json);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");
  ge::AttrUtils::SetBool(conv2d_desc, "support_dynamicshape", true);
  ge::AttrUtils::SetInt(conv2d_desc, "op_para_size", 512);
  ge::AttrUtils::SetInt(conv2d_desc, "atomic_op_para_size", 512);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  (void)ge::AttrUtils::SetListInt(conv2d_desc, "_atomic_context_id_list", {0});
  (void)ge::AttrUtils::SetStr(conv2d_desc, "_atomic_kernelname", "test");

  std::vector<std::string> names_prefix;
  names_prefix.emplace_back("_mix_aic");
  if (!single) {
    names_prefix.emplace_back("_mix_aiv");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_mix_aic_kernel_list_first_name", "aic");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
}

static ge::graphStatus TilingTestSuccess(TilingContext *tiling_context) {
  tiling_context->SetTilingKey(0);
  tiling_context->SetBlockDim(32);
  TilingData *tiling_data = tiling_context->GetRawTilingData();
  if (tiling_data == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Tiling data is nullptr");
    return ge::GRAPH_FAILED;
  }
  auto workspaces = tiling_context->GetWorkspaceSizes(2);
  workspaces[0] = 4096;
  workspaces[1] = 6904;
  int64_t data = 100;
  tiling_data->Append<int64_t>(data);
  tiling_data->SetDataSize(sizeof(int64_t));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTest1(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, node:[%s], input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           context->GetNodeName(), input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
    GELOGD("InferShapeTest1 index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseTEST(gert::KernelContext *kernel_context) {
  (void)kernel_context;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTest2(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

TEST_F(CannProfilerUT, ProfilingForFftsPlusL2Mix) {
  ge::ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, true);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = ge::MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  uint32_t need_mode = 1U;
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kNeedModeAddr, need_mode);

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  ASSERT_NE(root_graph, nullptr);
  root_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});

  OpImplSpaceRegistryV2Array space_registry_v2_array;
  gert::SpaceRegistryFaker facker;
  space_registry_v2_array[0] = (*facker.BuildRegistryArray())[0];
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->infer_shape = InferShapeTest1;
  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->tiling = TilingTestSuccess;
  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->tiling_parse = TilingParseTEST;
  space_registry_v2_array[0]->CreateOrGetOpImpl("SQRT_T")->infer_shape = InferShapeTest2;

  auto infer_fun = [](ge::Operator &op) -> ge::graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return ge::GRAPH_SUCCESS;
  };
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("CONV2D_T", infer_fun);
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("SQRT_T", infer_fun);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(root_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "LoweringMixL2Node");

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto execution_data = reinterpret_cast<const ExecutionData *>(model_executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  auto profiler = model_executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  EXPECT_EQ(profiler->InitForCannDevice(execution_data), ge::SUCCESS);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto execute_node = execution_data->base_ed.nodes[i];
    profiler->RecordLaunchBeginTime(*execute_node);
    profiler->exe_node_id_to_profiling_filler_[i] = [] (const KernelContext *, ProfilingInfoWrapper &) {
      return ge::SUCCESS;
    };
    EXPECT_EQ(profiler->DoProf(execute_node), ge::SUCCESS);
  }

  ge::AttrUtils::SetListInt(node->GetOpDesc(), "tbe_op_atomic_dtypes", {9});
  ge::AttrUtils::SetStr(node->GetOpDesc(), "_atomic_kernelname", "test");
  ge::AttrUtils::SetStr(node->GetOpDesc(), "_atomic_cube_vector_core_type", "AIC");
  EXPECT_EQ(profiler->InitForCannDevice(execution_data), ge::SUCCESS);
}

TEST_F(CannProfilerUT, InitProfiler_AicpuHostCompute_SetEngineType) {
  auto node1 = FakeNodeHelper::FakeNode("test2", "AicpuHostCompute", 1);
  CannProfilerV2 profilerV2{nullptr};
  profilerV2.prof_extend_infos_.resize(10, ProfExtendInfo{});
  MsprofApi fake_api;
  profilerV2.InitLaunchApi(12345, "AicpuHostCompute", fake_api);
  profilerV2.SetEngineType(node1.node.node_id, "AicpuHostCompute", nullptr);
  EXPECT_EQ(profilerV2.prof_extend_infos_[node1.node.node_id].engine_type,
            static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CPU));
}

TEST_F(CannProfilerUT, InitProfiler_Aiv_SetEngineType) {
  CannProfilerV2 profiler(nullptr);
  const NodeIdentity node_id = 0;
  profiler.prof_extend_infos_.resize(10, ProfExtendInfo{});
  auto dfx_extend_info = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[100]);
  *const_cast<uint32_t *>(reinterpret_cast<DfxExtendInfo *>(dfx_extend_info.get())->GetTaskTypeAddr()) =
      static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV);
  profiler.SetEngineType(node_id, "LaunchKernelWithHandle", reinterpret_cast<DfxExtendInfo *>(dfx_extend_info.get()));
  EXPECT_EQ(profiler.prof_extend_infos_[node_id].engine_type, static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV));
}

TEST_F(CannProfilerUT, BaseExecutorProfiler_InitNameAndTypeWithHash) {
  ExecutionData execution_data{};
  std::vector<ProfExtendInfo> prof_extend_infos;
  BaseExecutorProfiler profiler;
  Node node{};
  Node *nodes[] = {&node};
  execution_data.base_ed.node_num = 1;
  execution_data.base_ed.nodes = nodes;

  // Test case 1: Success,  1 node and node.context.kernel_extend_info is null
  profiler.InitNameAndTypeWithHash(execution_data, prof_extend_infos);

  uint8_t holder1[sizeof(KernelExtendInfo)];
  node.context.kernel_extend_info = holder1;
  auto extend_info = reinterpret_cast<KernelExtendInfo *>(holder1);

  // Test case 2: Failure, 1 node and node_id is invalid
  node.node_id = 2;
  EXPECT_NE(profiler.InitNameAndTypeWithHash(execution_data, prof_extend_infos), ge::SUCCESS);

  // Test case 3: Success, 1 node and node.context.compute_node_info is null
  const char *unknown_node_name = "UnknownNodeName";
  const size_t invalid_hash_id = 0UL;
  node.node_id = 0;
  extend_info->SetKernelType("^INVALID^");
  EXPECT_EQ(profiler.InitNameAndTypeWithHash(execution_data, prof_extend_infos), ge::SUCCESS);
  EXPECT_EQ(prof_extend_infos[0].kernel_type_idx, invalid_hash_id);
  EXPECT_EQ(prof_extend_infos[0].node_type_idx, MsprofGetHashId(unknown_node_name, strlen(unknown_node_name)));
  EXPECT_EQ(prof_extend_infos[0].node_name_idx, MsprofGetHashId(unknown_node_name, strlen(unknown_node_name)));
  EXPECT_EQ(prof_extend_infos[0].origin_name_hash_for_hash, UINT64_MAX);
  prof_extend_infos.clear();

  // Test case 4: Success, 1 node and node.context.compute_node_info is not null
  const char *origin_name = "TestNode_1";
  const char *node_name = "TestNode_1_AtomicClean";
  const char *node_type = "NodeType_Test";
  const char *kernel_type = "ModelExecute";
  uint8_t holder2[sizeof(ComputeNodeInfo)];
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(holder2);
  compute_node_info->SetNodeType(node_type);
  compute_node_info->SetNodeName(node_name);
  extend_info->SetKernelType(kernel_type);
  node.context.compute_node_info = holder2;

  EXPECT_EQ(profiler.InitNameAndTypeWithHash(execution_data, prof_extend_infos), ge::SUCCESS);
  EXPECT_EQ(prof_extend_infos[0].kernel_type_idx, MsprofGetHashId(kernel_type, strlen(kernel_type)));
  EXPECT_EQ(prof_extend_infos[0].node_type_idx, MsprofGetHashId(node_type, strlen(node_type)));
  EXPECT_EQ(prof_extend_infos[0].node_name_idx, MsprofGetHashId(node_name, strlen(node_name)));
  EXPECT_EQ(prof_extend_infos[0].origin_name_hash_for_hash, MsprofGetHashId(origin_name, strlen(origin_name)));
  prof_extend_infos.clear();
}
}  // namespace gert
