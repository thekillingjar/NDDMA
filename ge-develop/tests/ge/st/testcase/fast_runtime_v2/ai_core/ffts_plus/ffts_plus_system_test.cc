/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/model_v2_executor.h"
#include "framework/common/ge_types.h"
#include "exe_graph/runtime/kernel_context.h"
#include "engine/aicore/fe_rt2_common.h"
#include <gtest/gtest.h>
#include "lowering/model_converter.h"
#include "exe_graph/runtime/tiling_context.h"
#include "lowering/graph_converter.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_registry.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
// stub and faker
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "common/sgt_slice_type.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/magic_ops.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph_builder/bg_tiling.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "framework/executor/ge_executor.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "faker/space_registry_faker.h"
#include "common/global_variables/diagnose_switch.h"
#include "graph/load/model_manager/davinci_model.h"
#include "runtime/subscriber/global_dumper.h"

#include "macro_utils/dt_public_scope.h"
#include "common/dump/exception_dumper.h"
#include "subscriber/dumper/executor_dumper.h"
#include "register/kernel_registry.h"
#include "macro_utils/dt_public_unscope.h"

#include "../aicore_exe_graph_check.h"
using namespace std;
using namespace testing;
using namespace ge;

namespace gert {
namespace {

}  // namespace

ge::graphStatus TilingTestSuccess(TilingContext *tiling_context) {
  tiling_context->SetTilingKey(0);
  tiling_context->SetBlockDim(32);
  TilingData *tiling_data = tiling_context->GetRawTilingData();
  GELOGD("TilingTestSuccess tiling_data:%p.", tiling_data);
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

ge::graphStatus InferShapeTest1(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, node:[%s], input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           context->GetNodeName(), input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    return ge::GRAPH_FAILED;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeTest1 %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
    GELOGD("InferShapeTest1 index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferEmptyShape(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, node:[%s], input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           context->GetNodeName(), input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    return ge::GRAPH_FAILED;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferEmptyShape %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, 0);
    GELOGD("InferEmptyShape index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeFake(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeFake %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
    GELOGD("InferShapeFake index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseTEST(gert::KernelContext *kernel_context) {
  (void)kernel_context;
  GELOGD("TilingParseTEST");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeTest2(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeTest2 %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
    GELOGD("InferShapeTest2 index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

// 本用例需要特殊适配，不是仅仅load default space registry就可以的
class FFTSLoweringST : public testing::Test {

  void SetUp() override {
  }
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
 public:
  void TestFFTSSingleLowering(GeRootModelPtr &root_model, LoweringGlobalData &global_data, bool expect,
                              std::vector<JudgeInfo> &judge_v) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    if (expect) {
      ASSERT_NE(exe_graph, nullptr);
    } else {
      ASSERT_EQ(exe_graph, nullptr);
      return;
    }
    for (auto &judge_info : judge_v) {
      ASSERT_EQ(Judge2KernelPriority(exe_graph, judge_info), true);
    }
    ge::DumpGraph(exe_graph.get(), "LoweringSingleFFTSGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 2);
    auto output = TensorFaker().Shape({4,4,4,4}).DataType(ge::DT_INT64).Build();
    std::vector<Tensor *> outputs{output.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "FFTSNodeThread"), 1);

    Shape expect_out_shape = {4, 4, 4, 4};
    EXPECT_EQ(outputs.data
              ()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void TestFFTSLowering(GeRootModelPtr &root_model, LoweringGlobalData &global_data, const bool dump_flag, bool is_empty) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "LoweringFFTSGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 3);
    auto output = TensorFaker().Shape({4,4,4,4}).DataType(ge::DT_INT64).Build();
    std::vector<Tensor *> outputs{output.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "FFTSNodeThread"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("REDUCE_T", "AICoreUpdateContext"), 1);
    if (is_empty) {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("ADD_T", "SetNodeSkipCtx"), 1);
      Shape expect_out_shape = {0, 0, 0, 0};
      EXPECT_EQ(outputs.data()[0]->GetShape().GetStorageShape(), expect_out_shape);
    } else {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("ADD_T", "SetNodeSkipCtx"), 0);
      Shape expect_out_shape = {4, 4, 4, 4};
      EXPECT_EQ(outputs.data()[0]->GetShape().GetStorageShape(), expect_out_shape);
    }

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void TestFFTSStaticLowering(GeRootModelPtr &root_model, LoweringGlobalData &global_data) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "LoweringFFTSManualStaticGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({6, 4, 4, 4}, 2);
    auto output = TensorFaker().Shape({3,4,4,4}).DataType(ge::DT_INT64).Build();
    std::vector<Tensor *> outputs{output.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "StaManualUpdateContext"), 1);

    Shape expect_out_shape = {3, 4, 4, 4};
    EXPECT_EQ(outputs.data()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void TestFFTSAutoStaticLowering(GeRootModelPtr &root_model, LoweringGlobalData &global_data) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "LoweringFFTSAutoStaticGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({6, 4, 4, 4}, 2);
    auto output = TensorFaker().Shape({3,4,4,4}).DataType(ge::DT_INT64).Build();
    std::vector<Tensor *> outputs{output.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "StaAutoUpdateContext"), 1);

    Shape expect_out_shape = {3, 4, 4, 4};
    EXPECT_EQ(outputs.data()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void TestFFTSByMultiThreadExecutor(GeRootModelPtr &root_model, LoweringGlobalData &global_data,
                                     TaskProducerType type) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "LoweringFFTSGraph");

    TaskProducerFactory::GetInstance().SetProducerType(type);
    ASSERT_EQ(TaskProducerFactory::GetInstance().GetProducerType(), type);
    auto option = MultiThreadExecutorOption(kLeastThreadNumber);
    auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);
    ASSERT_NE(model_executor, nullptr);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 3);
    auto output = TensorFaker().Shape({4, 4, 4, 4}).DataType(ge::DT_INT64).Build();
    std::vector<Tensor *> outputs{output.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(), outputs.data(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    Shape expect_out_shape = {4, 4, 4, 4};
    EXPECT_EQ(outputs.data()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
};

/***********************************************************************************************************************
 *
 * Data           Data
 *   \             /
 *    \           /
 *     \         /                                       Data      Data
 *      \       /                                          \        /
 *   PartitionedCall                                        \      /
 *          |                                                 Add
 *          |                                                   |
 *          |                                                   |
 *      NetOutput                                               |
 *                                                        NetOutput
 *
 *
 **********************************************************************************************************************/
static void BuildFftsPlusSingleOpGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_plus_graph,
    TBEKernelStore *kernel_store = nullptr) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
  };
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);

  AttrUtils::SetStr(root_graph->FindNode("PartitionedCall_0")->GetOpDesc(), ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH, "ffts_plus");
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);

  DEF_GRAPH(g2) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto conv_0 = OP_CFG("CONV2D_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("sgt_graph/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT));
    CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Conv2D", conv_0));
  };
  ffts_plus_graph = ToComputeGraph(g2);
  SetGraphOutShapeRange(ffts_plus_graph);
  auto conv2d_desc = ffts_plus_graph->FindNode("sgt_graph/Conv2D")->GetOpDesc();
  AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_1")->GetOpDesc(), "index", 1);
  conv2d_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
  (void)ge::AttrUtils::SetBool(conv2d_desc, kUnknownShapeFromFe, true);
  std::vector<int32_t> unknown_axis = {0};
  ge::AttrUtils::SetListInt(conv2d_desc, "unknown_axis_index", unknown_axis);
  int64_t param_k = 0;
  int64_t param_b = 0;
  (void)ge::AttrUtils::SetInt(conv2d_desc, "param_k", param_k);
  (void)ge::AttrUtils::SetInt(conv2d_desc, "param_b", param_b);

  std::vector<uint32_t> input_tensor_indexes;
  std::vector<uint32_t> output_tensor_indexes;
  (void)ge::AttrUtils::SetListInt(conv2d_desc, ge::kInputTensorIndexs, input_tensor_indexes);
  (void)ge::AttrUtils::SetListInt(conv2d_desc, ge::kOutputTensorIndexs, output_tensor_indexes);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  conv2d_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  conv2d_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  AttrUtils::SetBool(conv2d_desc, "_kernel_list_first_name", true);

  ffts_plus_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(ffts_plus_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
}

TEST_F(FFTSLoweringST, ffts_plus_lowering_single_fail) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusSingleOpGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_LABEL));
  (void)ctx_def_0.mutable_label_ctx();

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto conv_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  auto part_node = root_graph->FindNode("PartitionedCall_0");
  (void)ge::AttrUtils::SetStr(part_node->GetOpDesc(), ge::kAttrLowingFunc, ge::kFFTSGraphLowerFunc);
  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(part_node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  conv_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameDvpp);
  std::vector<JudgeInfo> judge_v;
  TestFFTSSingleLowering(root_model, global_data, false, judge_v);
}

TEST_F(FFTSLoweringST, ffts_plus_memory_alloc) {
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_TRUE(level_1_allocator != nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 4U);
  ASSERT_TRUE(level_2_allocator != nullptr);
  auto span1 = level_2_allocator->Malloc(1024U);
  ASSERT_TRUE(span1 != nullptr);
  span1->Free();
  level_2_allocator->Recycle();
  ASSERT_EQ(0, level_2_allocator->GetOccupiedSpanCount());
}
}  // namespace gert
