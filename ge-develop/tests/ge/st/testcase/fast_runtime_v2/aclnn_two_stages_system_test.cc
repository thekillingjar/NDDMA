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

#include "faker/fake_value.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data.h"
#include "common/bg_test.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/data_flow_op_impl.h"
#include "lowering/model_converter.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "faker/magic_ops.h"
#include "check/executor_statistician.h"
#include "graph/utils/graph_dump_utils.h"
#include "dump/utils/dump_session_wrapper.h"
#include "lowering/graph_converter.h"
#include "graph/operator_reg.h"

using namespace ge;
namespace gert {
namespace {
graphStatus OpExecutePrepareDoNothing(gert::OpExecutePrepareContext *op_exe_prepare_context) {
  op_exe_prepare_context->SetWorkspaceSizes({100});
  return GRAPH_SUCCESS;
}
graphStatus OpExecuteLaunchDoNothing(gert::OpExecuteLaunchContext *) {
  return GRAPH_SUCCESS;
}

 graphStatus OpExecuteFuncDoNothing(gert::OpExecuteContext *) {
  return GRAPH_SUCCESS;
}

graphStatus InferShapeForAdd(gert::InferShapeContext *context) {
  GELOGD("InferShapeForAdd");
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    auto min_num = std::min(input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    if (min_num != 1) {
      GELOGE(PARAM_INVALID,
             "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
             input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    } else {
      if (input_shape_1.GetDimNum() > 1) {
        *output_shape = input_shape_1;
      } else {
        *output_shape = input_shape_0;
      }
      return GRAPH_SUCCESS;
    }
  }
  if (input_shape_0.GetDimNum() == 0) {
    *output_shape = input_shape_1;
    return GRAPH_SUCCESS;
  }
  if (input_shape_1.GetDimNum() == 0) {
    *output_shape = input_shape_0;
    return GRAPH_SUCCESS;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }

  return GRAPH_SUCCESS;
}

struct AddCompileInfo {
  int64_t a;
  int64_t b;
};
graphStatus TilingForAdd(gert::TilingContext *context) {
  GELOGD("TilingForAdd");
  auto ci = context->GetCompileInfo<AddCompileInfo>();
  GE_ASSERT_NOTNULL(ci);
  auto tiling_data = context->GetRawTilingData();
  GE_ASSERT_NOTNULL(tiling_data);
  tiling_data->Append(*ci);
  tiling_data->Append(ci->a);
  return GRAPH_SUCCESS;
}

graphStatus TilingParseForAdd(gert::KernelContext *) {
  return GRAPH_SUCCESS;
}

static void *CreateCompileInfo() {
  return new AddCompileInfo();
}

static void DeleteCompileInfo(void *const obj) {
  delete reinterpret_cast<AddCompileInfo *>(obj);
}
}
class AclnnTwoStagesSt : public testing::Test {
 public:
  void SetUp() override {
    auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    auto op_impl_func = space_registry->CreateOrGetOpImpl("Add");
    op_impl_func->infer_shape = InferShapeForAdd;
    op_impl_func->tiling = TilingForAdd;
    op_impl_func->tiling_parse = TilingParseForAdd;
    op_impl_func->compile_info_creator = CreateCompileInfo;
    op_impl_func->compile_info_deleter = DeleteCompileInfo;
    op_impl_func->op_execute_prepare_func = OpExecutePrepareDoNothing;
    op_impl_func->op_execute_launch_func = OpExecuteLaunchDoNothing;
    op_impl_func->op_execute_func = nullptr;

    auto op_impl_func_mul = space_registry->CreateOrGetOpImpl("Mul");
    op_impl_func_mul->infer_shape = InferShapeForAdd;
    op_impl_func_mul->tiling = TilingForAdd;
    op_impl_func_mul->tiling_parse = TilingParseForAdd;
    op_impl_func_mul->compile_info_creator = CreateCompileInfo;
    op_impl_func_mul->compile_info_deleter = DeleteCompileInfo;
    op_impl_func_mul->op_execute_func = OpExecuteFuncDoNothing;

    default_space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
  }

  void TearDown() override {
    execute_op_prepare_call_times.store(0UL);
    execute_op_launch_call_times.store(0UL);
    execute_op_func_call_times.store(0UL);
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(default_space_registry);
  }

 public:
  static std::atomic_size_t execute_op_prepare_call_times;
  static std::atomic_size_t execute_op_launch_call_times;
  static std::atomic_size_t execute_op_func_call_times;
  OpImplSpaceRegistryV2Ptr default_space_registry;
};
std::atomic_size_t AclnnTwoStagesSt::execute_op_prepare_call_times(0UL);
std::atomic_size_t AclnnTwoStagesSt::execute_op_launch_call_times(0UL);
std::atomic_size_t AclnnTwoStagesSt::execute_op_func_call_times(0UL);
const std::string DynamicAtomicStubName = "DynamicAtomicBin";

namespace ge{
REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)
REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
        .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                               DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
        .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                               DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
        .OP_END_FACTORY_REG(Mul)
}

namespace {
UINT32 StubExecutePrepareFuncSuccEmpty(gert::OpExecutePrepareContext *context) {
  context->SetWorkspaceSizes({100});
  AclnnTwoStagesSt::execute_op_prepare_call_times.fetch_add(1);
  return GRAPH_SUCCESS;
}

UINT32 StubExecuteLaunchFuncSuccEmpty(gert::OpExecuteLaunchContext *context) {
  (void)context;
  AclnnTwoStagesSt::execute_op_launch_call_times.fetch_add(1);
  return GRAPH_SUCCESS;
}

UINT32 StubExecuteFuncSuccEmpty(gert::OpExecuteContext *context) {
  (void)context;
  AclnnTwoStagesSt::execute_op_func_call_times.fetch_add(1);
  return GRAPH_SUCCESS;
}

graphStatus FakeFindExecuteFunc(KernelContext *context) {
  auto op_execute_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::OpExecuteFunc>(0);
  *op_execute_fun_ptr = StubExecuteFuncSuccEmpty;
  return GRAPH_SUCCESS;
}

graphStatus FakeFind2StageFuncs(KernelContext *context) {
  auto op_prepare_fun_ptr = context->GetOutputPointer<OpImplRegisterV2::OpExecPrepareFunc>(0);
  auto op_launch_fun_ptr = context->GetOutputPointer<OpImplRegisterV2::OpExecLaunchFunc>(1);
  *op_prepare_fun_ptr = StubExecutePrepareFuncSuccEmpty;
  *op_launch_fun_ptr = StubExecuteLaunchFuncSuccEmpty;
  return GRAPH_SUCCESS;
}

graphStatus FakeExecuteOpPrepare(KernelContext *context) {
  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.release());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 0;
  sizes->MutableData()[1] = 128;
  auto workspace_size_chain = context->GetOutput(static_cast<size_t>(OpExecutePrepareOutputIndex::kWorkspaceSize));
  workspace_size_chain->SetWithDefaultDeleter<uint8_t[]>(sizes);
  return GRAPH_SUCCESS;
}

void SetNoStorage(const OpDescPtr &op_desc, Format format, DataType dt,
                  std::initializer_list<int64_t> shape) {
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(format);
    op_desc->MutableInputDesc(i)->SetOriginFormat(format);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetDataType(dt);
    op_desc->MutableInputDesc(i)->SetOriginDataType(dt);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(format);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(format);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetDataType(dt);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(dt);
  }
}

void SetShapeRangeNoStorage(const OpDescPtr &op_desc, std::initializer_list<int64_t> min_shape,
                            std::initializer_list<int64_t> max_shape) {
  std::vector<std::pair<int64_t, int64_t>> range;
  for (size_t i = 0; i < min_shape.size(); ++i) {
    range.emplace_back(min_shape.begin()[i], max_shape.begin()[i]);
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetOriginShapeRange(range);
    op_desc->MutableInputDesc(i)->SetShapeRange(range);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetOriginShapeRange(range);
    op_desc->MutableOutputDesc(i)->SetShapeRange(range);
  }
}

/*
 *            NetOutput
 *                |
 *              mul1
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
ComputeGraphPtr BuildAddMulNodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->EDGE(0, 1)->NODE("mul1", "Mul")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("mul1", "Mul"));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  SetNoStorage(data1->GetOpDesc(), FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data1->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  SetNoStorage(data2->GetOpDesc(), FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(data2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  data2->GetOpDesc()->MutableAllInputName() = {{"x", 0}};

  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(add1->GetOpDesc(), FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(add1->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(add1->GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  AddCompileResult(add1, false);
  add1->GetOpDesc()->SetWorkspaceBytes({2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  auto add2 = graph->FindNode("mul1");
  add2->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x2", 1}};
  add2->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  SetNoStorage(add2->GetOpDesc(), FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1});
  SetShapeRangeNoStorage(add2->GetOpDesc(), {1, 1, 1, 1}, {-1, -1, -1, -1});
  AttrUtils::SetInt(add2->GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 1);
  AddCompileResult(add2, false);
  add2->GetOpDesc()->SetWorkspaceBytes({2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"mul1"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}
}  // namespace

TEST_F(AclnnTwoStagesSt, MultiThreadExecutor_Ok_OnlyTwoStagesAclnnOps) {
  constexpr int32_t kEnvNoOverwrite = 0;
  int32_t mmRet = 0;
  MM_SYS_SET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, "3", kEnvNoOverwrite, mmRet);
  (void) mmRet;

  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  auto add1 = graph->FindNode("add1");
  (void)AttrUtils::SetStr(add1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  auto add2 = graph->FindNode("add2");
  (void)AttrUtils::SetStr(add2->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto exe_graph_nodes = exe_graph->GetAllNodes();
  size_t connected_free_memory_count = 0U;
  for (const auto &node : exe_graph_nodes) {
    if (node->GetType() == "ExecuteOpLaunch") {
      auto out_data_nodes = node->GetAllOutNodes();
      for (const auto &out_node : out_data_nodes) {
        if (out_node->GetType() == "FreeMemory") {
          connected_free_memory_count++;
        }
      }
    }
  }
  ASSERT_EQ(connected_free_memory_count, 0U);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindOpExe2PhaseFunc", FakeFind2StageFuncs);

  TaskProducerFactory::GetInstance().SetProducerType(TaskProducerType::KERNEL);
  ASSERT_EQ(TaskProducerFactory::GetInstance().GetProducerType(), TaskProducerType::KERNEL);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  auto model_executor = ModelV2Executor::Create(exe_graph, option, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto multi_thread_ed = reinterpret_cast<const MultiThreadExecutionData *>(
      model_executor->GetExeGraphExecutor(SubExeGraphType::kMainExeGraph)->GetExecutionData());
  multi_thread_ed->scheduler->DumpBrief();
  ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  // 第一次执行，无缓存，全部算子调用tiling_func
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            GRAPH_SUCCESS);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_prepare_call_times, 2UL);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_launch_call_times, 2UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpPrepare"), 2);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpLaunch"), 2);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "BuildDualStageAclnnOpFwkData"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "FreeMemory"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "FreeMemoryHoldAddr"), 2);

  multi_thread_ed->scheduler->Dump();
  runtime_stub.Clear();
  rtStreamDestroy(stream);

  MM_SYS_UNSET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, mmRet);
  (void) mmRet;
}

TEST_F(AclnnTwoStagesSt, PriorityTopologicalExecute_Ok_OnlyTwoStagesAclnnOps) {
  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  auto add1 = graph->FindNode("add1");
  (void)AttrUtils::SetStr(add1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  auto add2 = graph->FindNode("add2");
  (void)AttrUtils::SetStr(add2->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindOpExe2PhaseFunc", FakeFind2StageFuncs);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            GRAPH_SUCCESS);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_prepare_call_times, 2UL);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_launch_call_times, 2UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpPrepare"), 2);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpLaunch"), 2);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "BuildDualStageAclnnOpFwkData"), 0);

  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(AclnnTwoStagesSt, MultiThreadExecutor_Ok_HybridAclnnOps) {
  auto graph = BuildAddMulNodeGraph();
  auto add1 = graph->FindNode("add1");
  (void)AttrUtils::SetStr(add1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  auto mul1 = graph->FindNode("mul1");
  (void)AttrUtils::SetStr(mul1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("Mul", AiCoreTaskDefFaker("MulStubBin").WithHandle())
                           .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindOpExe2PhaseFunc", FakeFind2StageFuncs);
  runtime_stub.GetKernelStub().SetUp("FindOpExeFunc", FakeFindExecuteFunc);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            GRAPH_SUCCESS);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_prepare_call_times, 1UL);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_launch_call_times, 1UL);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_func_call_times, 1UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpPrepare"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpLaunch"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Mul", "ExecuteOpFunc"), 1);  //
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "BuildDualStageAclnnOpFwkData"), 0);

  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(AclnnTwoStagesSt, PriorityTopologicalExecute_Ok_HybridAclnnOps) {
  auto graph = BuildAddMulNodeGraph();
  auto add1 = graph->FindNode("add1");
  (void)AttrUtils::SetStr(add1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  auto mul1 = graph->FindNode("mul1");
  (void)AttrUtils::SetStr(mul1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .AddTaskDef("Mul", AiCoreTaskDefFaker("MulStubBin").WithHandle())
      .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindOpExe2PhaseFunc", FakeFind2StageFuncs);
  runtime_stub.GetKernelStub().SetUp("FindOpExeFunc", FakeFindExecuteFunc);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            GRAPH_SUCCESS);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_prepare_call_times, 1UL);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_launch_call_times, 1UL);
  EXPECT_EQ(AclnnTwoStagesSt::execute_op_func_call_times, 1UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpPrepare"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ExecuteOpLaunch"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Mul", "ExecuteOpFunc"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "BuildDualStageAclnnOpFwkData"), 0);

  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(AclnnTwoStagesSt, PriorityTopologicalExecute_Ok_OnlyTwoStagesAclnnOps_WsZeroSize) {
  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  auto add1 = graph->FindNode("add1");
  (void)AttrUtils::SetStr(add1->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  auto add2 = graph->FindNode("add2");
  (void)AttrUtils::SetStr(add2->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindOpExe2PhaseFunc", FakeFind2StageFuncs);
  runtime_stub.GetKernelStub().SetUp("ExecuteOpPrepare", FakeExecuteOpPrepare);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            GRAPH_SUCCESS);

  runtime_stub.Clear();
  rtStreamDestroy(stream);
}
} // gert