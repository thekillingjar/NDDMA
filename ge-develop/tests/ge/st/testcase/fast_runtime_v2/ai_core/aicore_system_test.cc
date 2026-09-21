/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "lowering/model_converter.h"
#include "graph/operator_factory_impl.h"
#include "lowering/graph_converter.h"
#include "exe_graph/runtime/tiling_context.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "engine/aicore/fe_rt2_common.h"
#include "runtime/model_v2_executor.h"
#include "framework/common/ge_types.h"
#include "exe_graph/runtime/kernel_context.h"
#include <kernel/memory/mem_block.h>
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "engine/node_converter_utils.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "register/op_tiling_registry.h"
#include "graph/manager/mem_manager.h"
#include "graph/utils/graph_dump_utils.h"
// stub and faker
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/magic_ops.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "op_tiling/op_tiling_constants.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/graph_frame.h"
#include "macro_utils/dt_public_scope.h"
#include "common/dump/exception_dumper.h"
#include "register/kernel_registry.h"
#include "macro_utils/dt_public_unscope.h"
namespace gert {
using namespace ge;

namespace {

class AutofuseTestGraphBuilder {
 public:
  AutofuseTestGraphBuilder() = default;
  AutofuseTestGraphBuilder SetBinPath(const std::string &input_bin_path) {
    bin_path = input_bin_path;
    return *this;
  }
  AutofuseTestGraphBuilder SetAllSymbolNum(const int64_t input_all_sym_num) {
    all_sym_num = input_all_sym_num;
    return *this;
  }
  AutofuseTestGraphBuilder SetInferShapeCacheKey(const std::vector<std::string> &input_infer_shape_cache_key) {
    infer_shape_cache_key = input_infer_shape_cache_key;
    return *this;
  }
  ge::ComputeGraphPtr Build() {
    auto graph = ShareGraph::AutoFuseNodeGraph();
    graph->TopologicalSorting();
    auto fused_graph_node = graph->FindNode("fused_graph");
    auto fused_graph_node1 = graph->FindNode("fused_graph1");

    auto op_desc = fused_graph_node->GetOpDesc();
    (void)ge::AttrUtils::SetStr(op_desc, ge::kAttrLowingFunc, "kAutoFuseLoweringFunc");
    auto op_desc1 = fused_graph_node1->GetOpDesc();
    (void)ge::AttrUtils::SetStr(op_desc1, ge::kAttrLowingFunc, "kAutoFuseLoweringFunc");

    if (!bin_path.empty()) {
      (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", bin_path);
      (void)ge::AttrUtils::SetStr(fused_graph_node1->GetOpDesc(), "bin_file_path", bin_path);
    }
    if (all_sym_num >=0) {
      (void)ge::AttrUtils::SetInt(graph, "_all_symbol_num", all_sym_num);
    }
    if (infer_shape_cache_key.size() == 2) {
      (void)ge::AttrUtils::SetStr(
          fused_graph_node->GetOpDesc(), "_symbol_infer_shape_cache_key", infer_shape_cache_key[0]);
      (void)ge::AttrUtils::SetStr(
          fused_graph_node1->GetOpDesc(), "_symbol_infer_shape_cache_key", infer_shape_cache_key[1]);
    }
    return graph;
  }
 private:
  std::string bin_path;
  int64_t all_sym_num = -1;
  std::vector<std::string> infer_shape_cache_key;
};
struct FakeArgsInfo {
  ::domi::ArgsInfo::ArgsType arg_type;
  ::domi::ArgsInfo::ArgsFormat arg_fmt;
  int32_t statrt_index;
  uint32_t arg_num;
};

void GlobalDataWithArgsInfo(const NodePtr &aicore_node, LoweringGlobalData &origin_global_data,
                                          std::vector<std::vector<FakeArgsInfo>> &args_infos) {
  LoweringGlobalData::NodeCompileResult aicore_node_compile_result;
  auto task_defs = origin_global_data.FindCompiledResult(aicore_node)->GetTaskDefs();
  if (task_defs.size() !=args_infos.size()) {
    GELOGE(ge::PARAM_INVALID,
           "task_defs.size()[%zu] must be equal to args_infos.size()[%zu], return origin global data instead.",
           task_defs.size(), args_infos.size());
    return;
  }
  for (size_t i = 0U; i < task_defs.size(); ++i) {
    auto task_def = task_defs[i];
    auto kernel_def = task_def.mutable_kernel();
    for (size_t idx = 0U; idx < args_infos[i].size(); ++idx) {
      auto args_info_ = kernel_def->add_args_info();
      args_info_->set_arg_type(args_infos[i][idx].arg_type);
      args_info_->set_arg_format(args_infos[i][idx].arg_fmt);
      args_info_->set_start_index(args_infos[i][idx].statrt_index);
      args_info_->set_size(args_infos[i][idx].arg_num);
    }
    aicore_node_compile_result.task_defs.push_back(task_def);
  }
  origin_global_data.AddCompiledResult(aicore_node, aicore_node_compile_result);
  return;
}

ge::graphStatus TilingAddFail(TilingContext *tiling_context) {
  return ge::GRAPH_FAILED;
}
ge::graphStatus TilingAddSuccess(TilingContext *tiling_context) {
  return ge::GRAPH_SUCCESS;
}
struct V4CompileInfo : public optiling::CompileInfoBase {
  int64_t a;
  int64_t b;
  int64_t c;
};
bool V4TilingFail(const ge::Operator &, const optiling::CompileInfoPtr, optiling::OpRunInfoV2 &) {
  return false;
}
bool V4TilingSuccess(const ge::Operator &, const optiling::CompileInfoPtr, optiling::OpRunInfoV2 &) {
  return true;
}
void *V3TilingParse(const ge::Operator &, const ge::AscendString &) {
  static V4CompileInfo compile_info;
  return static_cast<void *>(&compile_info);
}
bool V3TilingFail(const ge::Operator &, const void *, optiling::OpRunInfoV2 &) {
  return false;
}
bool V3TilingSuccess(const ge::Operator &, const void *, optiling::OpRunInfoV2 &) {
  return true;
}
optiling::CompileInfoPtr V4TilingParse(const ge::Operator &, const ge::AscendString &) {
  return std::make_shared<V4CompileInfo>();
}
ge::ComputeGraphPtr BuildFallibleTilingNodeComputeGraph(bool is_mix = false) {
  auto graph = ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1}, {-1, -1, 3, 4}},
                                                {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  GE_ASSERT_NOTNULL(graph);
  graph->TopologicalSorting();
  auto add_node = graph->FindFirstNodeMatchType("Add");
  std::vector<std::string> dfx_opts = {kOpDfxPrintf, kOpDfxAssert};
  (void)ge::AttrUtils::SetListStr(add_node->GetOpDesc(), gert::kOpDfxOptions, dfx_opts);
  (void)ge::AttrUtils::SetInt(add_node->GetOpDesc(), gert::kOpDfxBufferSize, 12345);
  std::vector<std::vector<int64_t>> dyn_in_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{0}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  GE_ASSERT_TRUE(ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true));
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel"));
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "HelloWorld"));
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "testst");
  if (is_mix) {
    (void)ge::AttrUtils::SetInt(graph, ge::ATTR_MODEL_NOTIFY_NUM, 2);
    (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
    std::vector<uint32_t> core_num_v = {15, 8, 7};
    (void)ge::AttrUtils::SetListInt(add_node->GetOpDesc(), kMixCoreNumVec, core_num_v);
    add_node->GetOpDesc()->SetAttachedStreamId(1);
    std::vector<int32_t> notify_id_v = {0, 1};
    (void)ge::AttrUtils::SetListInt(add_node->GetOpDesc(), ge::RECV_ATTR_NOTIFY_ID, notify_id_v);
  }
  return graph;
}

ge::graphStatus InferShapeForAddEmpty(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeForAddEmpty %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    if (i == 1) {
      GELOGD("InferShapeForAddEmpty 1");
      output_shape->SetDim(i, 0);
      continue;
    }
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForAdd(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeForAdd %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Add).InferShape(InferShapeForAdd);

ge::ComputeGraphPtr BuildInferShapeNode(bool sup_empty, bool unknown_dim) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  GE_ASSERT_NOTNULL(graph);
  graph->TopologicalSorting();
  auto add_node = graph->FindFirstNodeMatchType("Add");
  std::vector<std::vector<int64_t>> dyn_in_vv = {{0,1}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  GE_ASSERT_NOTNULL(graph);
  if (sup_empty) {
    auto data1 = graph->FindNode("data1");
    std::vector<std::pair<int64_t, int64_t>> in_shape_range = {{1, -1}, {1, -1}, {3, 3}, {4, 4}};
    data1->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(in_shape_range);
    data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShapeRange(in_shape_range);
    auto data2 = graph->FindNode("data2");
    data2->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(in_shape_range);
    data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShapeRange(in_shape_range);
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {0, 100}, {3, 3}, {4, 4}};
    for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
      if (unknown_dim) {
        output_tensor->SetShape(GeShape({-2}));
      } else {
        output_tensor->SetShapeRange(shape_range);
      }
    }
  }
  return graph;
}

size_t GetNodeCountWithGraphStage(const ge::ExecuteGraphPtr &exe_graph, const std::string &node_type,
                                  const std::string &stage = "ALL") {
  std::vector<FastNode *> nodes;
  if (stage == "ALL") {
    nodes = exe_graph->GetAllNodes();
  } else {
    for (const auto &node : exe_graph->GetDirectNode()) {
      if (node->GetType() == stage) {
        auto sub_graph = ge::FastNodeUtils::GetSubgraphFromNode(node, 0U);
        nodes = sub_graph->GetAllNodes();
      }
    }
  }
  size_t cnt = 0;
  for (auto node : nodes) {
    if (node->GetType() == node_type) {
      cnt++;
    }
  }
  return cnt;
}
}  // namespace

enum class TilingType { kV3Tiling, kV4Tiling, kGert2Tiling };

class AICoreLoweringST : public testing::Test {
  public:
   KernelRegistry &registry = KernelRegistry::GetInstance();
  bg::GraphFrame *root_frame = nullptr;
  void SetUp() {
    ge::MemManager::Instance().Initialize({0, RT_MEMORY_HBM});
  }
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
 public:
  void TestFallibleTiling(bool rollback, TilingType tiling_type, bool rollback_fail = false, bool is_mix = false) {
    auto graph = BuildFallibleTilingNodeComputeGraph(is_mix);
    bool add_aicpu = !rollback_fail;

    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, add_aicpu).Build(2);
    ASSERT_NE(graph, nullptr);
    auto add = graph->FindNode("add1");
    ASSERT_NE(add, nullptr);
    auto op_desc = add->GetOpDesc();
    ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
    GertRuntimeStub stub;
    // todo 当前没有Op的打桩方法，暂时强改注册中心里面的Tiling函数了
    //    auto tiling_fun = OpImplRegistry::GetInstance().CreateOrGetOpImpl("Add").tiling;
    auto op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
        DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Add"));
    auto tiling_fun = op_impl->tiling;
    if (tiling_type == TilingType::kV4Tiling) {
      op_impl->tiling = nullptr;
      if (rollback) {
        optiling::OpTilingFuncRegistry tiling_register("Add", V4TilingFail, V4TilingParse);
      } else {
        optiling::OpTilingFuncRegistry tiling_register("Add", V4TilingSuccess, V4TilingParse);
      }
    } else if (tiling_type == TilingType::kV3Tiling) {
      op_impl->tiling = nullptr;
      if (rollback) {
        optiling::OpTilingFuncRegistry tiling_register("Add", V3TilingFail, V3TilingParse);
      } else {
        optiling::OpTilingFuncRegistry tiling_register("Add", V3TilingSuccess, V3TilingParse);
      }
    } else if (tiling_type == TilingType::kGert2Tiling) {
      if (rollback) {
        op_impl->tiling = TilingAddFail;
      } else {
        op_impl->tiling = TilingAddSuccess;
      }
    }

    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    model_desc_holder.MutableModelDesc().SetReusableStreamNum(2);
    model_desc_holder.MutableModelDesc().SetReusableNotifyNum(2);
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "AICoreLoweringSTGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto inputs = FakeTensors({1, 2, 3, 4}, 2);
    auto outputs = FakeTensors({1, 2, 3, 4}, 1);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    if (rollback_fail) {
      ASSERT_NE(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                        outputs.GetTensorList(), outputs.size()),
                ge::GRAPH_SUCCESS);
    } else {
      ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                        outputs.GetTensorList(), outputs.size()),
                ge::GRAPH_SUCCESS);
    }
    ess->PrintExecutionSummary();
    if (rollback) {
      if (rollback_fail) {
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "ReportRollbackError"), 1);
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithHandle"), 0);
      } else {
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AicpuLaunchTfKernel"), 1);
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithHandle"), 0);
      }
    } else {
      if (is_mix) {
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchMixKernelWithHandle"), 1);
      } else {
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AicpuLaunchTfKernel"), 0);
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithHandle"), 1);
      }
    }

    Shape expect_out_shape{1, 2, 3, 4};
    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

    op_impl->tiling = tiling_fun;
    auto registry = const_cast<std::unordered_map<std::string, optiling::OpTilingFuncInfo> *>(
        &optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo());
    registry->erase("Add");

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  void TestFallibleTiling1(bool rollback, TilingType tiling_type, bool is_mix = false) {
    auto graph = BuildFallibleTilingNodeComputeGraph(is_mix);
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false, true).Build(2);
    ASSERT_NE(graph, nullptr);
    auto add = graph->FindNode("add1");
    ASSERT_NE(add, nullptr);
    auto op_desc = add->GetOpDesc();
    ge::AttrUtils::SetBool(op_desc, "_memcheck", true);

    GertRuntimeStub stub;
    // todo 当前没有Op的打桩方法，暂时强改注册中心里面的Tiling函数了
    //    auto tiling_fun = OpImplRegistry::GetInstance().CreateOrGetOpImpl("Add").tiling;
    auto op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
        DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Add"));
    auto tiling_fun = op_impl->tiling;
    if (tiling_type == TilingType::kV4Tiling) {
      op_impl->tiling = nullptr;
      if (rollback) {
        optiling::OpTilingFuncRegistry tiling_register("Add", V4TilingFail, V4TilingParse);
      } else {
        optiling::OpTilingFuncRegistry tiling_register("Add", V4TilingSuccess, V4TilingParse);
      }
    } else if (tiling_type == TilingType::kV3Tiling) {
      op_impl->tiling = nullptr;
      if (rollback) {
        optiling::OpTilingFuncRegistry tiling_register("Add", V3TilingFail, V3TilingParse);
      } else {
        optiling::OpTilingFuncRegistry tiling_register("Add", V3TilingSuccess, V3TilingParse);
      }
    } else if (tiling_type == TilingType::kGert2Tiling) {
      if (rollback) {
        op_impl->tiling = TilingAddFail;
      } else {
        op_impl->tiling = TilingAddSuccess;
      }
    }

    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    model_desc_holder.MutableModelDesc().SetReusableStreamNum(2);
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "AICoreLoweringSTGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto inputs = FakeTensors({1, 2, 3, 4}, 2);
    auto outputs = FakeTensors({1, 2, 3, 4}, 1);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();
    if (rollback) {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AicpuLaunchTfKernel"), 1);
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 0);
    } else {
      if (is_mix) {
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchMixKernelWithFlag"), 1);
      } else {
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AicpuLaunchTfKernel"), 0);
        EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 1);
      }
    }

    Shape expect_out_shape{1, 2, 3, 4};
    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

    op_impl->tiling = tiling_fun;
    auto registry = const_cast<std::unordered_map<std::string, optiling::OpTilingFuncInfo> *>(
        &optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo());
    registry->erase("Add");

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  void TestInferShapeEmptyTensor(bool sup_empty, bool unknown_dim) {
    auto graph = BuildInferShapeNode(sup_empty, unknown_dim);
    auto data1 = graph->FindNode("data1");
    std::vector<std::pair<int64_t, int64_t>> range;
    data1->GetOpDesc()->MutableInputDesc(0)->GetShapeRange(range);
    std::cout << range.size() << std::endl;
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
    ASSERT_NE(graph, nullptr);
    auto op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
        DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Add"));
    auto infer_func = op_impl->infer_shape;
    if (sup_empty) {
      op_impl->infer_shape = InferShapeForAddEmpty;
    }
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "AICoreLoweringSTGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({2048}, 2);
    FakeTensors outputs = FakeTensors({2048}, 1);
    if (sup_empty) {
      inputs = FakeTensors({1, 0, 3, 4}, 2);
      outputs = FakeTensors({1, 0, 3, 4}, 1);
      for (size_t i = 0UL; i < inputs.size(); ++i) {
        const auto model_input_desc = model_executor->GetModelDesc().GetInputDesc(i);
        EXPECT_TRUE(model_input_desc->IsShapeInRange(inputs.GetTensorList()[i]->GetStorageShape()));
        EXPECT_TRUE(model_input_desc->IsOriginShapeInRange(inputs.GetTensorList()[i]->GetOriginShape()));
      }
    }

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();
    if (sup_empty) {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 0);
    } else {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 1);
    }
    Shape expect_out_shape;
    if (sup_empty) {
      expect_out_shape = {1, 0, 3, 4};
      //      OpImplRegistry::GetInstance().CreateOrGetOpImpl("Add").infer_shape = infer_func;
      op_impl->infer_shape = infer_func;
    } else {
      expect_out_shape = {2048};
    }

    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void TestNodeWithOptAndDynamic1() {
    auto graph = BuildFallibleTilingNodeComputeGraph(false);
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, false).Build(2);
    ASSERT_NE(graph, nullptr);

    auto add_node = graph->FindFirstNodeMatchType("Add");
    add_node->GetInDataAnchor(1)->UnlinkAll();

    std::vector<std::vector<int64_t>> dyn_in_vv = {{0}};
    (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
    std::vector<std::vector<int64_t>> dyn_out_vv = {{0}};
    (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
    (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
    (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);

    std::vector<FakeArgsInfo> args_info{
        {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
        {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
        {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
    std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
    GlobalDataWithArgsInfo(add_node, global_data, args_infos);

    // todo 当前没有Op的打桩方法，暂时强改注册中心里面的Tiling函数了
    //    auto tiling_fun = OpImplRegistry::GetInstance().CreateOrGetOpImpl("Add").tiling;
    auto op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
        DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Add"));
    ASSERT_NE(op_impl, nullptr);
    auto tiling_fun = op_impl->tiling;
    op_impl->tiling = TilingAddSuccess;
    auto infer_func = op_impl->infer_shape;
    op_impl->infer_shape = InferShapeForAdd;
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    model_desc_holder.MutableModelDesc().SetReusableStreamNum(2);
    model_desc_holder.MutableModelDesc().SetReusableNotifyNum(2);
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "AICoreLoweringSTGraph");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto inputs = FakeTensors({1, 2, 3, 4}, 2);
    auto outputs = FakeTensors({1, 2, 3, 4}, 1);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithHandle"), 1);

    Shape expect_out_shape{1, 2, 3, 4};
    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

    op_impl->tiling = tiling_fun;
    op_impl->infer_shape = infer_func;
    auto registry = const_cast<std::unordered_map<std::string, optiling::OpTilingFuncInfo> *>(
        &optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo());
    registry->erase("Add");

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
};

TEST_F(AICoreLoweringST, FallibleTiling_RollbackToAiCpu_WhenTilingFailed) {
  TestFallibleTiling(true, TilingType::kGert2Tiling);
  TestFallibleTiling(true, TilingType::kV3Tiling);
  TestFallibleTiling(true, TilingType::kV4Tiling);
  TestFallibleTiling(true, TilingType::kGert2Tiling, true);
}
TEST_F(AICoreLoweringST, FallibleTiling_LaunchAiCore_WhenTilingSuccess) {
  TestFallibleTiling(false, TilingType::kGert2Tiling);
  TestFallibleTiling(false, TilingType::kV3Tiling);
  TestFallibleTiling(false, TilingType::kV4Tiling);
}
TEST_F(AICoreLoweringST, FallibleTiling_RollbackToAiCpu_WhenTilingFailed1) {
  TestFallibleTiling1(true, TilingType::kGert2Tiling);
  TestFallibleTiling(true, TilingType::kV3Tiling);
  TestFallibleTiling(true, TilingType::kV4Tiling);
}
TEST_F(AICoreLoweringST, FallibleTiling_LaunchAiCore_WhenTilingSuccess1) {
  TestFallibleTiling1(false, TilingType::kGert2Tiling);
  TestFallibleTiling(false, TilingType::kV3Tiling);
  TestFallibleTiling(false, TilingType::kV4Tiling);
}

TEST_F(AICoreLoweringST, LaunchAiCore_EmptyTensor) {
  TestInferShapeEmptyTensor(true, false);
  TestInferShapeEmptyTensor(true, true);
  TestInferShapeEmptyTensor(false, false);
}

TEST_F(AICoreLoweringST, TestOptAndDynamic) {
  TestNodeWithOptAndDynamic1();
}

TEST_F(AICoreLoweringST, TestAutofuseNodeBase) {
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  // 1、构图
  std::string cmake_binary_dir = CMAKE_BINARY_DIR;
  std::string autofuse_stub_so_path = cmake_binary_dir + "/tests/depends/op_stub/libautofuse_stub.so";

  auto graph = AutofuseTestGraphBuilder().SetBinPath(autofuse_stub_so_path).SetAllSymbolNum(16).Build();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();

  // 3、图执行器
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "AutofuseSTGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({2, 3}, 4);
  FakeTensors outputs = FakeTensors({5, 6, 4}, 2);
  // 4、图执行
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(),
                                    inputs.size(),outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "LaunchKernelWithHandle"), 2);
  Shape expect_out_shape{5, 6, 4};
  auto output_shape0 = outputs.GetTensorList()[0]->GetShape().GetStorageShape();
  ASSERT_EQ(output_shape0.GetDimNum(), expect_out_shape.GetDimNum());
  for (size_t i = 0UL; i < output_shape0.GetDimNum(); i++) {
    EXPECT_EQ(output_shape0.GetDim(i), expect_out_shape.GetDim(i));
  }
  auto output_shape1 = outputs.GetTensorList()[1]->GetShape().GetStorageShape();
  ASSERT_EQ(output_shape1.GetDimNum(), expect_out_shape.GetDimNum());
  for (size_t i = 0UL; i < output_shape1.GetDimNum(); i++) {
    EXPECT_EQ(output_shape1.GetDim(i), expect_out_shape.GetDim(i));
  }
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(AICoreLoweringST, TestAutofuseNodeWithTilingCacheAndInferMerge) {
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  std::string cmake_binary_dir = CMAKE_BINARY_DIR;
  std::string autofuse_stub_so_path = cmake_binary_dir + "/tests/depends/op_stub/libautofuse_stub.so";
  // 构图
  auto graph = AutofuseTestGraphBuilder()
                   .SetBinPath(autofuse_stub_so_path)
                   .SetAllSymbolNum(16)
                   .SetInferShapeCacheKey({"xxxx", "xxxx"})
                   .Build();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();

  // 图执行器
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "AutofuseSTGraph");
  ASSERT_EQ(GetNodeCountWithGraphStage(exe_graph, "PrepareCacheableTilingFwkData", "Init"), 2);
  ASSERT_EQ(GetNodeCountWithGraphStage(exe_graph, "GetSymbolTilingCacheKey", "Main"), 2);

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  // 图执行
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({2, 3}, 4);
  FakeTensors outputs = FakeTensors({5, 6, 4}, 2);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  // 第一次迭代，添加Tiling缓存
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Tiling cache status: missed"), -1);
  ASSERT_EQ(stub.GetSlogStub().FindLog(DLOG_INFO, "Tiling cache status: hit"), -1);
  // 第二次迭代
  stub.GetSlogStub().Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Tiling cache status: hit"), -1);
  // 打开Tiling缓存不在构造SymbolTiling节点，改为构造CacheableSymbolTiling
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "Tiling"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "CacheableTiling"), 4);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "GetSymbolTilingCacheKey"), 4);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "InferShape"), 2);
  rtStreamDestroy(stream);
  stub.GetSlogStub().SetLevel(DLOG_ERROR);
  stub.GetSlogStub().Clear();
}

TEST_F(AICoreLoweringST, TestAutofuseNodeTilingParse) {
  // 1、构图
  std::string cmake_binary_dir = CMAKE_BINARY_DIR;
  std::string autofuse_stub_so_path = cmake_binary_dir + "/tests/depends/op_stub/libautofuse_stub.so";

  auto graph = AutofuseTestGraphBuilder().SetBinPath(autofuse_stub_so_path).SetAllSymbolNum(16).Build();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();

  // 3、图执行器
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "AutofuseSTGraph");
  // 校验TilingParse节点
  ASSERT_EQ(GetNodeCountWithGraphStage(exe_graph, "SymbolTilingParse", "Init"), 2);

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({2, 3}, 4);
  FakeTensors outputs = FakeTensors({5, 6, 4}, 2);
  // 4、图执行
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(),
                                    inputs.size(),outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "LaunchKernelWithHandle"), 2);
  Shape expect_out_shape{5, 6, 4};
  auto output_shape0 = outputs.GetTensorList()[0]->GetShape().GetStorageShape();
  ASSERT_EQ(output_shape0.GetDimNum(), expect_out_shape.GetDimNum());
  for (size_t i = 0UL; i < output_shape0.GetDimNum(); i++) {
    EXPECT_EQ(output_shape0.GetDim(i), expect_out_shape.GetDim(i));
  }
  auto output_shape1 = outputs.GetTensorList()[1]->GetShape().GetStorageShape();
  ASSERT_EQ(output_shape1.GetDimNum(), expect_out_shape.GetDimNum());
  for (size_t i = 0UL; i < output_shape1.GetDimNum(); i++) {
    EXPECT_EQ(output_shape1.GetDim(i), expect_out_shape.GetDim(i));
  }
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(AICoreLoweringST, TestAutofuseNodeDlopenAutofuseSoNodeCheck) {
  // 1、构图
  std::string cmake_binary_dir = CMAKE_BINARY_DIR;
  std::string autofuse_stub_so_path = cmake_binary_dir + "/tests/depends/op_stub/libautofuse_stub.so";

  auto graph = AutofuseTestGraphBuilder().SetBinPath(autofuse_stub_so_path).SetAllSymbolNum(16).Build();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();

  // 3、图执行器
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "AutofuseSTGraph");
  // 校验GetAutofuseFuncs节点
  ASSERT_EQ(GetNodeCountWithGraphStage(exe_graph, "GetAutofuseFuncs", "Init"), 2);

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({2, 3}, 4);
  FakeTensors outputs = FakeTensors({5, 6, 4}, 2);
  // 4、图执行
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(),
                                    inputs.size(),outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("AscBackend", "LaunchKernelWithHandle"), 2);
  Shape expect_out_shape{5, 6, 4};
  auto output_shape0 = outputs.GetTensorList()[0]->GetShape().GetStorageShape();
  ASSERT_EQ(output_shape0.GetDimNum(), expect_out_shape.GetDimNum());
  for (size_t i = 0UL; i < output_shape0.GetDimNum(); i++) {
    EXPECT_EQ(output_shape0.GetDim(i), expect_out_shape.GetDim(i));
  }
  auto output_shape1 = outputs.GetTensorList()[1]->GetShape().GetStorageShape();
  ASSERT_EQ(output_shape1.GetDimNum(), expect_out_shape.GetDimNum());
  for (size_t i = 0UL; i < output_shape1.GetDimNum(); i++) {
    EXPECT_EQ(output_shape1.GetDim(i), expect_out_shape.GetDim(i));
  }
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(AICoreLoweringST, AutofuseInferKernelTraceTest) {
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  std::string cmake_binary_dir = CMAKE_BINARY_DIR;
  std::string autofuse_stub_so_path = cmake_binary_dir + "/tests/depends/op_stub/libautofuse_stub.so";

  auto graph = AutofuseTestGraphBuilder().SetBinPath(autofuse_stub_so_path).SetAllSymbolNum(16).Build();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();

  // 3、图执行器
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "AutofuseSTGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({2, 3}, 4);
  FakeTensors outputs = FakeTensors({5, 6, 4}, 2);
  // 4、图执行
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(),
                                    inputs.size(),outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Symbolic infos: s0: 2, s1: 3, s2: 2."), -1);
  stub.GetSlogStub().SetLevel(DLOG_ERROR);
  stub.GetSlogStub().Clear();
}
// todo test launch with flag
}  // namespace gert