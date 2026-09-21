/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/registry/op_impl_space_registry_v2.h"

#include "runtime/model_v2_executor.h"
#include "framework/common/ge_types.h"
#include "engine/aicore/fe_rt2_common.h"
#include "exe_graph/runtime/kernel_context.h"
#include <gtest/gtest.h>
#include "lowering/model_converter.h"
#include "exe_graph/runtime/tiling_context.h"
#include "lowering/graph_converter.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_registry.h"
// stub and faker
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "common/sgt_slice_type.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/magic_ops.h"
#include "faker/kernel_run_context_facker.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph_builder/bg_tiling.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "framework/executor/ge_executor.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "graph/operator_factory_impl.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "engine/aicore/kernel/mixl2_update_kernel.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;
using namespace ge;

namespace gert {
namespace kernel {
}
using namespace kernel;
namespace {
void *StubCreateCompileInfo() {
  return nullptr;
}

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
ge::graphStatus TilingParseTEST(gert::KernelContext *kernel_context) {
  (void)kernel_context;
  GELOGD("TilingParseTEST");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeTest1(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeTest1 %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
    GELOGD("InferShapeTest1 index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
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
}  // namespace

class MixL2LoweringST : public testing::Test {
  void SetUp() override {
  }
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
  void TestMixL2SingleLowering(GeRootModelPtr &root_model, LoweringGlobalData &global_data, bool expect) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    if (expect) {
      ASSERT_NE(exe_graph, nullptr);
    } else {
      ASSERT_EQ(exe_graph, nullptr);
      return;
    }
    ge::DumpGraph(exe_graph.get(), "LoweringMixL2Node");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 2);
    FakeTensors outputs = FakeTensors({4, 4, 4, 4}, 1);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "FFTSUpdateMixL2Args"), 1);

    Shape expect_out_shape = {4, 4, 4, 4};
    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

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
 *      onv2d_sqrt                                          \      /
 *          |                                                 conv2d
 *          |                                                   |
 *          |                                                 sqrt
 *      NetOutput                                               |
 *                                                        NetOutput
 *
 *
 **********************************************************************************************************************/
static void BuildMixL2NodeGraph(ComputeGraphPtr &root_graph, ge::NodePtr &node, bool single) {

  DEF_GRAPH(fused_graph) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto ret_val_0 = OP_CFG(FRAMEWORKOP).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Attr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal");

    auto conv = OP_CFG("CONV2D_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto sqrt = OP_CFG("SQRT_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
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
  auto origin_fused_graph = ToComputeGraph(fused_graph);
  std::vector<int64_t> unknown_shape = {4,4,-1,4};
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto fused_conv = OP_CFG("CONV2D_T")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(2)
        .OutCnt(1)
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
        .Build("Conv2D_Sqrt");
    CHAIN(NODE("_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(fused_conv)
              ->EDGE(0, 0)
              ->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE(fused_conv));
  };
  uint32_t mem_offset = 0U;
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
  node = root_graph->FindNode("Conv2D_Sqrt");
  auto conv2d_desc = node->GetOpDesc();
  AttrUtils::SetGraph(conv2d_desc, "_original_fusion_graph", origin_fused_graph);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrLowingFunc, ge::kFFTSMixL2LowerFunc);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrCalcArgsSizeFunc, ge::kFFTSMixL2CalcFunc);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);
  conv2d_desc->MutableInputDesc(0)->SetShape(GeShape(unknown_shape));
  vector<int64_t> workspace_bytes = { 200, 300, 400};
  conv2d_desc->SetWorkspaceBytes(workspace_bytes);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("s_mix_aictbeKernel", std::move(test_bin));
  conv2d_desc->SetExtAttr(std::string("_mix_aic") + OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  conv2d_desc->SetExtAttr(std::string("_mix_aiv") + OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIV");
  AttrUtils::SetBool(conv2d_desc, "support_dynamicshape", true);
  AttrUtils::SetInt(conv2d_desc, "op_para_size", 512);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);

  std::vector<std::string> names_prefix;
  names_prefix.emplace_back("_mix_aic");
  if (!single) {
    names_prefix.emplace_back("_mix_aiv");
    AttrUtils::SetStr(node->GetOpDesc(), "_mix_aic_kernel_list_first_name", "aic");
    AttrUtils::SetStr(node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
}

TEST_F(MixL2LoweringST, lowering_output_shapes_empty_mix_l2_2_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);

  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->infer_shape = InferShapeTest1;
  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->tiling = TilingTestSuccess;
  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->tiling_parse = TilingParseTEST;
  space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->compile_info_creator = StubCreateCompileInfo;
  space_registry_v2_array[0]->CreateOrGetOpImpl("SQRT_T")->infer_shape = InferShapeTest2;

  auto infer_fun = [](Operator &op) -> graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferShapeFunc("CONV2D_T", infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("SQRT_T", infer_fun);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);

  auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range = {{0, -1}, {0, -1}, {0, 3}, {0, 4}};
  //out_desc->SetShapeRange(in_shape_range);
  out_desc->SetDataType(DT_FLOAT);
  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  TestMixL2SingleLowering(root_model, global_data, true);
}
}  // namespace gert
