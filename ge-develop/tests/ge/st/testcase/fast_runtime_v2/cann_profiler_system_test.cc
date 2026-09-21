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
#define private public
#include "base/registry/op_impl_space_registry_v2.h"
#undef private
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/fake_value.h"
#include "faker/ge_model_builder.h"
#include "stub/gert_runtime_stub.h"
#include "lowering/model_converter.h"
#include "aicpu_task_struct.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "subscriber/profiler/cann_host_profiler.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "common/global_variables/diagnose_switch.h"
#include "register/op_tiling_registry.h"
#include "check/executor_statistician.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_dump_utils.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "faker/model_desc_holder_faker.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "graph_builder/bg_tiling.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph/operator_factory_impl.h"

#include "macro_utils/dt_public_scope.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "common/sgt_slice_type.h"
#include "lowering/exe_graph_serializer.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
class CannProfilerST : public bg::BgTest {
  void SetUp() override {
  }
 public:
  static void BuildExecutorInner(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    auto graph = ShareGraph::BuildSingleNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
    ASSERT_NE(model_executor, nullptr);
  }

  static void BuildExecutorInner2(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    auto graph = ShareGraph::BuildTwoAddNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).
        AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
    ASSERT_NE(model_executor, nullptr);
  }

  static void BuildFftsPlusGraph(ge::ComputeGraphPtr &root_graph, ge::ComputeGraphPtr &ffts_plus_graph) {
    uint32_t mem_offset = 0U;
    DEF_GRAPH(g1) {
                    CHAIN(NODE("_arg_0", ge::DATA)->NODE("PartitionedCall_0", ge::PARTITIONEDCALL)->NODE("Node_Output", ge::NETOUTPUT));
                    CHAIN(NODE("_arg_1", ge::DATA)->NODE("PartitionedCall_0"));
                    CHAIN(NODE("_arg_2", ge::DATA)->NODE("PartitionedCall_0"));
                    CHAIN(NODE("shape", ge::CONSTANT)->NODE("PartitionedCall_0"));
                  };
    root_graph = ge::ToComputeGraph(g1);
    root_graph->SetGraphUnknownFlag(true);
    SetUnknownOpKernel(root_graph, mem_offset, true);

    ge::AttrUtils::SetStr(root_graph->FindNode("PartitionedCall_0")->GetOpDesc(), ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH, "ffts_plus");
    auto data_0_desc = root_graph->FindNode("_arg_0")->GetOpDesc();
    ge::AttrUtils::SetInt(data_0_desc, "index", 0);
    ge::AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(root_graph->FindNode("_arg_2")->GetOpDesc(), "index", 2);
    ge::AttrUtils::SetInt(root_graph->FindNode("shape")->GetOpDesc(), "index", 3);

    auto shape_const = root_graph->FindNode("shape");
    shape_const->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
    shape_const->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape({4}));
    shape_const->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(ge::GeShape({4}));
    shape_const->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
    shape_const->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
    ge::TensorUtils::SetSize(*shape_const->GetOpDesc()->MutableOutputDesc(0), 32);
    ge::GeTensor x_reshape_const_tensor(shape_const->GetOpDesc()->GetOutputDesc(0));
    std::vector<int64_t> x_reshape_const_data({4, 4, 4, 4});
    x_reshape_const_tensor.SetData(reinterpret_cast<uint8_t *>(x_reshape_const_data.data()),
                                   x_reshape_const_data.size() * sizeof(int64_t));
    ge::AttrUtils::SetTensor(shape_const->GetOpDesc(), "value", x_reshape_const_tensor);

    DEF_GRAPH(g2) {
                    auto data_0 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
                    auto data_1 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
                    auto data_2 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 2);
                    auto data_3 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 3);
                    auto conv_0 = OP_CFG("CONV2D_T")
                        .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                        .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeAIV)
                        .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
                    auto add_0 = OP_CFG("ADD_T").Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                        .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeAIV);
                    auto reduce_0 = OP_CFG("REDUCE_T").Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                        .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixAIV);
                    auto relu_0 = OP_CFG("RELU_T").Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                        .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeAIV);
                    auto reshape_0 = OP_CFG("Reshape").Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE");
                    auto identity_0 = OP_CFG("Identity").Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_RTS_FFTS_PLUS_OP_STORE");

                    CHAIN(NODE("sgt_graph/_arg_0", data_0)
                              ->EDGE(0, 0)
                              ->NODE("sgt_graph/Conv2D", conv_0)
                              ->EDGE(0, 0)
                              ->NODE("sgt_graph/Add", add_0)
                              ->EDGE(0, 0)
                              ->NODE("sgt_graph/ReduceMean", reduce_0)
                              ->EDGE(0, 0)
                              ->NODE("sgt_graph/Relu", relu_0)
                              ->EDGE(0, 0)
                              ->NODE("sgt_graph/identity", identity_0)
                              ->EDGE(0, 0)
                              ->NODE("sgt_graph/Node_Output", ge::NETOUTPUT));
                    CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Conv2D", conv_0));
                    CHAIN(NODE("sgt_graph/_arg_2", data_2)->EDGE(0, 0)->NODE("sgt_graph/Reshape", reshape_0));
                    CHAIN(NODE("sgt_graph/_arg_3", data_3)->EDGE(0, 1)->NODE("sgt_graph/Reshape", reshape_0));
                    CHAIN(NODE("sgt_graph/Reshape", reshape_0)->EDGE(0, 1)->NODE("sgt_graph/Add", add_0));
                  };
    ffts_plus_graph = ge::ToComputeGraph(g2);
    auto conv2d_desc = ffts_plus_graph->FindNode("sgt_graph/Conv2D")->GetOpDesc();
    auto add_desc = ffts_plus_graph->FindNode("sgt_graph/Add")->GetOpDesc();
    auto reduce_desc = ffts_plus_graph->FindNode("sgt_graph/ReduceMean")->GetOpDesc();
    auto relu_desc = ffts_plus_graph->FindNode("sgt_graph/Relu")->GetOpDesc();
    auto reshape_desc = ffts_plus_graph->FindNode("sgt_graph/Reshape")->GetOpDesc();
    auto identity_desc = ffts_plus_graph->FindNode("sgt_graph/identity")->GetOpDesc();
    conv2d_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
    add_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
    reduce_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
    relu_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
    (void)ge::AttrUtils::SetBool(conv2d_desc, kUnknownShapeFromFe, true);
    (void)ge::AttrUtils::SetBool(add_desc, kUnknownShapeFromFe, true);
    (void)ge::AttrUtils::SetBool(reduce_desc, kUnknownShapeFromFe, true);
    (void)ge::AttrUtils::SetBool(relu_desc, kUnknownShapeFromFe, true);
    ge::AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_0")->GetOpDesc(), "index", 0);
    ge::AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_1")->GetOpDesc(), "index", 1);
    ge::AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_2")->GetOpDesc(), "index", 2);
    ge::AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_3")->GetOpDesc(), "index", 3);

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
    (void)ge::AttrUtils::SetListInt(add_desc, ge::kInputTensorIndexs, input_tensor_indexes);
    (void)ge::AttrUtils::SetListInt(add_desc, ge::kOutputTensorIndexs, output_tensor_indexes);
    (void)ge::AttrUtils::SetListInt(reduce_desc, ge::kInputTensorIndexs, input_tensor_indexes);
    (void)ge::AttrUtils::SetListInt(reduce_desc, ge::kOutputTensorIndexs, output_tensor_indexes);
    (void)ge::AttrUtils::SetListInt(relu_desc, ge::kInputTensorIndexs, input_tensor_indexes);
    (void)ge::AttrUtils::SetListInt(relu_desc, ge::kOutputTensorIndexs, output_tensor_indexes);
    (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);
    (void)ge::AttrUtils::SetInt(add_desc, bg::kMaxTilingSize, 50);
    (void)ge::AttrUtils::SetInt(reduce_desc, bg::kMaxTilingSize, 50);
    (void)ge::AttrUtils::SetInt(relu_desc, bg::kMaxTilingSize, 50);

    string compile_info_key = "compile_info_key";
    string compile_info_json = "{\"_workspace_size_list\":[]}";
    std::vector<char> test_bin(64, '\0');
    ge::TBEKernelPtr test_kernel = ge::MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
    (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
    (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
    conv2d_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
    (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
    ge::AttrUtils::SetBool(conv2d_desc, "_kernel_list_first_name", true);
    (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
    (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
    add_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
    (void)ge::AttrUtils::SetStr(add_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");

    std::vector<std::string> names_prefix;
    names_prefix.emplace_back("_mix_aic");
    names_prefix.emplace_back("_mix_aiv");
    (void)ge::AttrUtils::SetListStr(reduce_desc, ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
    (void)ge::AttrUtils::SetStr(reduce_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
    (void)ge::AttrUtils::SetStr(reduce_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
    reduce_desc->SetExtAttr(std::string("_mix_aic") + ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
    reduce_desc->SetExtAttr(std::string("_mix_aiv") + ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
    (void)ge::AttrUtils::SetStr(reduce_desc, ge::TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");

    // force relu sink static bin
    // AttrUtils::SetBool(add_desc, "_kernel_list_first_name", true);
    (void)ge::AttrUtils::SetStr(relu_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
    (void)ge::AttrUtils::SetStr(relu_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
    relu_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
    (void)ge::AttrUtils::SetStr(relu_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    ge::AttrUtils::SetBool(relu_desc, "_kernel_list_first_name", true);

    ffts_plus_graph->SetGraphUnknownFlag(true);
    SetUnknownOpKernel(ffts_plus_graph, mem_offset);

    reshape_desc->SetOpEngineName("DNN_VM_GE_LOCAL");
    reshape_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    reshape_desc->AppendIrInput("x", ge::kIrInputRequired);
    reshape_desc->AppendIrInput("shape", ge::kIrInputRequired);
    reshape_desc->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
    reshape_desc->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT64);
    reshape_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape({-1, -1, 4, 4}));
    reshape_desc->MutableOutputDesc(0)->SetShape(ge::GeShape({-1, -1, 4, 4}));
    auto &name_index = reshape_desc->MutableAllInputName();
    name_index.clear();
    name_index["x"] = 0;
    name_index["shape"] = 1;


    identity_desc->SetOpEngineName("DNN_VM_RTS_FFTS_PLUS");
    identity_desc->SetOpKernelLibName("DNN_VM_RTS_FFTS_PLUS_OP_STORE");
    identity_desc->AppendIrInput("x", ge::kIrInputRequired);
    identity_desc->MutableOutputDesc(0)->SetDataType(ge::DT_INT64);
    identity_desc->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT64);
    identity_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape({4, 4, 4, 4}));
    identity_desc->MutableOutputDesc(0)->SetShape(ge::GeShape({4, 4, 4, 4}));

    AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
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

  static void BuildFftsPlusGraphAndTaskDef(ge::GeRootModelPtr &root_model, LoweringGlobalData &global_data) {
    ge::ComputeGraphPtr ffts_plus_graph;
    ge::ComputeGraphPtr root_graph;
    BuildFftsPlusGraph(root_graph, ffts_plus_graph);
    // Build FftsTaskDef.
    std::shared_ptr<domi::ModelTaskDef> model_task_def = ge::MakeShared<domi::ModelTaskDef>();
    auto &task_def = *model_task_def->add_task();
    InitFftsplusTaskDef(ffts_plus_graph, task_def);
    auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
    auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
    std::vector<uint32_t> ctx_id_vec;
    InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
    auto node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
    ctx_id_vec.push_back(ffts_plus_task_def.ffts_plus_ctx_size() - 1);
    (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kContextIdList, ctx_id_vec);
    ctx_id_vec.clear();
    auto &aic1_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
    InitFftsPlusAicCtxDef(ffts_plus_graph, aic1_ctx_def, "sgt_graph/Add");
    node = ffts_plus_graph->FindNode("sgt_graph/Add");
    ctx_id_vec.push_back(ffts_plus_task_def.ffts_plus_ctx_size() - 1);
    (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kContextIdList, ctx_id_vec);
    ctx_id_vec.clear();
    auto &aic2_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
    InitFftsPlusAicCtxDef(ffts_plus_graph, aic2_ctx_def, "sgt_graph/ReduceMean");
    aic2_ctx_def.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
    node = ffts_plus_graph->FindNode("sgt_graph/ReduceMean");
    uint32_t need_mode = 1U;
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kNeedModeAddr, need_mode);
    std::string json_str =
        "{\"dependencies\":[],\"thread_scopeId\":200,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
        "0,"
        "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
        "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
        "\"oriOutputTensorShape\":["
        "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{\"splitCutIndex\":1,\"reduceCutIndex\":2,\"cutId\":3}],"
        "\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
        "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
        "\"input_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}]],"
        "[[{\"lower\":9, \"higher\":10},{\"lower\":11, \"higher\":12}]]],"
        "\"output_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}]],"
        "[[{\"lower\":9, \"higher\":10},{\"lower\":11, \"higher\":12}]]],"
        "\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
        "],\"inputCutList\":[], \"outputCutList\":[]}";
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ffts::kAttrSgtJsonInfo, json_str);
    ctx_id_vec.push_back(ffts_plus_task_def.ffts_plus_ctx_size() - 1);
    (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kContextIdList, ctx_id_vec);
    ctx_id_vec.clear();
    auto &aic3_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
    InitFftsPlusAicCtxDef(ffts_plus_graph, aic3_ctx_def, "sgt_graph/Relu");
    node = ffts_plus_graph->FindNode("sgt_graph/Relu");
    ctx_id_vec.push_back(ffts_plus_task_def.ffts_plus_ctx_size() - 1);
    (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kContextIdList, ctx_id_vec);
    ctx_id_vec.clear();
    SetGraphOutShapeRange(ffts_plus_graph);
    node = ffts_plus_graph->FindNode("sgt_graph/identity");
    EXPECT_NE(node, nullptr);
    for (size_t i = 0Ul; i < 4; ++i) {
      auto &sdma_ctx = *ffts_plus_task_def.add_ffts_plus_ctx();
      InitSdmaDef(ffts_plus_graph, sdma_ctx, "sgt_graph/identity");
      ctx_id_vec.push_back(sdma_ctx.context_id());
    }
    (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kContextIdList, ctx_id_vec);

    auto &label_def = *ffts_plus_task_def.add_ffts_plus_ctx();
    label_def.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_LABEL));
    std::vector<int32_t> all_ctx_id_vec = {7};
    (void)ge::AttrUtils::SetListInt(ffts_plus_graph, "_all_ctx_id_list", all_ctx_id_vec);
    domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
    ffts_plus_sqe->set_ready_context_num(8);
    ffts_plus_sqe->set_total_context_num(8);

    auto part_node = root_graph->FindNode("PartitionedCall_0");
    (void)ge::AttrUtils::SetStr(part_node->GetOpDesc(), ge::kAttrLowingFunc, ge::kFFTSGraphLowerFunc);
    root_model = GeModelBuilder(root_graph).BuildGeRootModel();
    global_data = GlobalDataFaker(root_model).Build();
    global_data.AddCompiledResult(part_node, {{task_def}});
    OpImplSpaceRegistryV2Array space_registry_v2_array;
    space_registry_v2_array[0] = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    global_data.SetSpaceRegistriesV2(space_registry_v2_array);
    space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->infer_shape = InferShapeTest1;
    space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->tiling = TilingTestSuccess;
    space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T")->tiling_parse = TilingParseTEST;
    space_registry_v2_array[0]->CreateOrGetOpImpl("ADD_T")->infer_shape = InferShapeTest1;
    space_registry_v2_array[0]->CreateOrGetOpImpl("ADD_T")->tiling = TilingTestSuccess;
    space_registry_v2_array[0]->CreateOrGetOpImpl("ADD_T")->tiling_parse = TilingParseTEST;
    space_registry_v2_array[0]->CreateOrGetOpImpl("REDUCE_T")->infer_shape = InferShapeTest2;
    space_registry_v2_array[0]->CreateOrGetOpImpl("REDUCE_T")->tiling = TilingTestSuccess;
    space_registry_v2_array[0]->CreateOrGetOpImpl("REDUCE_T")->tiling_parse = TilingParseTEST;
    space_registry_v2_array[0]->CreateOrGetOpImpl("RELU_T")->infer_shape = InferShapeTest2;
    space_registry_v2_array[0]->CreateOrGetOpImpl("RELU_T")->tiling = TilingTestSuccess;
    space_registry_v2_array[0]->CreateOrGetOpImpl("RELU_T")->tiling_parse = TilingParseTEST;
    space_registry_v2_array[0]->CreateOrGetOpImpl("Reshape")->infer_shape = InferShapeTest1;
  }

  static void TestFFTSLowering(ge::GeRootModelPtr &root_model, LoweringGlobalData &global_data) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 3);
    auto output = TensorFaker().Shape({4, 4, 4, 4}).DataType(ge::DT_INT64).Build();
    std::vector<Tensor *> outputs{output.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    bg::BufferPool buffer_pool;
    bg::BufferPool dfx_extend_infos;
    size_t buffer_size;
    const auto compute_graph =
      exe_graph->TryGetExtAttr("_compute_graph", std::make_shared<ge::ComputeGraph>("Invalid"));

    const auto &nodes = compute_graph->GetAllNodes();
    for (size_t i = 0UL; i < nodes.size() - 1; ++i) {
      const auto &node = nodes.at(i);
      const auto &op_desc = node->GetOpDesc();
      (void)ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "1");
      (void)ge::AttrUtils::SetInt(op_desc, "_ffts_prof_ctx_type", 1);
      std::vector<uint32_t> ctx_ids;
      ctx_ids.push_back(0);
      (void)ge::AttrUtils::SetListInt(op_desc, "_context_id_list", ctx_ids);
      (void)ge::AttrUtils::SetInt(op_desc, "_op_impl_mode_enum", 0x40);
      std::vector<uint32_t> cmo_ctx_ids;
      cmo_ctx_ids.push_back(0);
      (void)ge::AttrUtils::SetListInt(op_desc, "_data_prof_ctx_id_vprefetch_idx", cmo_ctx_ids);
      (void)ge::AttrUtils::SetStr(op_desc, "_data_prof_name_vprefetch_idx", "test");
      (void)ge::AttrUtils::SetInt(op_desc, "_data_prof_typeprefetch_idx", 1);
      (void)ge::AttrUtils::SetListInt(op_desc, "_data_prof_ctx_id_vinvalidate_idx", cmo_ctx_ids);
      (void)ge::AttrUtils::SetStr(op_desc, "_data_prof_name_vinvalidate_idx", "test_inval");
      (void)ge::AttrUtils::SetInt(op_desc, "_data_prof_typeinvalidate_idx", 1);
      (void)ge::AttrUtils::SetListInt(op_desc, "_data_prof_ctx_id_vwrite back_idx", cmo_ctx_ids);
      (void)ge::AttrUtils::SetStr(op_desc, "_data_prof_name_vwrite back_idx", "test_wback");
      (void)ge::AttrUtils::SetInt(op_desc, "_data_prof_typewrite back_idx", 1);
    }

  //最后一个节点，不设CMO属性
  const auto &node_last = nodes.at(nodes.size() - 1);
  const auto &op_desc_last = node_last->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc_last, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "1");
  (void)ge::AttrUtils::SetInt(op_desc_last, "_ffts_prof_ctx_type", 1);
  std::vector<uint32_t> ctx_ids;
  ctx_ids.push_back(0);
  (void)ge::AttrUtils::SetListInt(op_desc_last, "_context_id_list", ctx_ids);
  (void)ge::AttrUtils::SetInt(op_desc_last, "_op_impl_mode_enum", 0x40);

    auto frame = bg::ValueHolder::GetCurrentFrame();
    ExeGraphSerializer(*frame).SetComputeGraph(compute_graph).SerializeDfxExtendInfo(dfx_extend_infos, buffer_pool);
    auto buffer = dfx_extend_infos.Serialize(buffer_size);
    ge::AttrUtils::SetZeroCopyBytes(exe_graph, "DfxExtendInfo", ge::Buffer::CopyFrom(buffer.get(), buffer_size));

    buffer = buffer_pool.Serialize(buffer_size);
    ge::AttrUtils::SetZeroCopyBytes(exe_graph, "buffer", ge::Buffer::CopyFrom(buffer.get(), buffer_size));

    ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2), nullptr);
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void BuildExecutorInner3(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    auto compute_graph = ShareGraph::BuildSingleNodeGraph();
    auto node = compute_graph->FindNode("add1");
    auto op_desc = node->GetOpDesc();
    const std::vector<uint32_t> input_dims {1, 2, 3, 4};
    ge::AttrUtils::SetListInt(op_desc, kContextIdList, input_dims);
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "xxx");
    ge::AttrUtils::SetInt(op_desc, "_ffts_prof_ctx_type", RT_CTX_TYPE_AICORE);
    // add cmo context info
    for (size_t i = 0; i < CMO_IDX_KEY.size(); i++) {
      const std::string ctx_id_attr = kDataProfCtxIdV + CMO_IDX_KEY[i];
      const std::string name_attr = kDataProfName + CMO_IDX_KEY[i];
      const std::string type_attr = kDataProfType + CMO_IDX_KEY[i];
      std::vector<uint32_t> cmo_context_ids{1, 2, 3};
      std::stringstream cmo_context_name;
      cmo_context_name << "write_back_" << i;
      uint32_t cmo_context_type = RT_CTX_TYPE_FLUSH_DATA + i;
      (void)ge::AttrUtils::SetListInt(op_desc, ctx_id_attr, cmo_context_ids);
      (void)ge::AttrUtils::SetStr(op_desc, name_attr, cmo_context_name.str());
      (void)ge::AttrUtils::SetInt(op_desc, type_attr, cmo_context_type);
    }
    compute_graph->TopologicalSorting();
    GeModelBuilder builder(compute_graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(compute_graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
    ASSERT_NE(model_executor, nullptr);
  }

  static void BuildExecutorInner4(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    auto compute_graph = ShareGraph::BuildSingleNodeGraph();
    auto node = compute_graph->FindNode("add1");
    auto op_desc = node->GetOpDesc();
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_VECTOR_CORE");
    compute_graph->TopologicalSorting();
    GeModelBuilder builder(compute_graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

    for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
      if (it.first == compute_graph->GetName()) {
        ge::AttrUtils::SetInt(it.second, ge::ATTR_MODEL_STREAM_NUM, 3);
        ge::AttrUtils::SetInt(it.second, ge::ATTR_MODEL_EVENT_NUM, 0);
        ge::AttrUtils::SetInt(it.second, ge::ATTR_MODEL_NOTIFY_NUM, 4);
        ge::AttrUtils::SetInt(it.second, "_attached_stream_num", 1);
      }
    }
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(compute_graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
    ASSERT_NE(model_executor, nullptr);
  }

  static std::unique_ptr<ModelV2Executor> BuildExecutor3(LoweringOption option = {}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner3(model_executor, option);
    return model_executor;
  }

  static std::unique_ptr<ModelV2Executor> BuildExecutor4(LoweringOption option = {}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner4(model_executor, option);
    return model_executor;
  }

  static std::unique_ptr<ModelV2Executor> BuildExecutor2(LoweringOption option = {}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner2(model_executor, option);
    // turn off profiling to prevent other cases from being affected
    ge::diagnoseSwitch::DisableProfiling();
    return model_executor;
  }
  static std::unique_ptr<ModelV2Executor> BuildExecutor(LoweringOption option = {}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner(model_executor, option);
    return model_executor;
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
    ge::AttrUtils::SetInt(conv2d_desc, "op_para_size", 32);
    ge::AttrUtils::SetInt(conv2d_desc, "atomic_op_para_size", 32);
    (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
    (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
    (void)ge::AttrUtils::SetListInt(conv2d_desc, "_atomic_context_id_list", {0});
    (void)ge::AttrUtils::SetStr(conv2d_desc, "_atomic_kernelname", "test");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_atomic_cube_vector_core_type", "AIC");

    std::vector<std::string> names_prefix;
    names_prefix.emplace_back("_mix_aic");
    if (!single) {
      names_prefix.emplace_back("_mix_aiv");
      ge::AttrUtils::SetStr(node->GetOpDesc(), "_mix_aic_kernel_list_first_name", "aic");
      ge::AttrUtils::SetStr(node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");
    }
    (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  }

  static void TestMixL2SingleLowering(ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data, bool expect) {
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
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
    ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2), nullptr);
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()), ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();

    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "FFTSUpdateMixL2Args"), 1);

    Shape expect_out_shape = {4, 4, 4, 4};
    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

 protected:
  void TearDown() {
    ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
    ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  }

  static void TestProfilingWhenZeroCpy1() {
    auto model_executor = BuildExecutor({.trust_shape_on_out_tensor = true, .always_zero_copy = true});
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2),
        nullptr);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestProfiling() {
    auto model_executor = BuildExecutor();
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler),
        nullptr);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestProfilingHost() {
    auto model_executor = BuildExecutor();
    auto execution_data = reinterpret_cast<const ExecutionData *>(model_executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
    for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
      std::string kernel_type = reinterpret_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[i]->context.kernel_extend_info)->GetKernelType();
      if (kernel_type == "LaunchKernelWithHandle") {
        const_cast<KernelExtendInfo *>(reinterpret_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[i]->context.kernel_extend_info))->SetKernelType("AicpuHostCompute");
      }
    }
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler),
        nullptr);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    auto profiler = model_executor->GetSubscribers().MutableBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
    EXPECT_EQ(profiler->InitForCannDevice(execution_data), ge::SUCCESS);
    MsprofApi fake_api;
    profiler->InitLaunchApi(12345, "AicpuHostCompute", fake_api);
    ASSERT_EQ(fake_api.type, MSPROF_REPORT_NODE_HOST_OP_EXEC_TYPE);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestMemoryProfiling() {
    auto model_executor = BuildExecutor({.trust_shape_on_out_tensor = true, .always_zero_copy = true});
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler),
        nullptr);

    // turn on prof and execute
    ge::diagnoseSwitch::EnableProfiling({ProfilingType::kMemory});
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    // turn off prof
    ge::diagnoseSwitch::DisableProfiling();
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestMemoryProfilingWithTwoNode() {
    auto model_executor = BuildExecutor2({.trust_shape_on_out_tensor = true, .always_zero_copy = true});
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler),
        nullptr);

    // turn on prof and execute
    ge::diagnoseSwitch::EnableProfiling({ProfilingType::kMemory});
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    // turn off prof
    ge::diagnoseSwitch::DisableProfiling();
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestContextInfoProfiling() {
    auto model_executor = BuildExecutor3();
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2),
        nullptr);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestMixVector() {
    auto model_executor = BuildExecutor4();
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    ASSERT_NE(
        model_executor->GetSubscribers().GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2),
        nullptr);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(model_executor.get()->GetModelDesc().GetReusableStreamNum(), 2);
    EXPECT_EQ(model_executor.get()->GetModelDesc().GetAttachedStreamNum(), 1);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  static void TestLaunchInfoProfiling() {
    ge::GeRootModelPtr root_model;
    LoweringGlobalData global_data;
    BuildFftsPlusGraphAndTaskDef(root_model, global_data);
    TestFFTSLowering(root_model, global_data);
  }

  static void TestMixL2Profiling(ge::ComputeGraphPtr &root_graph, ge::NodePtr &node) {
    // Build FftsTaskDef.
    std::shared_ptr<domi::ModelTaskDef> model_task_def = ge::MakeShared<domi::ModelTaskDef>();
    auto &task_def = *model_task_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_FFTS_PLUS));
    auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

    auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
    ctx_def_0.set_context_id(0);
    ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
    (void)ctx_def_0.mutable_label_ctx();
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

    LoweringGlobalData global_data;
    global_data.AddCompiledResult(node, {{task_def}});
    OpImplSpaceRegistryV2Array space_registry_v2_array;
    space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
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
    TestMixL2SingleLowering(root_graph, global_data, true);
  }

  static void TestMixL2MemSetProfiling() {
    ge::ComputeGraphPtr root_graph;
    ge::NodePtr node = nullptr;
    BuildMixL2NodeGraph(root_graph, node, true);
    ge::AttrUtils::SetListInt(node->GetOpDesc(), "tbe_op_atomic_dtypes", {9});
    TestMixL2Profiling(root_graph, node);
  }

  static void TestMixL2AtomicCleanProfiling() {
    ge::ComputeGraphPtr root_graph;
    ge::NodePtr node = nullptr;
    BuildMixL2NodeGraph(root_graph, node, true);
    TestMixL2Profiling(root_graph, node);
  }

  static void TestDavinciModelReport() {
    auto graph = ShareGraph::BuildWithKnownSubgraph();
    graph->TopologicalSorting();
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
    GertRuntimeStub fakeRuntime;
    auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost, ProfilingType::kTaskTime, ProfilingType::kDevice});
    gert::GlobalProfilingWrapper::GetInstance()->IncreaseProfCount();
    auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
    auto outputs = FakeTensors({2, 2}, 3);
    auto inputs = FakeTensors({2, 2}, 1, mem_block.get());
    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ge::diagnoseSwitch::DisableProfiling();
    ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost, ProfilingType::kTaskTime, ProfilingType::kDevice});
    gert::GlobalProfilingWrapper::GetInstance()->IncreaseProfCount();
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                  reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ge::diagnoseSwitch::DisableProfiling();
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
};

/**
 * 用例描述：单算子用例执行时打开profiling的task_time开关，统计上报信息数量
 *
 * 预置条件:
 * 1.构造单算子执行器
 * 2.配置零拷贝和trust_shape_on_out_tensor
 *
 * 测试步骤：
 * 1. 使能cann profiling的task_time开关
 * 2. 构造单算子执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验profiling上报的api、info、compact info数量符合预期
 */
TEST_F(CannProfilerST, CannDeviceProfiling_ReportTaskAndTensorInfo_EnableAlwaysZeroCopy) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfilingWhenZeroCpy1, 1, 0, 0, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfilingWhenZeroCpy1, 1, 1, 0, 1);
}

/**
 * 用例描述：单算子用例执行时打开profiling的api开关，统计上报信息数量
 *
 * 预置条件:
 * 1.构造单算子执行器
 * 2.配置零拷贝和trust_shape_on_out_tensor
 *
 * 测试步骤：
 * 1. 使能cann profiling的Api开关
 * 2. 构造单算子执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验profiling上报的api、info、compact info数量符合预期
 */
TEST_F(CannProfilerST, CannHostProfiling_ReportApi_EnableAlwaysZeroCopy) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfilingWhenZeroCpy1, 1, 0, 0, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHostL1, ProfilingType::kCannHost});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfilingWhenZeroCpy1, 18, 0, 0, 0);
}
/**
* 用例描述：单算子用例执行时打开profiling的task_time和host profiling开关，统计上报信息数量
*
* 预置条件:
* 1.构造单算子执行器
*
* 测试步骤：
* 1. 使能cann profiling的task_time开关
* 2. 构造单算子执行器执行
*
* 预期结果：
* 1. 执行成功
* 2. 校验profiling上报的api、info、compact info数量符合预期
*/
TEST_F(CannProfilerST, CannProfiling_HostAndDevice_Normal) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost, ProfilingType::kTaskTime});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfiling, 2, 0, 0, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost, ProfilingType::kDevice});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfiling, 2, 1, 0, 1);
}

// todo: 此处是覆盖profiling_v1中代码，后续需要伴随全部切换profiling_v2后删除
TEST_F(CannProfilerST, InitCannProfilingV1_Ok) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
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

/**
 * 用例描述：单算子用例执行时打开profiling的task_memory开关，统计上报信息数量
 *
 * 预置条件:
 * 1.构造单算子执行器
 * 2.配置零拷贝和trust_shape_on_out_tensor
 *
 * 测试步骤：
 * 1. 使能cann profiling的memory开关
 * 2. 构造单算子执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验profiling上报的task_memory数量符合预期
 */
TEST_F(CannProfilerST, CannHostProfiling_ReportMemory_EnableAlwaysZeroCopy) {
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestMemoryProfiling, 0, 34, 0, 0);
}

/**
 * 用例描述：单算子用例执行时打开profiling的task_memory开关，图中有两个算子，统计上报信息数量
 *
 * 预置条件:
 * 1.构造单算子执行器
 * 2.配置零拷贝和trust_shape_on_out_tensor
 *
 * 测试步骤：
 * 1. 使能cann profiling的memory开关
 * 2. 构造单算子执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验profiling上报的task_memory数量符合预期
 */
TEST_F(CannProfilerST, CannProfiling_ReportMemoryWithTwoNode_EnableAlwaysZeroCopy) {
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestMemoryProfilingWithTwoNode, 0, 69, 0, 0);
}

TEST_F(CannProfilerST, CannProfiling_test_contextinfo) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestContextInfoProfiling, 1, 3, 0, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestContextInfoProfiling, 1, 4, 0, 3);
}

TEST_F(CannProfilerST, CannProfiling_test_mix_vectorcore) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestMixVector, 1, 0, 0, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestMixVector, 1, 1, 0, 1);
}

TEST_F(CannProfilerST, CannProfiling_test_l2mix_blockdim) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestMixL2MemSetProfiling, 1, 2, 0, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestMixL2AtomicCleanProfiling, 1, 3, 0, 2);
}

TEST_F(CannProfilerST, CannProfiling_HostAicpu_Normal) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost, ProfilingType::kTaskTime, ProfilingType::kDevice});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfilingHost, 2, 1, 0, 1);
  ge::diagnoseSwitch::DisableProfiling();
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kCannHost, ProfilingType::kTaskTime});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestProfilingHost, 2, 0, 0, 0);
}

TEST_F(CannProfilerST, CannProfiling_davinci_model_report) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(CannProfilerST::TestDavinciModelReport, 2, 4, 4, 2);
}
}  // namespace gert
