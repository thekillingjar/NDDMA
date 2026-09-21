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
#include <nlohmann/json.hpp>
#define private public
#include "base/registry/op_impl_space_registry_v2.h"
#undef private
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <vector>
#include "sys/stat.h"
#include "inc/ffts_type.h"
#include "runtime/rt_model.h"
#include "rt_error_codes.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/args_format_desc_utils.h"
#include "common/l2_stream_info.h"
#include "common/resource_def.h"
#include "common/util/op_info_util.h"
#include "common/comm_error_codes.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_gentask_utils.h"
#include "../fe_test_utils.h"
#include "fe_llt_utils.h"
#include "securec.h"
#include "common/fe_inner_attr_define.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/tuning_utils.h"
#include "graph/node.h"
#include "fe_llt_utils.h"
#include "common/sgt_slice_type.h"
#include "common/ffts_plus_type.h"
#include "graph/model_serialize.h"
#include "graph/ge_attr_value.h"
#include "graph/detail/model_serialize_imp.h"
#define protected public
#define private public
#include "register/op_ext_calc_param_registry.h"
#include "common/fe_log.h"
#include "ops_kernel_builder/aicore_ops_kernel_builder.h"
#include "ops_kernel_builder/dsa_ops_kernel_builder.h"
#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "adapter/adapter_itf/op_store_adapter.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "fusion_manager/fusion_manager.h"
#include "ops_kernel_builder/task_builder/task_builder.h"
#include "param_calculate/tensorsize_calculator.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_optimizer.h"
#include "register/op_ext_gentask_registry.h"
#include "common/fe_inner_attr_define.h"
#include "common/platform_utils.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/tuning_utils.h"
#include "graph/node.h"

#include "common/sgt_slice_type.h"
#include "common/ffts_plus_type.h"

#include "graph/model_serialize.h"
#include "graph/ge_attr_value.h"
#include "graph/detail/model_serialize_imp.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#undef private
#undef protected
#define SET_SIZE 128

using namespace std;
using namespace testing;
using namespace ge;
using namespace domi;
using namespace fe;
using namespace ffts;

using AICoreOpsKernelBuilderPtr =  shared_ptr<fe::AICoreOpsKernelBuilder>;
using DsaOpsKernelBuilderPtr = shared_ptr<DsaOpsKernelBuilder>;
FEOpsStoreInfo cce_custom_opinfo_adapter {
      0,
      "cce-custom",
      fe::EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_custom_opinfo",
      ""
};
FEOpsStoreInfo tik_custom_opinfo_adapter  {
      1,
      "tik-custom",
      fe::EN_IMPL_CUSTOM_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_custom_opinfo",
      ""
};
FEOpsStoreInfo tbe_custom_opinfo_adapter  {
      2,
      "tbe-custom",
      fe::EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""
};
FEOpsStoreInfo cce_constant_opinfo_adapter  {
      3,
      "cce-constant",
      fe::EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_constant_opinfo",
      ""
};
FEOpsStoreInfo cce_general_opinfo_adapter  {
      4,
      "cce-general",
      fe::EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_general_opinfo",
      ""
};
FEOpsStoreInfo tik_opinfo_adapter  {
      5,
      "tik-builtin",
      fe::EN_IMPL_HW_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_opinfo",
      ""
};
FEOpsStoreInfo tbe_opinfo_adapter  {
      6,
      "tbe-builtin",
      fe::EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      ""
};
FEOpsStoreInfo rl_opinfo_adapter  {
      7,
      "rl-builtin",
      fe::EN_IMPL_RL,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/rl_opinfo",
      ""
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_adapter{
      cce_custom_opinfo_adapter ,
      tik_custom_opinfo_adapter ,
      tbe_custom_opinfo_adapter ,
      cce_constant_opinfo_adapter ,
      cce_general_opinfo_adapter ,
      tik_opinfo_adapter ,
      tbe_opinfo_adapter ,
      rl_opinfo_adapter
};

ge::Status tilingSinkSuc(const ge::Node &node) {
  return ge::SUCCESS;
}

ge::Status tilingSinkFail(const ge::Node &node) {
  return ge::FAILED;
}

ge::graphStatus CalcParamKernelFunc(gert::ExeResGenerationContext *context) {
  context->SetWorkspaceBytes({22});
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenTaskKernelFunc(const gert::ExeResGenerationContext *context,
                                  std::vector<std::vector<uint8_t>> &tasks) {
  const_cast<gert::ExeResGenerationContext *>(context)->SetWorkspaceBytes({33, 44});
  return ge::GRAPH_SUCCESS;
}

class STEST_TaskBuilder: public testing::Test {
protected:

    static void SetOpDecSize(NodePtr& node){
        OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
        for (int i = 0; i < node->GetOpDesc()->GetAllInputsDesc().size(); i++){
            ge::GeTensorDesc tensor = node->GetOpDesc()->GetAllInputsDesc().at(i);
            ge::TensorUtils::SetSize(tensor, SET_SIZE);
            node->GetOpDesc()->UpdateInputDesc(i, tensor);
        }
        OpDesc::Vistor<GeTensorDesc> tensors_output = node->GetOpDesc()->GetAllOutputsDesc();
        for (int i = 0; i < tensors_output.size(); i++){
            ge::GeTensorDesc tensor_output = tensors_output.at(i);
            ge::TensorUtils::SetSize(tensor_output, SET_SIZE);
            node->GetOpDesc()->UpdateOutputDesc(i, tensor_output);
        }
    }

    static void SetFFTSOpDecSize(NodePtr& node) {
      OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
      for (int i = 0; i < node->GetOpDesc()->GetAllInputsDesc().size(); i++) {
        ge::GeTensorDesc tensor = node->GetOpDesc()->GetAllInputsDesc().at(i);
        ge::TensorUtils::SetSize(tensor, 10000);
        node->GetOpDesc()->UpdateInputDesc(i, tensor);
      }
      OpDesc::Vistor<GeTensorDesc> tensorsOutput = node->GetOpDesc()->GetAllOutputsDesc();
      for (int i = 0; i < tensorsOutput.size(); i++) {
        ge::GeTensorDesc tensorOutput = tensorsOutput.at(i);
        ge::TensorUtils::SetSize(tensorOutput, 10000);
        node->GetOpDesc()->UpdateOutputDesc(i, tensorOutput);
      }
      for (auto const &anchor : node->GetAllInDataAnchors()) {
        (void)ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
      }
    }

    void SetUp()
    {
        rtContext_t rt_context;
        assert(rtCtxCreate(&rt_context, RT_CTX_GEN_MODE, 0) == ACL_RT_SUCCESS);
        assert(rtCtxSetCurrent(rt_context) == ACL_RT_SUCCESS);
        //cce::cceSysInit();

        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_adapter);
        context_ = CreateContext();
        task_builder_ = shared_ptr <TaskBuilder> (new (nothrow) TaskBuilder());
        FusionManager::Instance(fe::AI_CORE_NAME).ops_kernel_info_store_ = make_shared<fe::FEOpsKernelInfoStore>();
        FusionManager::Instance(fe::AI_CORE_NAME).graph_opt_ = make_shared<fe::FEGraphOptimizer>(FusionManager::Instance(fe::AI_CORE_NAME).ops_kernel_info_store_);
        TbeOpStoreAdapter tbe_adapter(AI_CORE_NAME);
        std:: map<string, string> options;

        aicore_ops_kernel_builder_ptr = make_shared<AICoreOpsKernelBuilder>();
        aicore_ops_kernel_builder_ptr->Initialize(options);
        dsa_ops_kernel_builder_ptr_ = std::shared_ptr<DsaOpsKernelBuilder>(new DsaOpsKernelBuilder());
        dsa_ops_kernel_builder_ptr_->Initialize(options);
        auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
        OpsKernelManager::Instance(fe::AI_CORE_NAME).Finalize();
        OpsKernelManager::Instance(fe::AI_CORE_NAME).Initialize();
    }

    void TearDown()
    {
        task_builder_.reset();
        DestroyContext(context_);

        rtContext_t rt_context;
        assert(rtCtxGetCurrent(&rt_context) == ACL_RT_SUCCESS);
        assert(rtCtxDestroy(rt_context) == ACL_RT_SUCCESS);
        aicore_ops_kernel_builder_ptr->Finalize();
        dsa_ops_kernel_builder_ptr_->Finalize();
    }

    static NodePtr CreateNodeWithoutAttrs(bool hasWeight = false)
    {
        FeTestOpDescBuilder builder;
        builder.SetName("test_tvm");
        builder.SetType("conv");
        builder.SetInputs( { 1 });
        builder.SetOutputs( { 1 });
        builder.AddInputDesc( { 1, 1, 1, 1 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
        builder.AddOutputDesc( { 1, 1, 1, 1 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
        if (hasWeight) {
            size_t len = 10;
            unique_ptr<float[]> buf(new float[len]);
            auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 }, ge::FORMAT_NCHW,
                    ge::DT_FLOAT);
            ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
        }

        return builder.Finish();
    }

    static NodePtr CreateNode(bool hasWeight = false)
    {
        NodePtr node = CreateNodeWithoutAttrs(hasWeight);

        std::string bin_file = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
        vector<char> buffer;
        assert(ReadBytesFromBinaryFile(bin_file.c_str(), buffer));
        OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
        node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

        ge::AttrUtils::SetInt(node->GetOpDesc(), "_fe_imply_type", (int64_t)fe::EN_IMPL_CUSTOM_TBE);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "cce_reductionLayer_1_10_float16__1_SUMSQ_1_0__kernel0");
        ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 1);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
        ge::AttrUtils::SetBool(node->GetOpDesc(), "is_first_node", true);
        ge::AttrUtils::SetBool(node->GetOpDesc(), "is_last_node", true);
        std::string meta_data = "";
        meta_data.append("cce_reductionLayer_1_10_float16__1_SUMSQ_1_0"); // binFileName
        meta_data.append(".o");  // binFileSuffix
        meta_data.append(",version,");
        meta_data.append("c53fcf5403daaf993a95a4aeea228eae30196565d8b287bae9a4ca6a52e58c2b");    // sha256
        meta_data.append(",shared");
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", meta_data);
        SetOpDecSize(node);
        return node;
    }
  static NodePtr CreateNormalNode(const int &type)
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("conv");
    builder.SetInputs({1});
    builder.SetOutputs({1});
    builder.AddInputDesc({1,2,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    builder.AddOutputDesc({1,2,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto node = builder.Finish();

    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin+strlen(tbeBin));
    OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);

    ge::AttrUtils::SetInt(node->GetOpDesc(), "_fe_imply_type", (int64_t)fe::EN_IMPL_CUSTOM_TBE);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    if (type == 1 || type ==3) {
      ge::AttrUtils::SetBool(node->GetOpDesc(), "is_first_node", true);
    }
    if (type == 2 || type ==3) {
      ge::AttrUtils::SetBool(node->GetOpDesc(), "is_last_node", true);
    }
    if (type == 4) {
      ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_KERNEL_LIST_FIRST_NAME, node->GetName());
    }
    if (type == 5) {
      ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {1<<16});
    }
    ge::AttrUtils::SetBool(node->GetOpDesc(), "support_dynamicshape", false);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "kernelname");

    SetOpDecSize(node);
    return node;
  }
  static NodePtr CreateIFANode()
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("IncreFlashAttention");
    builder.SetInputs({1, 2, 3, 6});
    builder.SetOutputs({1});
    builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "a");
    builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b0");
    builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b1");
    // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "c");
    // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "d");
    // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "e");
    // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "f");
    builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "g");
    builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "out");
    auto node = builder.Finish();
    auto op_desc = node->GetOpDesc();
    (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 7);
    ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
    std::vector<uint32_t> input_type_list = {0, 2, 2, 1};
    (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
    std::vector<std::string> input_name_list = {"a", "b", "b","g"};
    (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
    std::vector<uint32_t> output_type_list = {0};
    (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
    std::vector<std::string> output_name_list = {"out"};
    (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

    (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
    (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

    auto pure_op_desc = node->GetOpDescBarePtr();
    pure_op_desc->AppendIrInput("a", ge::kIrInputRequired);
    pure_op_desc->AppendIrInput("b", ge::kIrInputDynamic);
    pure_op_desc->AppendIrInput("c", ge::kIrInputOptional);
    pure_op_desc->AppendIrInput("d", ge::kIrInputOptional);
    pure_op_desc->AppendIrInput("e", ge::kIrInputOptional);
    pure_op_desc->AppendIrInput("f", ge::kIrInputOptional);
    pure_op_desc->AppendIrInput("g", ge::kIrInputOptional);
    pure_op_desc->AppendIrOutput("out", ge::kIrOutputRequired);

    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin+strlen(tbeBin));
    OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);

    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetBool(node->GetOpDesc(), "support_dynamicshape", false);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "kernelname");

    SetOpDecSize(node);
    return node;
  }
  static NodePtr CreateDynamicNode(const int &type)
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("conv");
    builder.SetInputs({1});
    builder.SetOutputs({1});
    builder.AddInputDesc({1,2,-1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    builder.AddOutputDesc({1,2,-1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto node = builder.Finish();

    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin+strlen(tbeBin));
    OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);

    ge::AttrUtils::SetInt(node->GetOpDesc(), "_fe_imply_type", (int64_t)fe::EN_IMPL_CUSTOM_TBE);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    if (type == 1 || type ==3) {
      ge::AttrUtils::SetBool(node->GetOpDesc(), "is_first_node", true);
    }
    if (type == 2 || type ==3) {
      ge::AttrUtils::SetBool(node->GetOpDesc(), "is_last_node", true);
    }
    if (type == 4) {
      ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_KERNEL_LIST_FIRST_NAME, node->GetName());
    }
    ge::AttrUtils::SetBool(node->GetOpDesc(), "support_dynamicshape", false);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "kernelname");

    SetOpDecSize(node);
    return node;
  }

    static RunContext CreateContext()
    {
        RunContext context;
        context.dataMemSize = 100;
        context.dataMemBase = (uint8_t *) (intptr_t) 1000;
        context.weightMemSize = 200;
        context.weightMemBase = (uint8_t *) (intptr_t) 1100;
        context.weightsBuffer = Buffer(20);

        return context;
    }

    static void DestroyContext(RunContext &context) {
    }

    static bool ReadBytesFromBinaryFile(const char* path, std::vector<char>& buffer)
    {
        if (path == nullptr)
            return false;

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if(!file.is_open())
            return false;

        std::streamsize size = file.tellg();

        if(size <= 0) {
            file.close();
            return false;
        }

        if (size > INT_MAX) {
            file.close();
            return false;
        }

        file.seekg(0, std::ios::beg);

        buffer.resize(size);
        file.read(buffer.data(), size);
        file.close();

        return true;
    }

protected:
    RunContext context_;
    std::shared_ptr<TaskBuilder> task_builder_;
    AICoreOpsKernelBuilderPtr aicore_ops_kernel_builder_ptr;
    DsaOpsKernelBuilderPtr dsa_ops_kernel_builder_ptr_;
};

TEST_F(STEST_TaskBuilder, case_all_success_01)
{
    NodePtr node = CreateNode();
    FillNodeParaType(node);
    std::vector<TaskDef> task_defs;
    EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, case_all_success_02)
{
  NodePtr node = CreateNode();
  FillNodeParaType(node);
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_KERNEL_LIST_FIRST_NAME, node->GetName());
  std::vector<TaskDef> task_defs;
  EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, case_all_success_03)
{
  SetFunctionState(FuncParamType::FUSION_L2, false);
  NodePtr node = CreateNode();
  FillNodeParaType(node);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_LX_FUSION_PASS, true);
  std::vector<TaskDef> task_defs;
  EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, case_all_success_cust_log)
{
  SetFunctionState(FuncParamType::FUSION_L2, false);
  NodePtr node = CreateNode();
  FillNodeParaType(node);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_LX_FUSION_PASS, true);
  node->GetOpDesc()->SetWorkspaceBytes({1, 1});
  node->GetOpDesc()->SetWorkspace({1, 1});
  std::vector<int64_t> mem_type_list = {1, 0};
  ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, mem_type_list);
  std::vector<TaskDef> task_defs;
  EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, case_all_success_04)
{
  SetFunctionState(FuncParamType::FUSION_L2, false);
  NodePtr node = CreateNode();
  FillNodeParaType(node);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_LX_FUSION_PASS, true);
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_KERNEL_LIST_FIRST_NAME, node->GetName());
  std::vector<TaskDef> task_defs;
  EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, case_all_success_05)
{
  NodePtr node = CreateNode();
  FillNodeParaType(node);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::ContextSwitch)] = 1;
  std::vector<TaskDef> task_defs;
  EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
  bool slow_switch = false;
  ge::AttrUtils::GetBool(node->GetOpDesc(), "_slow_context_switch", slow_switch);
  EXPECT_EQ(slow_switch, true);
  string buffer_type;
  ge::AttrUtils::GetStr(node->GetOpDesc(), "_switch_buffer_type", buffer_type);
  EXPECT_EQ(buffer_type.size(), 0);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::ContextSwitch)] = 0;
}

TEST_F(STEST_TaskBuilder, case_all_success_06)
{
  NodePtr node = CreateNode();
  FillNodeParaType(node);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {1<<16});
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::ContextSwitch)] = 1;
  std::vector<TaskDef> task_defs;
  EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
  bool slow_switch = false;
  ge::AttrUtils::GetBool(node->GetOpDesc(), "_slow_context_switch", slow_switch);
  EXPECT_EQ(slow_switch, true);
  string buffer_type;
  ge::AttrUtils::GetStr(node->GetOpDesc(), "_switch_buffer_type", buffer_type);
  EXPECT_EQ(buffer_type, "UF");
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::ContextSwitch)] = 0;
}

TEST_F(STEST_TaskBuilder, read_binary_file_success)
{
    std::string bin_file = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    vector<char> buffer;
    TbeJsonFileParseImpl tbe_json_file_parse_impl;
    EXPECT_EQ(tbe_json_file_parse_impl.ReadBytesFromBinaryFile(bin_file, buffer), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, read_binary_file_lock_fail)
{
    std::string bin_file = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    vector<char> buffer;

    // manually lock first
    std::string file = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    FILE *fp = fopen(file.c_str(), "r");
    if (fp == nullptr) {
        EXPECT_EQ(true, false);
    }
    if (FcntlLockFile(file, fileno(fp), F_RDLCK, 0) != fe::SUCCESS) {
        EXPECT_EQ(true, false);
    }

    TbeJsonFileParseImpl tbe_json_file_parse_impl;
    EXPECT_EQ(tbe_json_file_parse_impl.ReadBytesFromBinaryFile(bin_file, buffer), fe::SUCCESS);
    (void)FcntlLockFile(file, fileno(fp), F_UNLCK, 0);
    fclose(fp);
}

TEST_F(STEST_TaskBuilder, l2with_confirm)
{
    TaskL2Info ll;
    ll.nodeName = "1";
    //TaskL2Info *l2Data = &ll;
    TaskL2InfoMap hh;
    hh["0"] = ll;

    StreamL2Info::Instance().SetStreamL2Info(0, hh, "Batch_-1");
    SetFunctionState(FuncParamType::FUSION_L2, true);
    NodePtr node = CreateNode();
    FillNodeParaType(node);
    std::vector<TaskDef> task_defs;
    EXPECT_EQ(task_builder_->GenerateKernelTask(*node, context_, task_defs), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, dynamic_node_generate_task_1)
{
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateDynamicNode(0);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
  domi::KernelDef *kernel_def = task_defs[0].mutable_kernel();
  auto kernel_context = kernel_def->mutable_context();
  auto args_str = kernel_context->args_format();
  EXPECT_STREQ(args_str.c_str(), "{i_instance0*}{o_instance0*}{ws*}{t}");
}

TEST_F(STEST_TaskBuilder, dynamic_node_generate_task_2)
{
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateDynamicNode(1);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, dynamic_node_generate_task_3)
{
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateDynamicNode(2);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, dynamic_node_generate_task_4)
{
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateDynamicNode(3);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, dynamic_node_generate_task_5)
{
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateDynamicNode(4);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(STEST_TaskBuilder, mix_static_node_generate_task_1)
{

  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  string soc_version = "Ascend310P3";
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  fe::PlatformInfoManager::Instance().init_flag_ = true;
  fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.Init();
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
  fe::PlatformInfoManager::GeInstance().platform_info_map_.clear();
  fe::PlatformInfoManager::GeInstance().platform_infos_map_.clear();
  init_ret = fe::PlatformInfoManager::GeInstance().LoadConfigFile(real_path);
  fe::PlatformInfoManager::GeInstance().init_flag_ = true;
  fe::PlatformInfoManager::GeInstance().opti_compilation_info_.soc_version = soc_version;
  fe::PlatformInfoManager::GeInstance().opti_compilation_infos_.Init();

  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateNormalNode(1);
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  auto graph = std::make_shared<ComputeGraph>("test");
  (void)node->SetOwnerComputeGraph(graph);
  graph->SetGraphUnknownFlag(false);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(task_defs.size(), 2);
}

TEST_F(STEST_TaskBuilder, mix_static_node_generate_task_2)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateNormalNode(1);
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, 48);
  std::vector<int32_t> notify_id_v = {2, 3};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::RECV_ATTR_NOTIFY_ID, notify_id_v);
  auto graph = std::make_shared<ComputeGraph>("test");
  (void)node->SetOwnerComputeGraph(graph);
  graph->SetGraphUnknownFlag(false);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(task_defs.size(), 7);
}

TEST_F(STEST_TaskBuilder, mix_static_node_generate_task_3)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateNormalNode(1);
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixAICore);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, 48);
  std::vector<int32_t> notify_id_v = {2};  // size not right
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::RECV_ATTR_NOTIFY_ID, notify_id_v);
  auto graph = std::make_shared<ComputeGraph>("test");
  (void)node->SetOwnerComputeGraph(graph);
  graph->SetGraphUnknownFlag(false);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(task_defs.size(), 2);
}

TEST_F(STEST_TaskBuilder, mix_dynamic_node_generate_task_1)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateNormalNode(1);
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  auto graph = std::make_shared<ComputeGraph>("test");
  (void)node->SetOwnerComputeGraph(graph);
  graph->SetGraphUnknownFlag(true);
  graph->SetParentNode(node);
  FillNodeParaType(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(ge::AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_MIX_CORE_NUM_VEC), true);
}

TEST_F(STEST_TaskBuilder, test_op_args_format_1)
{
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  std::vector<domi::TaskDef> task_defs;
  ge::NodePtr node = CreateIFANode();
  ge::AttrUtils::SetStr(node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, fe::kCoreTypeAIC);
  auto graph = std::make_shared<ComputeGraph>("test");
  (void)node->SetOwnerComputeGraph(graph);
  graph->SetGraphUnknownFlag(true);
  graph->SetParentNode(node);
  Status status = task_builder_->GenerateKernelTask(*node, context_, task_defs);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1);
  auto &task = task_defs[0];
  auto args_format_str = task.kernel().context().args_format();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i_desc1}{i2*}{i3*}{i4*}{i5*}{i6*}{o0*}");
}

ComputeGraphPtr BuildGraph_Readonly_ScopeWrite1() {

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast", "Cast");
  OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

  vector<int64_t> dim_a = {8, 4, 16, 16};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_NCHW);
  tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT);

  op_desc_cast1->AddOutputDesc(tensor_desc_a);

  op_desc_relu->AddInputDesc(tensor_desc_a);
  op_desc_relu->AddOutputDesc(tensor_desc_a);
  op_desc_output->AddInputDesc(tensor_desc_a);


  NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
  NodePtr node_relu = graph->AddNode(op_desc_relu);
  NodePtr node_output = graph->AddNode(op_desc_output);
  GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_output->GetInDataAnchor(0));

  std::string subgraph_name_1 = "instance_branch_1";
  ComputeGraphPtr subgraph_1 = std::make_shared<ComputeGraph>(subgraph_name_1);
  subgraph_1->SetParentNode(node_cast1);
  subgraph_1->SetParentGraph(graph);
  node_relu->GetOpDesc()->AddSubgraphName("branch1");
  node_relu->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name_1);
  return graph;
}

NodePtr MakeNode(const ComputeGraphPtr &graph, uint32_t in_num, uint32_t out_num, string name, string type) {
  GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (auto i = 0; i < in_num; ++i) {
    op_desc->AddInputDesc(test_desc);
  }
  for (auto i = 0; i < out_num; ++i) {
    op_desc->AddOutputDesc(test_desc);
  }
  return graph->AddNode(op_desc);
}

NodePtr CreateNode1(OpDescPtr op, ComputeGraphPtr owner_graph)
{ return owner_graph->AddNode(op); }

TEST_F(STEST_TaskBuilder, generate_auto_aic_aiv_ctx_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  std::vector<int64_t> workspaces = {10, 30, 40};
  src_op->SetWorkspaceBytes(workspaces);
  std::vector<int64_t> workspace_list{1, 2, 3};
  src_op->SetWorkspace(workspace_list);
  int64_t non_tail_workspace_size = 3;
  (void)ge::AttrUtils::SetInt(src_op, fe::NON_TAIL_WORKSPACE_SIZE, non_tail_workspace_size);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::AUTO_THREAD);
  slice_info_ptr->input_tensor_slice = {{{{0, 3}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}},
                                        {{{3, 6}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}}};
slice_info_ptr->output_tensor_slice = {{{{0, 3}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}, {{0, 3}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}},
                                       {{{3, 6}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}, {{3, 6}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}}};
  src_op->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetListStr(src_op, fe::ATTR_NAME_THREAD_CUBE_VECTOR_CORE_TYPE, {"AIC"});
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  auto src_node = graph->AddNode(src_op);
  SetFFTSOpDecSize(src_node);
  std::vector<domi::TaskDef> tasks;
  ge::ComputeGraphPtr tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph");
  ge::OpDescPtr memset_op_desc_ptr =  make_shared<ge::OpDesc>("memset_node", fe::MEMSET_OP_TYPE);
  ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, src_op->GetId());
  std::vector<int64_t> output_index = {0, 1};
  (void)ge::AttrUtils::SetListInt(memset_op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);
  std::vector<int64_t> workspace_index = {0, 1};
  (void)ge::AttrUtils::SetListInt(memset_op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_INDEX, workspace_index);

  src_op->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, generate_kernel_task_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({5, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({5, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  (void)ge::AttrUtils::SetStr(src_op, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  auto src_node = graph->AddNode(src_op);
  std::vector<domi::TaskDef> tasks;

  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_TaskBuilder, generate_manual_mix_aic_aiv_ctx_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({5, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({5, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::MANUAL_THREAD);
  src_op->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetStr(src_op, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(src_op, "compile_info_json", json_str);
  auto src_node = graph->AddNode(src_op);
  SetFFTSOpDecSize(src_node);
  std::vector<domi::TaskDef> tasks;

  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, generate_auto_mix_aic_aiv_ctx_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::AUTO_THREAD);
  slice_info_ptr->input_tensor_slice = {{{{0, 3}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}},
                                        {{{3, 6}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}}};
  slice_info_ptr->output_tensor_slice = {{{{0, 3}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}},
                                        {{{3, 6}, {0, 2}, {0, 3}, {0, 3}, {0, 2}}}};
  src_op->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetBool(src_op, kStaticToDynamicSoftSyncOp, true);
  (void)ge::AttrUtils::SetListStr(src_op, fe::ATTR_NAME_THREAD_CUBE_VECTOR_CORE_TYPE, {"MIX_AIC"});
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  auto src_node = graph->AddNode(src_op);
  SetFFTSOpDecSize(src_node);
  std::vector<domi::TaskDef> tasks;

  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, generate_ffts_dsa_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
   std::vector<domi::TaskDef> task_defs;
  (void)ge::AttrUtils::SetBool(src_op, "_ffts_plus", true);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(STEST_TaskBuilder, generate_dsa_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
   std::vector<domi::TaskDef> task_defs;
  (void)ge::AttrUtils::SetBool(src_op, "_ffts_plus", false);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(STEST_TaskBuilder, workspace_size_1_generate_dsa_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
   std::vector<domi::TaskDef> task_defs;
  (void)ge::AttrUtils::SetBool(src_op, "_ffts_plus", false);
  src_op->SetWorkspace({1024});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(STEST_TaskBuilder, generate_dsa_static_const_as_value_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const3 = std::make_shared<OpDesc>("const3", "Const");
  OpDescPtr const4 = std::make_shared<OpDesc>("const4", "Const");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  const1->AddOutputDesc(src_tensor_desc);
  const2->AddOutputDesc(src_tensor_desc);
  const3->AddOutputDesc(src_tensor_desc);
  const4->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto const1_node = graph->AddNode(const1);
  auto const2_node = graph->AddNode(const2);
  auto const3_node = graph->AddNode(const3);
  auto const4_node = graph->AddNode(const4);
  auto src_node = graph->AddNode(src_op);

  ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(3));
  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(STEST_TaskBuilder, generate_dsa_dynamic_const_as_addr_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const3 = std::make_shared<OpDesc>("const3", "Const");
  OpDescPtr const4 = std::make_shared<OpDesc>("const4", "Const");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  const1->AddOutputDesc(src_tensor_desc);
  const2->AddOutputDesc(src_tensor_desc);
  const3->AddOutputDesc(src_tensor_desc);
  const4->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto const1_node = graph->AddNode(const1);
  auto const2_node = graph->AddNode(const2);
  auto const3_node = graph->AddNode(const3);
  auto const4_node = graph->AddNode(const4);
  auto src_node = graph->AddNode(src_op);

  ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(3));
  (void)ge::AttrUtils::SetBool(graph, "_graph_unknown_flag", true);
  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(STEST_TaskBuilder, CalcSingleTensorSize)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = MakeNode(graph, 1, 1, "node", "Test");
  GeTensorDesc tensor_desc(GeShape({1,-1,-1,224}), FORMAT_NCHW, DT_UNDEFINED);
  node->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  int64_t tensor_size = 0;
  GeTensorDescPtr input_desc = node->GetOpDesc()->MutableInputDesc(0);
  (void)ge::AttrUtils::SetBool(input_desc, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  Status status = TensorSizeCalculator::CalcSingleTensorSize(*node->GetOpDesc(), input_desc, "TEST", 1, true, tensor_size);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(STEST_TaskBuilder, generate_dynamic_mix_aic_aiv_ctx_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, -1, 3, -1, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::AUTO_THREAD);
  src_op->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  uint32_t task_ratio = 1U;
  (void)ge::AttrUtils::SetInt(src_op, kTaskRadio, task_ratio);
  auto src_node = graph->AddNode(src_op);
  SetFFTSOpDecSize(src_node);
  std::vector<domi::TaskDef> tasks;
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, generate_memset_ctx_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::MANUAL_THREAD);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});

  OpDescPtr memset_op = std::make_shared<OpDesc>("MemSet1", "MemSet");
  ComputeGraphPtr temp_graph = std::make_shared<ComputeGraph>("temp");
  auto memset_node = temp_graph->AddNode(memset_op);
  (void)src_op->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  std::vector<domi::TaskDef> tasks;

  auto src_node = graph->AddNode(src_op);
  SetFFTSOpDecSize(src_node);
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<int64_t> work_spaces = memset_node->GetOpDesc()->GetWorkspace();
  size_t workspace_size = work_spaces.size();
  size_t zero_workspace_size = 0;
  for (size_t index = 0; index < workspace_size; index++) {
    if (work_spaces[index] == 0) {
      zero_workspace_size++;
    }
  }
  EXPECT_EQ(workspace_size, 3);
  EXPECT_EQ(zero_workspace_size, 1);
}

TEST_F(STEST_TaskBuilder, tiling_sink_suc)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::MANUAL_THREAD);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});

  auto src_node = graph->AddNode(src_op);
  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  REGISTER_NODE_EXT_CALC_PARAM("A", tilingSinkSuc);
  ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, tiling_sink_fail)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ThreadMode::MANUAL_THREAD);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});

  auto src_node = graph->AddNode(src_op);
  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  REGISTER_NODE_EXT_CALC_PARAM("A", tilingSinkFail);
  ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_TaskBuilder, op_open_interface_test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();
  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
  workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 2);
  EXPECT_EQ(workspace_bytes[1], 44);
}

namespace {
  ge::graphStatus InferShapeStub(gert::InferShapeContext *context) { return fe::SUCCESS;}
} // namespace

TEST_F(STEST_TaskBuilder, tiling_sink_calc_param){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();
  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  bool proc_flag = false;
  Status ret = aicore_ops_kernel_builder_ptr->CalcTilingSinkRunningParam(true, *src_node, proc_flag);
  proc_flag = false;
  IMPL_OP(FIA).InferShape(InferShapeStub);
  IMPL_OP(FIA).TilingInputsDataDependency({1}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU});
  ret = aicore_ops_kernel_builder_ptr->CalcTilingSinkRunningParam(true, *src_node, proc_flag);
  proc_flag = false;
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = aicore_ops_kernel_builder_ptr->CalcTilingSinkRunningParam(true, *src_node, proc_flag);
  proc_flag = true;
  ret = aicore_ops_kernel_builder_ptr->CalcTilingSinkRunningParam(true, *src_node, proc_flag);
  proc_flag = false;
  ge::AttrUtils::SetStr(src_op, OPS_PATH_NAME_PREFIX, "ops_test");
  ret = aicore_ops_kernel_builder_ptr->CalcTilingSinkRunningParam(true, *src_node, proc_flag);
}

TEST_F(STEST_TaskBuilder, tiling_sink_gen_task_fail){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;

  domi::TaskDef task_def_new;
  domi::TaskDef *task_def_real = &task_def_new;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def_real->mutable_ffts_plus_task();
  task_def_real->set_type(RT_MODEL_TASK_FFTS_PLUS_TASK);
  ffts_plus_task_def->set_op_index(1);
  task_def_real->set_stream_id(1);
  ge::ArgsFormatDescUtils args_desc_util;
  vector<ge::ArgDesc> arg_descs;
  args_desc_util.Append(arg_descs, ge::AddrType::WORKSPACE);
  args_desc_util.Append(arg_descs, ge::AddrType::TILING_FFTS);
  const std::string args_format = args_desc_util.ToString(arg_descs);
  auto kernel_def = task_def_real->mutable_kernel();
  auto mutable_context = kernel_def->mutable_context();
  mutable_context->set_args_format(args_format);
  tasks.push_back(*task_def_real);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
}

TEST_F(STEST_TaskBuilder, op_gentask_success){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  domi::TaskDef head_task;
  head_task.set_type(RT_MODEL_TASK_EVENT_WAIT);
  tasks.insert(tasks.begin(), head_task);
  domi::TaskDef tail_task;
  tail_task.set_type(RT_MODEL_TASK_EVENT_WAIT);
  tasks.push_back(tail_task);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, false, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
}

TEST_F(STEST_TaskBuilder, tiling_sink_gen_task_310p_suc){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;
  NamedAttrs attr;
  ge::AttrUtils::SetInt(attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  std::vector<NamedAttrs> attrs;
  attrs.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attrs);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, attrs);
  domi::TaskDef task_def_new;
  task_def_new.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_def = task_def_new.mutable_kernel_with_handle();
  kernel_def->set_args_size(66);
  string args(66, '1');
  kernel_def->set_args(args.data(), 66);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{t_ffts.non_tail}");
  tasks.push_back(task_def_new);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
}

TEST_F(STEST_TaskBuilder, tiling_sink_gen_task_310p_oppkernel_path_fail){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetInt(src_op, ge::ATTR_NAME_BINARY_SOURCE, 0);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;
  NamedAttrs attr;
  ge::AttrUtils::SetInt(attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  std::vector<NamedAttrs> attrs;
  attrs.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attrs);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, attrs);
  domi::TaskDef task_def_new;
  task_def_new.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_def = task_def_new.mutable_kernel_with_handle();
  kernel_def->set_args_size(66);
  string args(66, '1');
  kernel_def->set_args(args.data(), 66);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{t_ffts.non_tail}");
  tasks.push_back(task_def_new);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
}

TEST_F(STEST_TaskBuilder, tiling_sink_gen_task_310p_oppkernel_path){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetInt(src_op, ge::ATTR_NAME_BINARY_SOURCE, 0);
  const char_t *const kAscendHomePath = "ASCEND_HOME_PATH";
  std::string home_path = __FILE__;
  home_path = home_path.substr(0, home_path.rfind("/") + 1);
  // mmSetEnv(kAscendHomePath, home_path.c_str(), 1);
  int32_t err = 0;
  MM_SYS_SET_ENV(MM_ENV_ASCEND_HOME_PATH, home_path.c_str(), 1, err);
  const auto opp_latest_path = home_path + "/opp_latest/built-in/op_impl/ai_core/tbe/op_master_device/lib/";
  system(("mkdir -p " + opp_latest_path).c_str());
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;
  NamedAttrs attr;
  ge::AttrUtils::SetInt(attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  std::vector<NamedAttrs> attrs;
  attrs.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attrs);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, attrs);
  domi::TaskDef task_def_new;
  task_def_new.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_def = task_def_new.mutable_kernel_with_handle();
  kernel_def->set_args_size(66);
  string args(66, '1');
  kernel_def->set_args(args.data(), 66);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{t_ffts.non_tail}");
  tasks.push_back(task_def_new);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
  system(("rm -rf " + opp_latest_path).c_str());
  unsetenv(kAscendHomePath);
}

TEST_F(STEST_TaskBuilder, tiling_sink_gen_task_310p_add_tiling){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;
  NamedAttrs attr;
  ge::AttrUtils::SetInt(attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  std::vector<NamedAttrs> attrs;
  attrs.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attrs);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, attrs);
  domi::TaskDef task_def_new;
  task_def_new.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_def = task_def_new.mutable_kernel_with_handle();
  kernel_def->set_args_size(44);
  string args(44, '1');
  kernel_def->set_args(args.data(), 44);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}");
  tasks.push_back(task_def_new);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
}

TEST_F(STEST_TaskBuilder, tiling_sink_gen_task_multi_ops_path){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetInt(src_op, ge::ATTR_NAME_BINARY_SOURCE, 1);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetStr(src_op, OPS_PATH_NAME_PREFIX, "ops_test");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  (void)ge::AttrUtils::SetStr(src_op, OPS_PATH_NAME_PREFIX, "ops_test");
  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;

  (void)ge::AttrUtils::SetStr(src_op, OPS_PATH_NAME_PREFIX, "ops_test");
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  (void)ge::AttrUtils::SetStr(src_op, OPS_PATH_NAME_PREFIX, "");
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;
  NamedAttrs attr;
  ge::AttrUtils::SetInt(attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  std::vector<NamedAttrs> attrs;
  attrs.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attrs);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, attrs);
  domi::TaskDef task_def_new;
  task_def_new.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_def = task_def_new.mutable_kernel_with_handle();
  kernel_def->set_args_size(66);
  string args(66, '1');
  kernel_def->set_args(args.data(), 66);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{t_ffts.non_tail}");
  tasks.push_back(task_def_new);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
}

TEST_F(STEST_TaskBuilder, parse_impl_failed)
{
    string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_failed_2.json";
    TbeJsonFileParseImpl tbe_json_file_parse_impl;
    Status ret = tbe_json_file_parse_impl.Initialize(json_file_path);
    EXPECT_EQ(fe::SUCCESS, ret);
    int32_t block_dim = 0;
    ret = tbe_json_file_parse_impl.ParseJsonAttr(false, kKeyBlockDim, block_dim, block_dim);
    EXPECT_EQ(fe::FAILED, ret);
    string magic;
    ret = tbe_json_file_parse_impl.ParseJsonAttr(true, kKeyMagic, magic, magic);
    EXPECT_EQ(fe::FAILED, ret);
    uint32_t task_ratio = 0;
    ret = tbe_json_file_parse_impl.ParseJsonAttr(false, kKeyTaskRation, task_ratio, task_ratio);
    EXPECT_EQ(fe::FAILED, ret);
    vector<int64_t> compress_param_vec;
    ret = tbe_json_file_parse_impl.ParseJsonAttr(false, kKeyCompressParameters, compress_param_vec, compress_param_vec);
    EXPECT_EQ(fe::FAILED, ret);
    int64_t weight_repeat = 0;
    ret = tbe_json_file_parse_impl.ParseJsonAttr(false, kKeyWeightRepeat, weight_repeat, weight_repeat);
    EXPECT_EQ(fe::FAILED, ret);
    string file_name;
    std::vector<char> buffer;
    ret = tbe_json_file_parse_impl.ReadBytesFromBinaryFile(file_name, buffer);
    EXPECT_EQ(fe::FAILED, ret);
    string kernel_list_first;
    ret = tbe_json_file_parse_impl.ParseTvmKernelList(kernel_list_first);
    EXPECT_EQ(fe::SUCCESS, ret);
    vector<int64_t> tvm_workspace_sizes;
    vector<int64_t> tvm_workspace_types;
    ret = tbe_json_file_parse_impl.ParseTvmWorkSpace(tvm_workspace_sizes, tvm_workspace_types);
    EXPECT_EQ(fe::FAILED, ret);
    string meta_data;
    ret = tbe_json_file_parse_impl.ParseTvmMetaData(meta_data);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, parse_mix_impl_suc)
{
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_mix_ratio.json";
  TbeJsonFileParseImpl tbe_json_file_parse_impl;
  Status ret = tbe_json_file_parse_impl.Initialize(json_file_path);
  EXPECT_EQ(fe::SUCCESS, ret);
  int32_t cube_ratio;
  int32_t vector_ratio;
  bool dyn_ratio;
  ret = tbe_json_file_parse_impl.ParseTvmTaskRatio(cube_ratio, vector_ratio, dyn_ratio);
  EXPECT_EQ(fe::SUCCESS, ret);
  string kernel_list_first;
  TilingWithRatio tiling_ratio;
  ret = tbe_json_file_parse_impl.ParseTvmKernelList(kernel_list_first);
  EXPECT_EQ(fe::SUCCESS, ret);
  ret = tbe_json_file_parse_impl.ParseListTilingRatio(tiling_ratio);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, parse_vector_core_mix_impl_suc)
{
    string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_vector_core_mix_ratio.json";
    TbeJsonFileParseImpl tbe_json_file_parse_impl;
    Status ret = tbe_json_file_parse_impl.Initialize(json_file_path);
    EXPECT_EQ(fe::SUCCESS, ret);
    int32_t cube_ratio;
    int32_t vector_ratio;
    bool dyn_ratio;
    ret = tbe_json_file_parse_impl.ParseTvmTaskRatio(cube_ratio, vector_ratio, dyn_ratio);
    EXPECT_EQ(fe::SUCCESS, ret);
    string kernel_list_first;
    TilingWithRatio tiling_ratio;
    ret = tbe_json_file_parse_impl.ParseTvmKernelList(kernel_list_first);
    EXPECT_EQ(fe::SUCCESS, ret);
    ret = tbe_json_file_parse_impl.ParseListTilingRatio(tiling_ratio);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_TaskBuilder, tiling_sink_for_sk){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr src_op = std::make_shared<OpDesc>("FIA", "FIA");
  GeTensorDesc src_tensor_desc(GeShape({1, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({6, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1});
  src_op->SetWorkspaceBytes({1});
  src_op->AddInputDesc(src_tensor_desc);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_OUTPUT_INDEX, {1, 0});
  (void)ge::AttrUtils::SetListInt(src_op, TBE_OP_ATOMIC_WORKSPACE_INDEX, {1});
  OpExtGenTaskRegistry::GetInstance().names_to_register_func_.clear();

  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);

  IMPL_OP(FIA).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FIA");
  op_impl_func->calc_op_param = CalcParamKernelFunc;
  op_impl_func->gen_task = GenTaskKernelFunc;

  Status ret = aicore_ops_kernel_builder_ptr->CalcExtOpRunningParam(*src_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto workspace_bytes = src_op->GetWorkspaceBytes();
  EXPECT_EQ(workspace_bytes.size(), 1);
  EXPECT_EQ(workspace_bytes[0], 22);
  std::vector<domi::TaskDef> tasks;
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context_, tasks);
  EXPECT_NE(tasks.size(), 0);
  bool reg_flag = false;
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);
  // custom op
  ge::AttrUtils::SetInt(src_op, fe::FE_IMPLY_TYPE, 2);
  ret = fe::GenerateOpExtTask(*src_node, true, tasks, reg_flag);
  EXPECT_NE(tasks.size(), 0);

  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*src_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  ParamDef param;
  NamedAttrs attr;
  ge::AttrUtils::SetInt(attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  std::vector<NamedAttrs> attrs;
  attrs.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attrs);
  ge::AttrUtils::SetListNamedAttrs(src_op, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, attrs);
  domi::TaskDef task_def_new;
  task_def_new.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_def = task_def_new.mutable_kernel_with_handle();
  kernel_def->set_args_size(66);
  string args(66, '1');
  kernel_def->set_args(args.data(), 66);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{t_ffts.non_tail}");
  tasks.push_back(task_def_new);
  ret = GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
  EXPECT_NE(tasks.size(), 0);
  domi::TaskDef aicpu_task;
  ret = CreateTilingTaskSuperKernel(op_exe_res_ctx, aicpu_task, param);
  EXPECT_NE(ret, fe::FAILED);
  ret = GenerateTaskSuperKernel(op_exe_res_ctx, tasks, param);
  EXPECT_NE(ret, fe::FAILED);
}
