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
#include <iostream>
#include <list>
#include "base/registry/op_impl_space_registry_v2.h"
#include "inc/ffts_type.h"
#include "common/sgt_slice_type.h"
#define private public
#define protected public
#include "fe_llt_utils.h"
#include "ops_kernel_builder/aicore_ops_kernel_builder.h"
#include "ops_kernel_builder/task_builder/task_builder.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "common/constants_define.h"
#include "common/aicore_util_attr_define.h"
#include "register/op_ext_gentask_registry.h"
using namespace std;
using namespace fe;
using namespace ge;
using AICoreOpsKernelBuilderPtr =  shared_ptr<AICoreOpsKernelBuilder>;
const int64_t SET_SIZE = 10000;

class AICoreOpsKernelBuilderTest : public testing::Test{
 protected:
  static void SetUpTestCase() {
    cout << "AICoreOpsKernelBuilderTest SetUP" << endl;
  }
  static void TearDownTestCase() {
    cout << "AICoreOpsKernelBuilderTest SetUP" << endl;
  }
  // Some expensive resource shared by all tests.
  virtual void SetUp(){
    auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
    aicore_ops_kernel_builder_ptr = make_shared<AICoreOpsKernelBuilder>();
    std::map<std::string, std::string> options;
    aicore_ops_kernel_builder_ptr->Initialize(options);
  }
  virtual void TearDown(){
    cout << "a test Tear Down" << endl;
    aicore_ops_kernel_builder_ptr->Finalize();

  }

  OpDescPtr GreateOpDesc() {
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);
    shared_ptr<ge::GeTensorDesc> tensor_desc_ptr = make_shared<ge::GeTensorDesc>();
    tensor_desc_ptr->SetShape(shape);
    tensor_desc_ptr->SetDataType(ge::DT_FLOAT);
    tensor_desc_ptr->SetFormat(ge::FORMAT_NCHW);

    OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("test_op_desc", "conv");
    op_desc_ptr->SetId(123456);
    op_desc_ptr->AddInputDesc(tensor_desc_ptr->Clone());
    op_desc_ptr->AddOutputDesc(tensor_desc_ptr->Clone());

    return op_desc_ptr;
  }

  static void SetOpDecSize(NodePtr& node) {
		OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
		for (int i = 0; i < node->GetOpDesc()->GetAllInputsDesc().size(); i++) {
			ge::GeTensorDesc tensor = node->GetOpDesc()->GetAllInputsDesc().at(i);
			ge::TensorUtils::SetSize(tensor, SET_SIZE);
			node->GetOpDesc()->UpdateInputDesc(i, tensor);
		}
		OpDesc::Vistor<GeTensorDesc> tensorsOutput = node->GetOpDesc()->GetAllOutputsDesc();
		for (int i = 0; i < tensorsOutput.size(); i++) {
			ge::GeTensorDesc tensorOutput = tensorsOutput.at(i);
			ge::TensorUtils::SetSize(tensorOutput, SET_SIZE);
			node->GetOpDesc()->UpdateOutputDesc(i, tensorOutput);
		}
	}

  static RunContext CreateContext()
  {
    RunContext context;
    context.dataMemSize = 101;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.weightMemSize = 200;
    context.weightMemBase = (uint8_t *) (intptr_t) 1101;
    context.weightsBuffer = Buffer(20);

    return context;
  }

 public:
  AICoreOpsKernelBuilderPtr aicore_ops_kernel_builder_ptr;
};

namespace fe {
  extern void ProcDfxBufferSize(const ge::OpDescPtr op_desc);
}
TEST_F(AICoreOpsKernelBuilderTest, dfx_buffer_size_test)
{
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_graph");
  OpDescPtr op_desc_ptr = GreateOpDesc();
  NodePtr node = graph->AddNode(op_desc_ptr);

  int64_t buffer_size = 2345;
  ge::AttrUtils::SetInt(op_desc_ptr, kOpDfxBufferSize, buffer_size);
  ProcDfxBufferSize(op_desc_ptr);
  auto new_workspaces = op_desc_ptr->GetWorkspaceBytes();
  EXPECT_EQ(new_workspaces.size(), 0);

  op_desc_ptr->SetWorkspaceBytes({1111, 2222});
  ProcDfxBufferSize(op_desc_ptr);
  new_workspaces = op_desc_ptr->GetWorkspaceBytes();
  EXPECT_EQ(new_workspaces[0], 3456);
}

TEST_F(AICoreOpsKernelBuilderTest, calcoprunningparam_success_1){
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_graph");
  OpDescPtr op_desc_ptr = GreateOpDesc();
  NodePtr node = graph->AddNode(op_desc_ptr);

  Status status = aicore_ops_kernel_builder_ptr->CalcOpRunningParam(*node);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(AICoreOpsKernelBuilderTest, calcoprunningparam_success_2){
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_graph");
  OpDescPtr op_desc_ptr = GreateOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "ROIPooling");
  ge::AttrUtils::SetStr(op_desc_ptr, "_unregst_oppath", "/usr/local");
  NodePtr node = graph->AddNode(op_desc_ptr);

  Status status = aicore_ops_kernel_builder_ptr->CalcOpRunningParam(*node);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(AICoreOpsKernelBuilderTest, calc_op_running_param_succ)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);

  Status ret = aicore_ops_kernel_builder_ptr->CalcOpRunningParam(*src_node.get());
  int64_t data_size = 0;
  ge::GeTensorDesc tensor_desc = src_op->GetInputDesc(0);
  ge::TensorUtils::GetSize(tensor_desc, data_size);

  EXPECT_EQ(2457632, data_size);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(AICoreOpsKernelBuilderTest, generate_task_failed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({5, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({5, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  (void)ge::AttrUtils::SetBool(src_op, kTypeFFTSPlus, true);
  auto src_node = graph->AddNode(src_op);
  SetOpDecSize(src_node);
  ge::RunContext context = CreateContext();
  std::vector<domi::TaskDef> tasks;
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context, tasks);
  EXPECT_EQ(fe::FAILED, ret);
}

ge::Status TestOpExtGenTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  domi::TaskDef tmp_task;
  tasks.emplace_back(tmp_task);
  return ge::SUCCESS;
}

ge::Status TestOpExtGenTaskFail(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  return ge::FAILED;
}

TEST_F(AICoreOpsKernelBuilderTest, generate_task_with_memset)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({5, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({5, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  auto src_node = graph->AddNode(src_op);
  FillNodeParaType(src_node);
  auto memset_node = graph->AddNode(src_op);
  src_node->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  auto owner_graph = src_node->GetOwnerComputeGraph();
  owner_graph->SetGraphUnknownFlag(true);
  SetOpDecSize(src_node);
  ge::RunContext context = CreateContext();
  REGISTER_NODE_EXT_GENTASK("A", TestOpExtGenTask);
  std::vector<domi::TaskDef> tasks;
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(AICoreOpsKernelBuilderTest, generate_task_suc3)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({5, 2, 3, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({5, 11, 3, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetInt(src_op, FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(src_op, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  auto src_node = graph->AddNode(src_op);
  FillNodeParaType(src_node);
  SetOpDecSize(src_node);
  ge::RunContext context = CreateContext();
  std::vector<domi::TaskDef> tasks;
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
  tasks.clear();
  REGISTER_NODE_EXT_GENTASK("A", TestOpExtGenTaskFail);
  ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node, context, tasks);
  EXPECT_NE(fe::SUCCESS, ret);
}

TEST_F(AICoreOpsKernelBuilderTest, GenerateTask_mix_l2_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({10}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(src_tensor_desc, 64);
  src_tensor_desc.SetOriginShape(GeShape({10}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::RunContext context = CreateContext();
  std::vector<domi::TaskDef> tasks;
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_ALIAS_ENGINE_NAME, "ffts_plus");
  string bin_magic = "FFTS_BINARY_MAGIC_ELF_MIX_AIC";
  (void)ge::AttrUtils::SetStr(src_op, "tvm_magic", bin_magic);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(src_op, "compile_info_json", json_str);
  src_op->AppendIrInput("__input0", ge::kIrInputRequired);
  std::vector<uint32_t> input_type_list = {0};
  src_op->AppendIrOutput("__output0", ge::kIrOutputRequired);
  (void)ge::AttrUtils::SetListInt(src_op, kInputParaTypeList, input_type_list);
  std::vector<uint32_t> output_type_list = {0};
  (void)ge::AttrUtils::SetListInt(src_op, kOutputParaTypeList, output_type_list);
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node.get(), context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(AICoreOpsKernelBuilderTest, GenManualMixAICAIVCtxDef_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, 6, 6, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(src_tensor_desc, 4096);
  src_tensor_desc.SetOriginShape(GeShape({10}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::RunContext context = CreateContext();
  std::vector<domi::TaskDef> tasks;
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
(void)ge::AttrUtils::SetInt(src_op, kOpKernelAllInputSize, 2);
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ffts::ThreadMode::MANUAL_THREAD);
  src_op->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetStr(src_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  vector<uint32_t> input_type_list = {1, 1};
  (void)ge::AttrUtils::SetListInt(src_op, kInputParaTypeList, input_type_list);
  (void)ge::AttrUtils::SetStr(src_op, "optionalInputMode", "gen_placeholder");
  vector<int64_t> workspace_sizes = {12, 44};
  vector<int64_t> workspace_offsets = {12, 44};
  src_op->SetWorkspaceBytes(workspace_sizes);
  src_op->SetWorkspace(workspace_offsets);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixAIC);
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node.get(), context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(AICoreOpsKernelBuilderTest, GenDynamicAICAIVCtxDef_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("A", "A");
  GeTensorDesc src_tensor_desc(GeShape({6, -1, -1, 3, 2}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(src_tensor_desc, 64);
  src_tensor_desc.SetOriginShape(GeShape({10}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  auto src_node = graph->AddNode(src_op);
  ge::RunContext context = CreateContext();
  std::vector<domi::TaskDef> tasks;
  (void)ge::AttrUtils::SetBool(src_op, fe::kTypeFFTSPlus, true);
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->thread_mode = static_cast<uint32_t>(ffts::ThreadMode::AUTO_THREAD);
  src_op->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetBool(src_op, "OwnerGraphIsUnknown", true);
  (void)ge::AttrUtils::SetStr(src_op, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  Status ret = aicore_ops_kernel_builder_ptr->GenerateTask(*src_node.get(), context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
}
