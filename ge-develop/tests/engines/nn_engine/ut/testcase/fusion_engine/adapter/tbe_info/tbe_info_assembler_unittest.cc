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
#include "graph/node.h"
#include "graph/op_desc.h"
#include <string>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "ge/ge_api_types.h"
#include "common/util/op_info_util.h"
#include "ops_store/op_kernel_info.h"
#include "common/fe_op_info_common.h"
#include "common/fe_inner_attr_define.h"
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "common/util/json_util.h"
#include "ge_local_context.h"
#include "graph_constructor/fe_llt_utils.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;
using namespace te;

class UTEST_TbeInfoAssembler : public testing::Test
{
protected:
  static NodePtr CreateConstDependNode(ComputeGraphPtr graph, ge::DataType dType) {
    vector<int64_t> dim = {1,2,3,4};
    ge::GeShape shape(dim);
    ge::GeTensorDesc tensorDesc(shape, ge::FORMAT_NCHW, dType);
    OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr const_desc = std::make_shared<OpDesc>("const", "Const");

    op_desc->AddInputDesc("x", tensorDesc);
    op_desc->AddInputDesc("y", tensorDesc);
    op_desc->AddOutputDesc("z", tensorDesc);
    const_desc->AddOutputDesc("z", tensorDesc);

    ge::GeTensorPtr constTensor = std::make_shared<ge::GeTensor>(tensorDesc);
    if (dType == ge::DT_INT32) {
      vector<int32_t> data_value(24, 4);
      constTensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), data_value.size() * sizeof(int32_t));
    } else if (dType == ge::DT_FLOAT16) {
      vector<int16_t> data_value(24, 4);
      constTensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), data_value.size() * sizeof(int16_t));
    } else {
      vector<float> data_value(24, 4.3);
      constTensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), data_value.size() * sizeof(float));
    }
    OpDescUtils::SetWeights(const_desc, constTensor);

    NodePtr relu_node = graph->AddNode(op_desc);
    NodePtr const_node = graph->AddNode(const_desc);
    ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(1));
    return relu_node;
  }
};

TEST_F(UTEST_TbeInfoAssembler, Test_FeedFlagInt64ToTbeOpInfo){
  std::string op_name    = "conv";
  std::string op_module   = "";
  std::string op_type     = "tbe";
  std::string core_type   = "AIcoreEngine";

  vector<int64_t> dim_data = {1024, 2, 1024, 1024};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NCHW, DT_INT2);
  vector<int64_t> dim_weight = {1024, 2, 1024, 1024};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);

  vector<int64_t> dim_out = {1024, 2, 1024, 2};
  GeShape shape_out(dim_out);
  GeTensorDesc OutDesc(shape_out);

  OpDescPtr conv_op = std::make_shared<OpDesc>("Conv", "conv");
  conv_op->AddInputDesc("x", data_desc);
  conv_op->AddInputDesc("w1", weight_desc);
  conv_op->AddOutputDesc("y", OutDesc);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node_ptr = graph->AddNode(conv_op);
  Node* node = node_ptr.get();

  TbeInfoAssembler tbe;
  TbeOpInfo op_info(op_name, op_module, op_type, core_type);
  Status ret = tbe.FeedFlagInt64ToTbeOpInfo(node, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
  bool flag = false;
  auto op_desc_ptr = node->GetOpDesc();
  ret = tbe.JudgeShapeToSetFlag(op_desc_ptr, true, op_info, flag);
  EXPECT_EQ(fe::SUCCESS, ret);
  ge::AttrUtils::SetBool(*(conv_op.get()), ATTR_NAME_UNKNOWN_SHAPE, true);
  ret = tbe.FeedFlagInt64ToTbeOpInfo(node, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_info.GetFlagUseInt64(), true);
}

TEST_F(UTEST_TbeInfoAssembler, get_options_01) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  PlatformInfoManager::Instance().opti_compilation_infos_.SetAICoreNum(10);
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.Initialize();
  putenv("NPU_COLLECT_PATH=x");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  auto options = tbe_info_assembler.GetAllOptionsForTBE(*node->GetOpDesc(), "AiCoreEngine", nullptr);
  EXPECT_EQ(options[ge::OP_DEBUG_LEVEL], "");

  putenv("NPU_COLLECT_PATH=");
  options = tbe_info_assembler.GetAllOptionsForTBE(*node->GetOpDesc(), "AiCoreEngine", nullptr);
  EXPECT_EQ(options[ge::OP_DEBUG_LEVEL], "");
}

TEST_F(UTEST_TbeInfoAssembler, get_options_02) {
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");
  PlatformInfoManager::Instance().opti_compilation_infos_.SetAICoreNum(10);
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.Initialize();
  putenv("NPU_COLLECT_PATH=x");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  OpKernelInfoPtr op_kernel_ptr = std::make_shared<OpKernelInfo>("Relu");
  op_kernel_ptr->op_flag_vec_[3] = true;
  auto options = tbe_info_assembler.GetAllOptionsForTBE(*node->GetOpDesc(), "AiCoreEngine", op_kernel_ptr);
  EXPECT_EQ(options[ge::OP_DEBUG_LEVEL], "");

  putenv("NPU_COLLECT_PATH=");
  options = tbe_info_assembler.GetAllOptionsForTBE(*node->GetOpDesc(), "AiCoreEngine", nullptr);
  EXPECT_NE(options[ge::OP_DEBUG_LEVEL], "2");
}

TEST_F(UTEST_TbeInfoAssembler, GetAllOptions_SOFTSYNC_OP) {
  SetPlatformSocVersion("Ascend310P3");
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.Initialize();
  auto geGraphOptions = ge::GetThreadLocalContext().GetAllGraphOptions();
  geGraphOptions[ge::AICORE_NUM] = "";
  geGraphOptions["ge.hardwareInfo"] = "";
  ge::GetThreadLocalContext().SetGraphOption(geGraphOptions);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  OpKernelInfoPtr op_kernel_ptr = std::make_shared<OpKernelInfo>("ReduceSumD");
  op_kernel_ptr->op_param_vec_.fill(1);
  op_kernel_ptr->op_flag_vec_.fill(true);

  std::map<string, string> geOptions;
  geOptions.emplace(ge::VIRTUAL_TYPE, "1");
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.InitConfigParamFromOptions(geOptions);
  config.hardware_info_map_.clear();
  auto options = tbe_info_assembler.GetAllOptionsForTBE(*node->GetOpDesc(), "AiCoreEngine", op_kernel_ptr);
  EXPECT_EQ(options[ge::AICORE_NUM], "8");
}

TEST_F(UTEST_TbeInfoAssembler, GetAllOptions_SOFTSYNC_OP_RNN) {
  SetPlatformSocVersion("Ascend310P3");
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.Initialize();
  auto geGraphOptions = ge::GetThreadLocalContext().GetAllGraphOptions();
  geGraphOptions[ge::AICORE_NUM] = "";
  geGraphOptions["ge.hardwareInfo"] = "";
  ge::GetThreadLocalContext().SetGraphOption(geGraphOptions);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  OpKernelInfoPtr op_kernel_ptr = std::make_shared<OpKernelInfo>("DynamicRNN");
  op_kernel_ptr->op_param_vec_.fill(2);
  op_kernel_ptr->op_flag_vec_.fill(true);

  std::map<string, string> geOptions;
  geOptions.emplace(ge::VIRTUAL_TYPE, "1");
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.InitConfigParamFromOptions(geOptions);
  config.hardware_info_map_.clear();
  auto options = tbe_info_assembler.GetAllOptionsForTBE(*node->GetOpDesc(), "AiCoreEngine", op_kernel_ptr);
  EXPECT_EQ(options[ge::AICORE_NUM], "1");
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> shape = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", shape, "float", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 0, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_OPTIONAL;
  std::vector<int64_t> shape = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", shape, "float", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 0, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT);
  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> shape = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", shape, "float", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 1, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_FLOAT16);
  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> shape = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", shape, "float16", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 1, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_5) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateConstDependNode(graph, ge::DT_INT32);
  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> shape = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", shape, "int32", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 1, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_6) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  ge::GeTensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_INT64);
  op_desc->AddInputDesc(tensorDesc);
  op_desc->AddOutputDesc(tensorDesc);

  ge::GeTensorPtr constTensor = std::make_shared<ge::GeTensor>(tensorDesc);
  vector<int64_t> data_value(24, 64);
  constTensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), data_value.size() * sizeof(int64_t));
  ge::AttrUtils::SetTensor(op_desc->MutableInputDesc(0), ge::ATTR_NAME_VALUE, constTensor);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> dim_vec = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", dim_vec, "int64", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 0, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_7) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  ge::GeTensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_BOOL);
  op_desc->AddInputDesc(tensorDesc);
  op_desc->AddOutputDesc(tensorDesc);

  ge::GeTensorPtr constTensor = std::make_shared<ge::GeTensor>(tensorDesc);
  vector<int64_t> data_value(24, 64);
  constTensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), data_value.size() * sizeof(int64_t));
  ge::AttrUtils::SetTensor(op_desc->MutableInputDesc(0), ge::ATTR_NAME_VALUE, constTensor);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> dim_vec = {1,2,3,4};
  te::TbeOpTensor op_tensor("x", dim_vec, "int64", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 0, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, set_tensor_constvalue_8) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,1};
  ge::GeShape shape(dim);
  ge::GeTensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_BF16);
  op_desc->AddInputDesc(tensorDesc);
  op_desc->AddOutputDesc(tensorDesc);

  ge::GeTensorPtr constTensor = std::make_shared<ge::GeTensor>(tensorDesc);
  vector<float> fp_data_value = {1.3, 0.64, 54.983, -234.86, -0.378, -1.345};
  vector<uint16_t> data_value;
  for (float val : fp_data_value) {
    union {
      uint32_t hex;
      float val;
    } trans_val;
    trans_val.val = val;
    uint16_t int_val = trans_val.hex >> 16;
    data_value.push_back(int_val);
  }

  constTensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), data_value.size() * sizeof(uint16_t));
  ge::AttrUtils::SetTensor(op_desc->MutableInputDesc(0), ge::ATTR_NAME_VALUE, constTensor);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  InputOrOutputInfoPtr tensor_info_ptr = std::make_shared<InputOrOutputInfo>("input");
  tensor_info_ptr->op_const_value_depend_ = CONST_REQUIRED;
  std::vector<int64_t> dim_vec = {1,2,3,1};
  te::TbeOpTensor op_tensor("x", dim_vec, "int16", "NCHW");
  TbeInfoAssembler tbe_info_assembler;
  Status ret = tbe_info_assembler.SetTensorConstValue(node.get(), 0, tensor_info_ptr, op_tensor);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeInfoAssembler, feed_l1_input_tensor1) {
  ToOpStructPtr l1_info = std::make_shared<ToOpStruct>();
  l1_info->slice_output_shape = {{2,2,2,2}};
  l1_info->slice_input_offset = {{2,2,2,2}};
  l1_info->slice_output_offset = {{2,2,2,2}};
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);

  IndexNameMap input_idx_name_map;
  uint32_t index_in_opdesc;
  te::TbeOpTensor input_tensor;
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.FeedL1InputTensor(l1_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
  EXPECT_EQ(input_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_l1_input_tensor2) {
  ToOpStructPtr l1_info = std::make_shared<ToOpStruct>();
  l1_info->slice_input_shape = {{2,2,2,2}};
  l1_info->slice_output_shape = {{2,2,2,2}};
  l1_info->slice_output_offset = {{2,2,2,2}};
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);
  IndexNameMap input_idx_name_map;
  uint32_t index_in_opdesc;
  te::TbeOpTensor input_tensor;
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.FeedL1InputTensor(l1_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
  EXPECT_EQ(input_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_l1_input_tensor3) {
  ToOpStructPtr l1_info = std::make_shared<ToOpStruct>();
  l1_info->slice_input_shape = {{2,2,2,2}};
  l1_info->slice_output_shape = {{2,2,2,2}};
  l1_info->slice_input_offset = {{2,2,2,2}};
  l1_info->slice_output_offset = {{2,2,2,2}};
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);
  vector<int64_t> inputOffset = {1};
  op_desc->SetInputOffset(inputOffset);
  IndexNameMap input_idx_name_map;
  uint32_t index_in_opdesc = 1;
  te::TbeOpTensor input_tensor;
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.FeedL1InputTensor(l1_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
  EXPECT_EQ(input_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_l2_input_tensor1) {
  ToOpStructPtr l2_info = std::make_shared<ToOpStruct>();
  l2_info->slice_output_shape = {{2,2,2,2}};
  l2_info->slice_input_offset = {{2,2,2,2}};
  l2_info->slice_output_offset = {{2,2,2,2}};
  ge::OpDescPtr op_desc;
  IndexNameMap input_idx_name_map;
  uint32_t index_in_opdesc;
  te::TbeOpTensor input_tensor;
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.FeedL2InputTensor(l2_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
  EXPECT_EQ(input_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_l2_input_tensor2) {
 ToOpStructPtr l2_info = std::make_shared<ToOpStruct>();
  l2_info->slice_input_shape = {{2,2,2,2}};
  l2_info->slice_output_shape = {{2,2,2,2}};
  l2_info->slice_output_offset = {{2,2,2,2}};
  ge::OpDescPtr op_desc;
  IndexNameMap input_idx_name_map;
  uint32_t index_in_opdesc;
  te::TbeOpTensor input_tensor;
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.FeedL2InputTensor(l2_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
  EXPECT_EQ(input_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_l2_input_tensor3) {
  ToOpStructPtr l2_info = std::make_shared<ToOpStruct>();
  l2_info->slice_input_shape = {{2,2,2,2}};
  l2_info->slice_output_shape = {{2,2,2,2}};
  l2_info->slice_input_offset = {{2,2,2,2}};
  l2_info->slice_output_offset = {{2,2,2,2}};
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);
  vector<int64_t> inputOffset = {1};
  op_desc->SetInputOffset(inputOffset);
  uint32_t index_in_opdesc = 2;
  IndexNameMap input_idx_name_map;
  te::TbeOpTensor input_tensor;
  TbeInfoAssembler tbe_info_assembler;
  tbe_info_assembler.FeedL2InputTensor(l2_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
  EXPECT_EQ(input_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_fusion_output_tensor1) {
  ToOpStructPtr fusion_info = std::make_shared<ToOpStruct>();
  fusion_info->slice_input_shape = {{2,2,2,2}};
  fusion_info->slice_input_offset = {{2,2,2,2}};
  fusion_info->slice_output_offset = {{2,2,2,2}};
  uint32_t index_in_opdesc;
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);
  vector<int64_t> inputOffset = {1};
  op_desc->SetInputOffset(inputOffset);
  IndexNameMap output_idx_name_map;
  TbeInfoAssembler tbe_info_assembler;
  te::TbeOpTensor output_tensor;
  tbe_info_assembler.FeedFusionOutputTensor(fusion_info, op_desc, output_idx_name_map, index_in_opdesc, output_tensor);
  EXPECT_EQ(output_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_fusion_output_tensor2) {
  ToOpStructPtr fusion_info = std::make_shared<ToOpStruct>();
  fusion_info->slice_input_shape = {{2,2,2,2}};
  fusion_info->slice_output_shape = {{2,2,2,2}};
  fusion_info->slice_input_offset = {{2,2,2,2}};
  uint32_t index_in_opdesc;
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);
  vector<int64_t> inputOffset = {1};
  op_desc->SetInputOffset(inputOffset);
  IndexNameMap output_idx_name_map;
  TbeInfoAssembler tbe_info_assembler;
  te::TbeOpTensor output_tensor;
  tbe_info_assembler.FeedFusionOutputTensor(fusion_info, op_desc, output_idx_name_map, index_in_opdesc, output_tensor);
  EXPECT_EQ(output_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, feed_fusion_output_tensor3) {
  ToOpStructPtr fusion_info = std::make_shared<ToOpStruct>();
  fusion_info->slice_input_shape = {{2,2,2,2}};
  fusion_info->slice_output_shape = {{2,2,2,2}};
  fusion_info->slice_input_offset = {{2,2,2,2}};
  fusion_info->slice_output_offset = {{2,2,2,2}};
  uint32_t index_in_opdesc = 1;
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW);
  op_desc->AddInputDesc(out_desc);
  op_desc->AddOutputDesc(out_desc);
  vector<int64_t> outputOffset = {1};
  op_desc->SetOutputOffset(outputOffset);
  IndexNameMap output_idx_name_map;
  TbeInfoAssembler tbe_info_assembler;
  te::TbeOpTensor output_tensor;
  tbe_info_assembler.FeedFusionOutputTensor(fusion_info, op_desc, output_idx_name_map, index_in_opdesc, output_tensor);
  EXPECT_EQ(output_idx_name_map.size(), 0);
}

TEST_F(UTEST_TbeInfoAssembler, getOutputInplaceAttr) {
  TbeInfoAssembler tbe_info_assembler;
  vector<vector<int64_t>> output_inplace ={{0, 0}, {0,1}};
  std::map<size_t, std::pair<size_t, size_t>> output_ir_real_index = {{0, {0,1}}};
  std::map<size_t, std::pair<size_t, size_t>> input_ir_real_index = {{0, {0,1}}, {1, {1, 1}}};
  vector<vector<int64_t>> real_output_inplace={};
  bool ret = tbe_info_assembler.SetOutputRealIndexInplaceAttr(output_inplace, input_ir_real_index, output_ir_real_index, real_output_inplace);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(real_output_inplace, output_inplace);
  output_ir_real_index = {{1, {0,1}}};
  ret = tbe_info_assembler.SetOutputRealIndexInplaceAttr(output_inplace, input_ir_real_index, output_ir_real_index, real_output_inplace);
  EXPECT_EQ(ret, false);
  output_ir_real_index = {{0, {0,1}}};
  input_ir_real_index = {{2, {0,1}}};
  ret = tbe_info_assembler.SetOutputRealIndexInplaceAttr(output_inplace, input_ir_real_index, output_ir_real_index, real_output_inplace);
  EXPECT_EQ(ret, false);
}

TEST_F(UTEST_TbeInfoAssembler, set_input_tensor_base_info) {
  TbeInfoAssembler tbe_info_assembler;
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_BF16);
  op_desc->AddInputDesc(out_desc);
  te::TbeOpTensor output_tensor;
  tbe_info_assembler.SetInputTensorBaseInfo(op_desc, 0, output_tensor);
}

TEST_F(UTEST_TbeInfoAssembler, set_input_tensor_base_info01) {
  TbeInfoAssembler tbe_info_assembler;
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(out_desc);
  te::TbeOpTensor output_tensor;
  tbe_info_assembler.SetInputTensorBaseInfo(op_desc, 0, output_tensor);
}

TEST_F(UTEST_TbeInfoAssembler, test_get_op_input_l1_attr) {
  TbeInfoAssembler tbe_info_assembler;
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {1,2,3,4};
  ge::GeShape shape(dim);
  GeTensorDesc out_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(out_desc);
  std::vector<int64_t> op_input_l1_flag_ori = {1, 1, 1};
  ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OP_INPUT_L1_FLAG, op_input_l1_flag_ori);
  std::vector<int64_t> op_input_l1_flag;
  std::vector<int64_t> op_input_l1_addr;
  std::vector<int64_t> op_input_l1_valid_size;
  tbe_info_assembler.GetOpInputL1Attr(op_desc, op_input_l1_flag, op_input_l1_addr, op_input_l1_valid_size);
  std::vector<int64_t> op_input_l1_addr_ori = {1, 1, 1};
  ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OP_INPUT_L1_ADDR, op_input_l1_addr_ori);
  tbe_info_assembler.GetOpInputL1Attr(op_desc, op_input_l1_flag, op_input_l1_addr, op_input_l1_valid_size);
}