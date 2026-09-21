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
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "common/fe_type_utils.h"
#include "common/util/op_info_util.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "fusion_manager/fusion_manager.h"
#include "graph/ge_tensor.h"

#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace testing;
using namespace fe;

using fe::FEOpsKernelInfoStore;
using fe::InputOrOutputInfoPtr;
using ge::AttrUtils;
using ge::ConstGeTensorDescPtr;
using ge::DataType;
using ge::Format;
using ge::GeAttrValue;
using ge::GeShape;
using ge::GeTensorDesc;
using ge::GeTensorDescPtr;
using ge::OpDesc;
using ge::OpDescPtr;
using std::map;
using std::vector;
using namespace ge;

using OpImplTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;

class FormatSelectorReduceNzUTest : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "FEOpsKernelInfoStoreTest SetUP" << endl; }
  static void TearDownTestCase() { cout << "FEOpsKernelInfoStoreTest SetUP" << endl; }
  // Some expensive resource shared by all tests.
  virtual void SetUp() {
    FEOpsStoreInfo tbe_builtin{
        6, "tbe-builtin", EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo", ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    std::map<std::string, std::string> options;
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
    cout << "a test Set Up" << endl;
  }
  virtual void TearDown() {
    cout << "a test Tear Down" << endl;
  }

 protected:
  static void CreateOneOpGraph(ComputeGraphPtr graph, OpDescPtr op_desc_ptr) { graph->AddNode(op_desc_ptr); }
  static GeTensorDesc CreateTensorDesc(vector<int64_t> dim_data, ge::Format format, ge::DataType data_type) {
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, format, data_type);
    data_desc.SetOriginFormat(format);
    data_desc.SetOriginShape(shape_data);
    data_desc.SetOriginDataType(data_type);
    return data_desc;
  }

 public:
  const string EQUAL_ORIGIN_FORMAT =
      "NCHW,NCHW,NHWC,NHWC,HWCN,HWCN,CHWN,CHWN,NDHWC,NDHWC,NCDHW,NCDHW,DHWCN,DHWCN,DHWNC,DHWNC,ND,ND";
};

TEST_F(FormatSelectorReduceNzUTest, nz2nz_success) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 15, 15};
  vector<int64_t> dim_data2 = {1, 1, 15, 15};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, false);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 1});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  FormatDtypeInfo format_dtype_info;
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);

  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));
  string expect_input_format = EQUAL_ORIGIN_FORMAT + ",NC1HWC0,FRACTAL_NZ,FRACTAL_NZ";
  string expect_output_format = EQUAL_ORIGIN_FORMAT + ",ND,FRACTAL_NZ,FRACTAL_NZ";

  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}

TEST_F(FormatSelectorReduceNzUTest, nz2nd_success_1) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {2, 16, 32};
  vector<int64_t> dim_data2 = {1, 1, 1};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_INT8));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 1, 2});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  FormatDtypeInfo format_dtype_info;
  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));
  string expect_input_format = EQUAL_ORIGIN_FORMAT + ",FRACTAL_NZ,FRACTAL_NZ";
  string expect_output_format = EQUAL_ORIGIN_FORMAT + ",ND,ND";

  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}

TEST_F(FormatSelectorReduceNzUTest, nz2nd_success_2) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {2, 16, 32};
  vector<int64_t> dim_data2 = {2, 1, 1};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_INT8));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {1, 2});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  FormatDtypeInfo format_dtype_info;
  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));
  string expect_input_format = EQUAL_ORIGIN_FORMAT + ",FRACTAL_NZ,FRACTAL_NZ";
  string expect_output_format = EQUAL_ORIGIN_FORMAT + ",ND,ND";
  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}

TEST_F(FormatSelectorReduceNzUTest, nz2nd_last_checkfailed_1) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {2, 16, 31};
  vector<int64_t> dim_data2 = {1, 16, 1};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_INT8));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 2});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  FormatDtypeInfo format_dtype_info;
  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));
  string expect_input_format = EQUAL_ORIGIN_FORMAT;
  string expect_output_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}

TEST_F(FormatSelectorReduceNzUTest, nz2nd_last_checkfailed_2) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {2, 16, 31};
  vector<int64_t> dim_data2 = {1, 1, 1};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_INT8));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  FormatDtypeInfo format_dtype_info;
  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));
  string expect_input_format = EQUAL_ORIGIN_FORMAT;
  string expect_output_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}

TEST_F(FormatSelectorReduceNzUTest, nz2nd_lastbutone_checkfailed_1) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {2, 15, 32};
  vector<int64_t> dim_data2 = {1, 1, 32};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_INT8));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 1});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  FormatDtypeInfo format_dtype_info;
  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));

  string expect_input_format = EQUAL_ORIGIN_FORMAT;
  string expect_output_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}

TEST_F(FormatSelectorReduceNzUTest, nz2nd_lastbutone_checkfailed_2) {
  // 1. create ops
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {2, 15, 32};
  vector<int64_t> dim_data2 = {2, 1, 32};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_INT8));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {1});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // 2. call functions
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  FormatDtypeInfo format_dtype_info;
  Status result =
      format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node, false, format_dtype_info);

  // 3. check result
  EXPECT_EQ(fe::SUCCESS, result);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  string input_formats = GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName()));
  string output_formats = GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName()));

  string expect_input_format = EQUAL_ORIGIN_FORMAT;
  string expect_output_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(expect_input_format, input_formats);
  EXPECT_EQ(expect_output_format, output_formats);
}
