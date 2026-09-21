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
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "fe_llt_utils.h"
#include "common/util/op_info_util.h"
#include "common/fe_type_utils.h"
#define private public
#define protected public
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

#include "graph/ge_tensor.h"
#include "fusion_manager/fusion_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace testing;
using namespace fe;

using fe::FEOpsKernelInfoStore;
using ge::GeTensorDesc;
using ge::GeShape;
using ge::AttrUtils;
using ge::Format;
using ge::DataType;
using ge::ConstGeTensorDescPtr;
using ge::GeTensorDescPtr;
using ge::OpDescPtr;
using ge::OpDesc;
using fe::InputOrOutputInfoPtr;
using ge::GeAttrValue;
using std::vector;
using std::map;
using namespace ge;
using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;
using OpImplTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;

class FormatDtypeSelectorManagerFz3dUTest : public testing::Test{
 protected:
  static void SetUpTestCase() {
    cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
  }
  static void TearDownTestCase() {
    cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
  }
  // Some expensive resource shared by all tests.
  virtual void SetUp(){
    fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(AI_CORE_NAME);
    FEOpsStoreInfo tbe_builtin {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo",
        ""
    };
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr->Initialize(options);
    cout << "a test Set Up" << endl;
  }
  virtual void TearDown(){
    cout << "a test Tear Down" << endl;
    fe_ops_kernel_info_store_ptr->Finalize();

  }

 protected:
  static void CreateOneOpGraph(ComputeGraphPtr graph,  OpDescPtr op_desc_ptr) {
    graph->AddNode(op_desc_ptr);
  }
  static GeTensorDesc CreateTensorDesc(vector<int64_t> dim_data,  ge::Format format, ge::DataType data_type) {
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, format, data_type);
    data_desc.SetOriginFormat(format);
    data_desc.SetOriginShape(shape_data);
    data_desc.SetOriginDataType(data_type);
    return data_desc;
  }
 public:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
  const string EQUAL_ORIGIN_FORMAT="NCHW,NCHW,NHWC,NHWC,HWCN,HWCN,CHWN,CHWN,NDHWC,NDHWC,NCDHW,NCDHW,DHWCN,DHWCN,DHWNC,DHWNC,ND,ND";
  const string EQUAL_HEAVY_FORMAT=",NC1HWC0,NC1HWC0,C1HWNCoC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z,FRACTAL_NZ,FRACTAL_NZ";
};

// format are all NDHWC
// original x: 4d, y: 4d z:4d (axis is less than 5)
// they are not support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz_3d_1) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 8, 8};
  vector<int64_t> dim_data2 = {1, 32, 16, 16};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);
  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format = EQUAL_ORIGIN_FORMAT + ",NC1HWC0,NC1HWC0";
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all DHWCN
// original x: 4d, y: 4d z:4d (axis is less than 5)
// they are not support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz_3d_2) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 8, 8};
  vector<int64_t> dim_data2 = {1, 32, 16, 16};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_DHWCN, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, FORMAT_DHWCN, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_DHWCN, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all NDHWC
// original x: 5d, y: 5d z:5d (C of each tensor is 16 aligned)
// Axis N of input2 is not 16 aligned
// they do not support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_3) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 8, 8, 16};
  vector<int64_t> dim_data2 = {1, 32, 16, 16, 32};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}


// format are all NDHWC
// original x: 5d, y: 5d z:5d (C of each tensor is 16 aligned)
// Axis N is not 16 aligned
// they support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_4) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 8, 8, 16};
  vector<int64_t> dim_data2 = {128, 32, 16, 16, 32};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format = EQUAL_ORIGIN_FORMAT;
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all NDHWC
// original x: 5d, y: 5d z:5d (axis N and H need to BroadCast)
// they do not support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_5) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {32, 32, 15, 8, 33};
  vector<int64_t> dim_data2 = {16, 32, 8, 8, 33};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format =
      EQUAL_ORIGIN_FORMAT +
      ",NC1HWC0,NC1HWC0,FRACTAL_NZ,FRACTAL_NZ,NDC1HWC0,NDC1HWC0";
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all NDHWC
// original x: 5d, y: 5d z:5d (axis H and D need to BroadCast)
// they support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_6) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {32, 32, 15, 8, 33};
  vector<int64_t> dim_data2 = {32, 16, 8, 8, 33};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format =
      EQUAL_ORIGIN_FORMAT +
      ",NC1HWC0,NC1HWC0,C1HWNCoC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z,FRACTAL_NZ,FRACTAL_NZ,NDC1HWC0,NDC1HWC0,FRACTAL_Z_3D,FRACTAL_Z_3D";
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all NDHWC
// original x: 5d, y: 5d z:5d (axis N\D\H need to BroadCast)
// they do not support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_7) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {1, 16, 1, 8, 32};
  vector<int64_t> dim_data2 = {16, 32, 8, 8, 32};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, ge::FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, ge::FORMAT_NDHWC, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format =
      EQUAL_ORIGIN_FORMAT +
      ",NC1HWC0,NC1HWC0,FRACTAL_NZ,FRACTAL_NZ,NDC1HWC0,NDC1HWC0";
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all NCDHW
// original x: 5d, y: 5d z:5d (axis C need to BroadCast)
// they do not support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_8) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 32, 8, 8};
  vector<int64_t> dim_data2 = {16, 16, 32, 8, 8};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, ge::FORMAT_NCDHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, ge::FORMAT_NCDHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, ge::FORMAT_NCDHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format =
      EQUAL_ORIGIN_FORMAT +
      ",FRACTAL_NZ,FRACTAL_NZ";
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// format are all NCDHW
// original x: 5d, y: 1d z:5d (y is scalar)
// they  support fz3d
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_broadcast_selector_fz3d_9) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 32, 8, 8};
  vector<int64_t> dim_data2 = {1};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, ge::FORMAT_NCDHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y",
                            CreateTensorDesc(dim_data2, ge::FORMAT_NCDHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z",
                             CreateTensorDesc(dim_data1, ge::FORMAT_NCDHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format =
      EQUAL_ORIGIN_FORMAT +
      ",NC1HWC0,NC1HWC0,C1HWNCoC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z,NDC1HWC0,NDC1HWC0,FRACTAL_Z_3D,FRACTAL_Z_3D";
  string y_expect_format =
      EQUAL_ORIGIN_FORMAT + ",ND,ND,ND,ND,ND,ND,ND,ND,ND,ND";
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(y_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format,
            GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_reduce_selector_fz3d_1) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16, 16};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc(
      "y", CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag = false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_NDC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z_3D);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

// NCDHW
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_reduce_selector_fz3d_2) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 16, 32, 16, 16};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NCDHW, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc(
      "y", CreateTensorDesc(dim_data1, FORMAT_NCDHW, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag = false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_NDC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z_3D);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

/* shape of two tensor is not 5d */
TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_reduce_selector_fz3d_3) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  vector<int64_t> dim_data2 = {16, 8, 1, 1};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc(
      "y", CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_NDC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z_3D);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);
  }
}

TEST_F(FormatDtypeSelectorManagerFz3dUTest, op_reduce_selector_fz3d_4) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16, 16};
  vector<int64_t> dim_data2 = {16, 8, 1, 1, 19};
  op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc(
      "y", CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin",
                                                                                                        op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 4});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag = false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  uint32_t count = 0;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_NDC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    if (count == 0) {
      EXPECT_EQ(support_flag, false);
    } else {
      EXPECT_EQ(support_flag, false);
    }

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z_3D);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    count++;
  }
}
