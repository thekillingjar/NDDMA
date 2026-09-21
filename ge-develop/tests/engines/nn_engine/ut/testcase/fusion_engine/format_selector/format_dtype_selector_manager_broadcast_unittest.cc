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
#include "adapter/common/op_store_adapter_manager.h"
#include "graph/ge_tensor.h"
#include "fusion_manager/fusion_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "ops_store/ops_kernel_manager.h"
#include "format_selector/common/format_dtype_selector_base.h"
#include "format_selector/builtin/broadcast/format_process/broadcast_process_fractal_z.h"
#include "format_selector/builtin/broadcast/format_process/broadcast_process_fractal_z_3d.h"
#include "format_selector/builtin/broadcast/format_process/broadcast_format_process.h"
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

class FormatDtypeSelectorManagerBroadcastUTest : public testing::Test{
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

// grads/x1: 4d, x2: scalar
// grads/x1: 5HD/6D/FZ/NZ, x2: ND
// y1:5HD/6D/FZ/NZ, y2: ND
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_partscalar_success)
{
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("minimum", "MinimumGrad");
  vector<int64_t> dim_data1 = {16,128,128,64};
  vector<int64_t> dim_data2 = {16,128,128,64};
  vector<int64_t> dim_data3;
  vector<int64_t> dim_data4 = {16,128,128,64};
  vector<int64_t> dim_dat5;

  op_desc_ptr->AddInputDesc("grads", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("x1", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("x2", CreateTensorDesc(dim_data3, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y1", CreateTensorDesc(dim_data4, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y2", CreateTensorDesc(dim_dat5, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string expect_format = EQUAL_ORIGIN_FORMAT + EQUAL_HEAVY_FORMAT;
  string x2_expect_format = EQUAL_ORIGIN_FORMAT+",ND,ND,ND,ND,ND,ND,ND,ND";
  EXPECT_EQ(expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  EXPECT_EQ(x2_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[2]->GetUniqueName())));

  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
  EXPECT_EQ(x2_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[1]->GetUniqueName())));
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_enhanced_success_1)
{
  GeTensorDesc tensor_desc = CreateTensorDesc({16,-1,128,64}, FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc_ptr1 = std::make_shared<OpDesc>("relu_grad_v1", "ReluGradV1");
  op_desc_ptr1->AddInputDesc("gradients", tensor_desc);
  op_desc_ptr1->AddInputDesc("features", tensor_desc);
  op_desc_ptr1->AddOutputDesc("backprops", tensor_desc);

  OpDescPtr op_desc_ptr2 = std::make_shared<OpDesc>("relu_grad_v2", "ReluGradV2");
  op_desc_ptr2->AddInputDesc("gradients", tensor_desc);
  op_desc_ptr2->AddInputDesc("features", tensor_desc);
  op_desc_ptr2->AddOutputDesc("backprops", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node1 = graph->AddNode(op_desc_ptr1);
  ge::NodePtr node2 = graph->AddNode(op_desc_ptr2);

  OpKernelInfoPtr op_kernel_info_ptr1 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType
          ("tbe-builtin", op_desc_ptr1->GetType());
  OpKernelInfoPtr op_kernel_info_ptr2 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType
          ("tbe-builtin", op_desc_ptr2->GetType());
  EXPECT_NE(op_kernel_info_ptr1, nullptr);
  EXPECT_NE(op_kernel_info_ptr2, nullptr);
  EXPECT_EQ(op_kernel_info_ptr1->GetAllInputInfo().size(), 2);
  EXPECT_EQ(op_kernel_info_ptr1->GetAllOutputInfo().size(), 1);
  EXPECT_EQ(op_kernel_info_ptr2->GetAllInputInfo().size(), 2);
  EXPECT_EQ(op_kernel_info_ptr2->GetAllOutputInfo().size(), 1);

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);

  // check inputs
  FormatDtypeInfo format_dtype_info1;
  Status result1 = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr1, node1,
                                                                 false, format_dtype_info1);
  EXPECT_EQ(fe::SUCCESS, result1);
  vector<Format> op1_input0_format = format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllInputInfo()[0]->GetUniqueName());
  vector<Format> op1_input1_format = format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllInputInfo()[1]->GetUniqueName());
  vector<Format> op1_output0_format = format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllOutputInfo()[0]->GetUniqueName());
  EXPECT_EQ(std::find(op1_input0_format.begin(), op1_input0_format.end(), ge::FORMAT_NC1HWC0), op1_input0_format.end());
  EXPECT_EQ(std::find(op1_input1_format.begin(), op1_input1_format.end(), ge::FORMAT_NC1HWC0), op1_input1_format.end());
  EXPECT_EQ(std::find(op1_output0_format.begin(), op1_output0_format.end(), ge::FORMAT_NC1HWC0), op1_output0_format.end());

  FormatDtypeInfo format_dtype_info2;
  Status result2 = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr2, node2,
                                                                 false, format_dtype_info2);
  EXPECT_EQ(fe::SUCCESS, result2);
  vector<Format> op2_input0_format = format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllInputInfo()[0]->GetUniqueName());
  vector<Format> op2_input1_format = format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllInputInfo()[1]->GetUniqueName());
  vector<Format> op2_output0_format = format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllOutputInfo()[0]->GetUniqueName());
  EXPECT_NE(std::find(op2_input0_format.begin(), op2_input0_format.end(), ge::FORMAT_NC1HWC0), op2_input0_format.end());
  EXPECT_NE(std::find(op2_input1_format.begin(), op2_input1_format.end(), ge::FORMAT_NC1HWC0), op2_input1_format.end());
  EXPECT_NE(std::find(op2_output0_format.begin(), op2_output0_format.end(), ge::FORMAT_NC1HWC0),
            op2_output0_format.end());

  EXPECT_EQ(op1_input0_format.size() + 2, op2_input0_format.size());
  EXPECT_EQ(op1_input1_format.size() + 2, op2_input1_format.size());
  EXPECT_EQ(op1_output0_format.size() + 2, op2_output0_format.size());
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_enhanced_success_2)
{
  GeTensorDesc tensor_desc1 = CreateTensorDesc({16,16,128,64}, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_desc2 = CreateTensorDesc({16,32,128,64}, FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc_ptr1 = std::make_shared<OpDesc>("relu_grad_v1", "ReluGradV1");
  op_desc_ptr1->AddInputDesc("gradients", tensor_desc1);
  op_desc_ptr1->AddInputDesc("features", tensor_desc2);
  op_desc_ptr1->AddOutputDesc("backprops", tensor_desc2);

  OpDescPtr op_desc_ptr2 = std::make_shared<OpDesc>("relu_grad_v2", "ReluGradV2");
  op_desc_ptr2->AddInputDesc("gradients", tensor_desc1);
  op_desc_ptr2->AddInputDesc("features", tensor_desc2);
  op_desc_ptr2->AddOutputDesc("backprops", tensor_desc2);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node1 = graph->AddNode(op_desc_ptr1);
  ge::NodePtr node2 = graph->AddNode(op_desc_ptr2);

  OpKernelInfoPtr op_kernel_info_ptr1 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType
          ("tbe-builtin", op_desc_ptr1->GetType());
  OpKernelInfoPtr op_kernel_info_ptr2 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType
          ("tbe-builtin", op_desc_ptr2->GetType());
  EXPECT_NE(op_kernel_info_ptr1, nullptr);
  EXPECT_NE(op_kernel_info_ptr2, nullptr);
  EXPECT_EQ(op_kernel_info_ptr1->GetAllInputInfo().size(), 2);
  EXPECT_EQ(op_kernel_info_ptr1->GetAllOutputInfo().size(), 1);
  EXPECT_EQ(op_kernel_info_ptr2->GetAllInputInfo().size(), 2);
  EXPECT_EQ(op_kernel_info_ptr2->GetAllOutputInfo().size(), 1);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);

  // check inputs
  FormatDtypeInfo format_dtype_info1;
  Status result1 = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr1, node1,
                                                                 false, format_dtype_info1);
  EXPECT_EQ(fe::SUCCESS, result1);
  vector<Format> op1_input0_format = format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllInputInfo()[0]->GetUniqueName());
  vector<Format> op1_input1_format = format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllInputInfo()[1]->GetUniqueName());
  vector<Format> op1_output0_format = format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllOutputInfo()[0]->GetUniqueName());
  EXPECT_EQ(std::find(op1_input0_format.begin(), op1_input0_format.end(), ge::FORMAT_NC1HWC0), op1_input0_format.end());
  EXPECT_EQ(std::find(op1_input1_format.begin(), op1_input1_format.end(), ge::FORMAT_NC1HWC0), op1_input1_format.end());
  EXPECT_EQ(std::find(op1_output0_format.begin(), op1_output0_format.end(), ge::FORMAT_NC1HWC0), op1_output0_format.end());

  FormatDtypeInfo format_dtype_info2;
  Status result2 = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr2, node2,
                                                                 false, format_dtype_info2);
  EXPECT_EQ(fe::SUCCESS, result2);
  vector<Format> op2_input0_format = format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllInputInfo()[0]->GetUniqueName());
  vector<Format> op2_input1_format = format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllInputInfo()[1]->GetUniqueName());
  vector<Format> op2_output0_format = format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllOutputInfo()[0]->GetUniqueName());
  EXPECT_NE(std::find(op2_input0_format.begin(), op2_input0_format.end(), ge::FORMAT_NC1HWC0), op2_input0_format.end());
  EXPECT_NE(std::find(op2_input1_format.begin(), op2_input1_format.end(), ge::FORMAT_NC1HWC0), op2_input1_format.end());
  EXPECT_NE(std::find(op2_output0_format.begin(), op2_output0_format.end(), ge::FORMAT_NC1HWC0),
            op2_output0_format.end());

  EXPECT_EQ(op1_input0_format.size() + 2, op2_input0_format.size());
  EXPECT_EQ(op1_input1_format.size() + 2, op2_input1_format.size());
  EXPECT_EQ(op1_output0_format.size() + 2, op2_output0_format.size());
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_enhanced_success_3)
{
  GeTensorDesc tensor_desc1 = CreateTensorDesc({16,-1,128,64}, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_desc2 = CreateTensorDesc({}, FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_desc_ptr1 = std::make_shared<OpDesc>("relu_grad_v1", "ReluGradV1");
  op_desc_ptr1->AddInputDesc("gradients", tensor_desc1);
  op_desc_ptr1->AddInputDesc("features", tensor_desc2);
  op_desc_ptr1->AddOutputDesc("backprops", tensor_desc1);

  OpDescPtr op_desc_ptr2 = std::make_shared<OpDesc>("relu_grad_v2", "ReluGradV2");
  op_desc_ptr2->AddInputDesc("gradients", tensor_desc1);
  op_desc_ptr2->AddInputDesc("features", tensor_desc2);
  op_desc_ptr2->AddOutputDesc("backprops", tensor_desc1);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node1 = graph->AddNode(op_desc_ptr1);
  ge::NodePtr node2 = graph->AddNode(op_desc_ptr2);

  OpKernelInfoPtr op_kernel_info_ptr1 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType
          ("tbe-builtin", op_desc_ptr1->GetType());
  OpKernelInfoPtr op_kernel_info_ptr2 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType
          ("tbe-builtin", op_desc_ptr2->GetType());
  EXPECT_NE(op_kernel_info_ptr1, nullptr);
  EXPECT_NE(op_kernel_info_ptr2, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);

  // check inputs
  FormatDtypeInfo format_dtype_info1;
  Status result1 = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr1, node1,
                                                                 false, format_dtype_info1);
  EXPECT_EQ(fe::SUCCESS, result1);

  FormatDtypeInfo format_dtype_info2;
  Status result2 = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr2, node2,
                                                                 false, format_dtype_info2);
  EXPECT_EQ(fe::SUCCESS, result2);

  EXPECT_EQ(format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllInputInfo()[0]->GetUniqueName()),
            format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllInputInfo()[0]->GetUniqueName()));
  EXPECT_EQ(format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllInputInfo()[1]->GetUniqueName()),
            format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllInputInfo()[1]->GetUniqueName()));
  EXPECT_EQ(format_dtype_info1.format_map.at(op_kernel_info_ptr1->GetAllOutputInfo()[0]->GetUniqueName()),
            format_dtype_info2.format_map.at(op_kernel_info_ptr2->GetAllOutputInfo()[0]->GetUniqueName()));
}

// x: 4d, y: 4d (the last two axis are not 16 aligned)
// x: 5hd, y:5hd, z:5hd
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_5hd_success)
{
const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
vector<int64_t> dim_data1 = {16, 32, 8, 8};
vector<int64_t> dim_data2 = {1, 32, 16, 16};
op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
EXPECT_NE(op_kernel_info_ptr, nullptr);

FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
FormatDtypeInfo format_dtype_info;
Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                           false, format_dtype_info);
EXPECT_EQ(fe::SUCCESS, result);

string x_expect_format = EQUAL_ORIGIN_FORMAT+",NC1HWC0,NC1HWC0";
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// x: 4d, y: 4d (the last two axis are not 16 aligned)
// x: 5hd/6d/fz, y:5hd/6d/fz, z:5hd/6d/fz
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_6d_success)
{
const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
vector<int64_t> dim_data1 = {16, 32, 8, 8};
vector<int64_t> dim_data2 = {16, 32, 16, 16};
op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
EXPECT_NE(op_kernel_info_ptr, nullptr);

FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
FormatDtypeInfo format_dtype_info;
Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                           false, format_dtype_info);
EXPECT_EQ(fe::SUCCESS, result);

string x_expect_format = EQUAL_ORIGIN_FORMAT+",NC1HWC0,NC1HWC0,C1HWNCoC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z";
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// x: 4d, y: 2d(the last two axis are 16 aligned)
// x: NZ, y: NZ, z: NZ
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_nz_success)
{
const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
vector<int64_t> dim_data1 = {16, 32, 16, 16};
vector<int64_t> dim_data2 = {16,16};
op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
EXPECT_NE(op_kernel_info_ptr, nullptr);

FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
FormatDtypeInfo format_dtype_info;
Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                           false, format_dtype_info);
EXPECT_EQ(fe::SUCCESS, result);
string x_expect_format = EQUAL_ORIGIN_FORMAT+",FRACTAL_NZ,FRACTAL_NZ";
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}

// x: 4d, y: 4d(the second last axis needs broadcast)
// do not support nz
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_nz_not_success) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 16, 32};
  vector<int64_t> dim_data2 = {16, 32, 1, 32};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin",
                                                                                                        op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);
  string x_expect_format = EQUAL_ORIGIN_FORMAT + ",NC1HWC0,NC1HWC0,C1HWNCoC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z";
  EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}


// x: 4d, y: 2d (the last two axis are not 16 aligned)
// x: null, y:null, z:null
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_no_heavyformat_success) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  vector<int64_t> dim_data2 = {16, 8};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin",
                                                                                                        op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);
  EXPECT_EQ(EQUAL_ORIGIN_FORMAT, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  EXPECT_EQ(EQUAL_ORIGIN_FORMAT, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(EQUAL_ORIGIN_FORMAT, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}


TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_dynamicinput_success) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("addN", "AddN");
  vector<int64_t> dim_data1 = {16, 16, 128, 64};
  vector<int64_t> dim_data2 = {16, 16, 128, 64};
  vector<int64_t> dim_data3 = {16, 16, 128, 64};
  op_desc_ptr->AddInputDesc("x0", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("x1", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data3, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin",
                                                                                                        op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

// check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  string x_expect_format = EQUAL_ORIGIN_FORMAT + EQUAL_HEAVY_FORMAT;
  EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
  vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
  EXPECT_EQ(x_expect_format, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}


// format are all NDHWC
// original x: 4d, y: 4d z:4d (axis is less than 5)
// they are not support 6hd
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_6hd_success) {
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

// format are all NDHWC
// original x: 5d, y: 5d z:5d (C of each tensor is 16 aligned)
// they do not support 6hd
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_6hd_success_1) {
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
// they support 6hd
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_6hd_success_2) {
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
// original x: 5d, y: 5d z:5d (axis N\D\H need to BroadCast)
// they support 6hd
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_6hd_success_3) {
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
// original x: 5d, y: 5d z:5d (axis N\D\H need to BroadCast)
// they support 6hd
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_6hd_success_4) {
    const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
    vector<int64_t> dim_data1 = {1, 32, 16, 1, 8};
    vector<int64_t> dim_data2 = {16, 32, 32, 8, 8};
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
            ",NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0";
    EXPECT_EQ(x_expect_format,
              GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
    EXPECT_EQ(x_expect_format,
              GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(x_expect_format,
              GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}
TEST_F(FormatDtypeSelectorManagerBroadcastUTest, GetDataTypeFromOpDescByKey_fail) {
  using FormatDtypeOpBuiltinSelectorPtr = std::shared_ptr<FormatDtypeOpBuiltinSelector>;
  FormatDtypeOpBuiltinSelectorPtr ptr = std::make_shared<FormatDtypeOpBuiltinSelector>();
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("equal", "Equal");
  vector<ge::DataType> result;
  Status ret = ptr->GetDataTypeFromOpDescByKey(op_desc_ptr, "test", result);
  EXPECT_EQ(ret, fe::FAILED);
  std::map<std::string, vector<ge::DataType>> data_type_map = { {"test", {DT_FLOAT16} }};
  (void)op_desc_ptr->SetExtAttr("ext_dynamic_datatype", data_type_map);
  ret = ptr->GetDataTypeFromOpDescByKey(op_desc_ptr, "test_fail", result);
  EXPECT_EQ(ret, fe::FAILED);
  ret = ptr->GetDataTypeFromOpDescByKey(op_desc_ptr, "test", result);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, get_reduce_new_format_dtype_test) {
  using FormatDtypeOpBuiltinSelectorPtr = std::shared_ptr<FormatDtypeOpBuiltinSelector>;
  FormatDtypeOpBuiltinSelectorPtr ptr = std::make_shared<FormatDtypeOpBuiltinSelector>();
  OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("conv");
  op_kernel_info_ptr->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::OpPattern)] =
          static_cast<int64_t>(OpPattern::OP_PATTERN_REDUCE);
  const OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  string input_key;
  string output_key;
  FormatDtypeInfo format_dtype_info;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<ge::Format> format_vec2 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<ge::Format> format_vec3 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<uint32_t> sub_format_vec = {0, 0};
  format_dtype_info.format_map.emplace("x", format_vec1);
  format_dtype_info.format_map.emplace("y", format_vec2);
  format_dtype_info.format_map.emplace("z", format_vec3);
  format_dtype_info.sub_format_map.emplace("x", sub_format_vec);
  format_dtype_info.sub_format_map.emplace("y", sub_format_vec);
  format_dtype_info.sub_format_map.emplace("z", sub_format_vec);
  Status result = ptr->GetReduceNewFormatDType(op_kernel_info_ptr, op_desc, input_key,
                                               output_key, format_dtype_info);
  EXPECT_EQ(result, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, get_sub_format_test) {
  using FormatDtypeOpBuiltinSelectorPtr = std::shared_ptr<FormatDtypeOpBuiltinSelector>;
  FormatDtypeOpBuiltinSelectorPtr ptr = std::make_shared<FormatDtypeOpBuiltinSelector>();
  OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("conv");
  op_kernel_info_ptr->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::OpPattern)] =
          static_cast<int64_t>(OpPattern::OP_PATTERN_REDUCE);
  const OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  string input_key;
  string output_key;
  vector<ge::Format> format_vec;
  vector<uint32_t> sub_format_vec;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc);
  InputOrOutputInfoPtr info_ptr = std::make_shared<InputOrOutputInfo>("test");
  Status result = ptr->GetSupportFormatSubFormat(op_kernel_info_ptr, info_ptr, node, format_vec, sub_format_vec);
  EXPECT_EQ(result, fe::SUCCESS);
  map<string, vector<ge::Format>> format_map;
  result = ptr->GetAllSupportFormat(op_kernel_info_ptr, node, false, format_map);
  EXPECT_EQ(result, fe::SUCCESS);
  std::vector<bool> fallback_res;
  result = ptr->GetFallbackFlags(op_kernel_info_ptr, node, false, fallback_res);
  EXPECT_EQ(result, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, check_part_nonscalar_inputs_test) {
  FormatProccessInputArg input_arg(FORMAT_NCHW, DT_FLOAT, ge::GeShape({1}), ge::FORMAT_FRACTAL_Z, 2);
  BroadcastProcessFractalZ broadcastprocessfractalz;
  Status ret = broadcastprocessfractalz.CheckPartNonScalarInputs(input_arg);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, check_part_nonscalar_inputs_test1) {
  FormatProccessInputArg input_arg(FORMAT_NCHW, DT_FLOAT, ge::GeShape({1}), ge::FORMAT_FRACTAL_Z_3D, 2);
  BroadcastProcessFractalZ3D broadcastprocessfractalz3d;
  Status ret = broadcastprocessfractalz3d.CheckPartNonScalarInputs(input_arg);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, check_part_nonscalar_inputs_test2) {
  FormatProccessInputArg input_arg(FORMAT_NCHW, DT_FLOAT, ge::GeShape({-1, 32, 32, 16}), ge::FORMAT_FRACTAL_Z, 2);
  BroadcastProcessFractalZ broadcastprocessfractalz;
  Status ret = broadcastprocessfractalz.CheckPartNonScalarInputs(input_arg);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_0) {
  BroadcastProcessFractalZ broadcast_format_pr;
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("minimum", "MinimumGrad");


  vector<ge::Format> input_format = {ge::FORMAT_C1HWC0, ge::FORMAT_NCHW};
  vector<ge::DataType> input_datype = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16}), ge::GeShape({2,4,5,45})};
  ge::Format new_format = ge::FORMAT_ND;
  bool ret =  broadcast_format_pr.CheckNewShapeSupportBroadcast(*op_desc_ptr, input_format, input_datype, input_shape, new_format, 1);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_1) {
  BroadcastProcessFractalZ broadcast_format_pr;
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Test");
  vector<ge::Format> input_format = {ge::FORMAT_C1HWC0, ge::FORMAT_NCHW};
  vector<ge::DataType> input_datype = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  vector<ge::GeShape> input_shape = {ge::GeShape({2,4,5,45})};
  ge::Format new_format = ge::FORMAT_ND;
  bool ret = broadcast_format_pr.CheckNewShapeSupportBroadcast(*op_desc, input_format, input_datype, input_shape, new_format, 1);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_2) {
  BroadcastProcessFractalZ broadcast_format_pr;
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Test");
  vector<ge::Format> input_format;
  vector<ge::DataType> input_datype;
  vector<ge::GeShape> input_shape;
  ge::Format new_format = ge::FORMAT_ND;
  bool ret = broadcast_format_pr.CheckNewShapeSupportBroadcast(*op_desc, input_format, input_datype, input_shape, new_format, 1);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_3) {
  BroadcastProcessFractalZ broadcast_format_pr;
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16}), ge::GeShape({2,4,5,45})};
  bool ret = broadcast_format_pr.CheckShapeWithSubformatSupportBroadcast(input_shape);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_4) {
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16})};
  BroadcastProcessFractalZ broadcast_format_pr;
  bool ret = broadcast_format_pr.CheckShapeWithSubformatSupportBroadcast(input_shape);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_5) {
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16}), ge::GeShape({2,4,1,5,45})};
  BroadcastProcessFractalZ broadcast_format_pr;
  bool ret = broadcast_format_pr.CheckShapeWithSubformatSupportBroadcast(input_shape);
  EXPECT_EQ(ret, true);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, converage_04) {
  std::shared_ptr<FormatDtypeSelectorBase> base = std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  string key = "1";
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test");
  auto node = graph->AddNode(op_desc);
  vector<uint32_t> result;
  EXPECT_EQ(fe::FAILED, base->GetSubFormatFromOpDescByKey(op_desc, key, result));
  map<string, vector<uint32_t>> sub_format_map;
  sub_format_map["1"] = {1, 1, 1};
  op_desc->SetExtAttr("ext_dynamic_sub_format", sub_format_map);
  EXPECT_EQ(fe::SUCCESS, base->GetSubFormatFromOpDescByKey(op_desc, key, result));
  key = "2";
  EXPECT_EQ(fe::FAILED, base->GetSubFormatFromOpDescByKey(op_desc, key, result));
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, broadcast_check_newshape_support_6) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr opdesc_ptr = std::make_shared<OpDesc>("BroadcastOp", "BroadcastOp");
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  GeTensorDesc in_desc2(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape);
  in_desc1.SetOriginShape(shape);
  in_desc2.SetFormat(ge::FORMAT_NCHW);
  in_desc2.SetOriginFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetOriginDataType(DT_FLOAT);
  in_desc2.SetShape(shape);
  in_desc2.SetOriginShape(shape);
  opdesc_ptr->AddInputDesc("x", in_desc1);
  opdesc_ptr->AddInputDesc("y", in_desc1);
  opdesc_ptr->AddOutputDesc("z", in_desc2);
  NodePtr node = graph->AddNode(opdesc_ptr);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());

  vector<uint32_t> sub_formats;
  FormatDtypeOpBuiltinSelector built_in_selector;
  Status status = built_in_selector.GetBroadcastNewSubformatVec(op_kernel_info_ptr, node, "x", 16, sub_formats);
  std::map<std::string, vector<uint32_t>> sub_format_map;

  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(sub_formats.empty(), true);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, op_broadcast_selector_partscalar)
{
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("minimum", "MinimumGrad");
  vector<int64_t> dim_data1 = {16,128,128,64};
  vector<int64_t> dim_data2 = {16,128,128,64};
  vector<int64_t> dim_data3;
  vector<int64_t> dim_data4 = {16,128,128,64};
  vector<int64_t> dim_dat5;

  op_desc_ptr->AddInputDesc("grads", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("x1", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddInputDesc("x2", CreateTensorDesc(dim_data3, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y1", CreateTensorDesc(dim_data4, FORMAT_ND, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y2", CreateTensorDesc(dim_dat5, FORMAT_ND, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);
}

TEST_F(FormatDtypeSelectorManagerBroadcastUTest, coverage_03) {
  const std::string name = "CpuEngine";
  OpStoreAdapterManager::Instance(name);
}
