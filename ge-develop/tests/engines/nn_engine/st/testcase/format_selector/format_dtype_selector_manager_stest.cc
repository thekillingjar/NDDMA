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
#include "common/util/op_info_util.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "graph/ge_tensor.h"
#include "common/platform_utils.h"
#include "fusion_manager/fusion_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "common/fe_type_utils.h"
#include "format_selector/builtin/reduce/reduce_format_selector/reduce_process_nz.h"
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

class FormatDtypeSelectorManagerSTest : public testing::Test {
 protected:

  static void SetUpTestCase() {
    fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910A");
    PlatformUtils::Instance().soc_version_ = "Ascend910A";
    cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
  }

  static void TearDownTestCase() {
    cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
  }

  // Some expensive resource shared by all tests.
  void SetUp() {
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(AI_CORE_NAME);
    FEOpsStoreInfo tbe_builtin{
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo",
        ""
    };
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    std::map<std::string, std::string> options;
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    fe_ops_kernel_info_store_ptr_->Initialize(options);
    auto reflection_builder_ptr = std::make_shared<ge::RefRelations>();
    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(
        fe::AI_CORE_NAME, reflection_builder_ptr);
    op_format_dtype_judge_ptr_->Initialize();
    conv_input_name_map_ = {
        {0, "x"}, {1, "y"}, {2, "z"}
    };
    conv_output_name_map_ = {
        {0, "o"},
    };
    graph_ = make_shared<ge::ComputeGraph>("test");
    conv_op_ = std::make_shared<OpDesc>("test", "Conv2D");
    auto shape = ge::GeShape();
    GeTensorDesc input_desc(shape);
    conv_op_->AddInputDesc("x", input_desc);
    conv_op_->AddInputDesc("y", input_desc);
    conv_op_->AddOutputDesc("o", input_desc);
    conv_node_ = graph_->AddNode(conv_op_);
    conv_kernel_ = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr("Conv2D");
    ASSERT_NE(conv_kernel_, nullptr);
    cout << "a test Set Up" << endl;
  }

  void TearDown() {
    cout << "a test Tear Down" << endl;
    fe_ops_kernel_info_store_ptr_->Finalize();

  }

 protected:

  static void CreateOneOpGraph(ComputeGraphPtr graph, OpDescPtr op_desc_ptr) {
    graph->AddNode(op_desc_ptr);
  }

  static GeTensorDesc CreateTensorDesc(vector<int64_t> dim_data, ge::Format format, ge::DataType data_type) {
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, format, data_type);
    data_desc.SetOriginFormat(format);
    data_desc.SetOriginShape(shape_data);
    data_desc.SetOriginDataType(data_type);
    return data_desc;
  }

 public:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  shared_ptr<fe::OpFormatDtypeJudge> op_format_dtype_judge_ptr_;
  std::map<uint32_t, std::string> conv_input_name_map_;
  std::map<uint32_t, std::string> conv_output_name_map_;
  OpKernelInfoPtr conv_kernel_ = nullptr;

  ge::ComputeGraphPtr graph_;
  ge::NodePtr conv_node_;
  ge::OpDescPtr conv_op_;
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr;
  shared_ptr<ge::GeTensorDesc> input1_desc_ptr;
  shared_ptr<ge::GeTensorDesc> input2_desc_ptr;
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr;
  shared_ptr<ge::OpDesc> op_desc_ptr;
  const string EQUAL_ORIGIN_FORMAT = "NCHW,NCHW,NHWC,NHWC,HWCN,HWCN,CHWN,CHWN,NDHWC,NDHWC,NCDHW,NCDHW,DHWCN,DHWCN,DHWNC,DHWNC,ND,ND";
  const string EQUAL_HEAVY_FORMAT = ",NC1HWC0,NC1HWC0,C1HWNCoC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z,FRACTAL_NZ,FRACTAL_NZ";
};


bool SelectTbeOpFormatStub(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str)
{
    op_format_dtype_str = "{\"input0\":{\"name\":\"x\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"},\"output0\":{\"name\":\"y\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"}}";
    return true;
}

bool SelectTbeOpFormatStub1(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str)
{
    op_format_dtype_str = "";
    return true;
}

TEST_F(FormatDtypeSelectorManagerSTest, init_format_success) {
    OpDescPtr opdesc_ptr = std::make_shared<OpDesc>("test", "KernelFormatOp1");
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_OP_KERNEL, op_kernel_info_ptr->GetOpPattern());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ("input0.x", inputs[0]->GetUniqueName());
    EXPECT_EQ("output0.y", outputs[0]->GetUniqueName());
    EXPECT_EQ(3, inputs[0]->GetFormat().size());
    EXPECT_EQ(3, outputs[0]->GetFormat().size());

    opdesc_ptr = std::make_shared<OpDesc>("test", "KernelFormatOp2");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_OP_KERNEL, op_kernel_info_ptr->GetOpPattern());

    opdesc_ptr = std::make_shared<OpDesc>("test", "FormatAgnosticOp");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_FORMAT_AGNOSTIC, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    int format_size = 35;
    EXPECT_EQ(format_size - 1, inputs[0]->GetFormat().size());
    EXPECT_EQ(format_size - 1, outputs[0]->GetFormat().size());

    opdesc_ptr = std::make_shared<OpDesc>("test", "BroadcastOp");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_BROADCAST, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(3, inputs[0]->GetDataType().size());
    EXPECT_EQ(3, outputs[0]->GetDataType().size());

    opdesc_ptr = std::make_shared<OpDesc>("test", "ReduceOp");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_REDUCE, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(2, inputs[0]->GetDataType().size());
    EXPECT_EQ(2, outputs[0]->GetDataType().size());
}

TEST_F(FormatDtypeSelectorManagerSTest, init_customize_format_success) {
    OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithFormat");
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_OP_CUSTOMIZE, op_kernel_info_ptr->GetOpPattern());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, inputs[0]->GetDataType().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetDataType().size());

    op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    EXPECT_EQ(OP_PATTERN_OP_CUSTOMIZE, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, inputs[0]->GetDataType().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetDataType().size());
}

TEST_F(FormatDtypeSelectorManagerSTest, op_kernel_selector_query_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "KernelFormatOp1");
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                               false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
    vector<ge::Format> format_vec = format_dtype_info.format_map.at(inputs[0]->GetUniqueName());
    vector<uint32_t> sub_format_vec = format_dtype_info.sub_format_map.at(inputs[0]->GetUniqueName());
    vector<ge::DataType> data_type_vec = format_dtype_info.data_type_map.at(inputs[0]->GetUniqueName());
    EXPECT_EQ(ge::FORMAT_NCHW, format_vec[0]);
    EXPECT_EQ(ge::FORMAT_NC1HWC0, format_vec[1]);
    EXPECT_EQ(ge::FORMAT_ND, format_vec[2]);
    EXPECT_EQ(ge::DT_FLOAT16, data_type_vec[0]);
    EXPECT_EQ(ge::DT_FLOAT, data_type_vec[1]);
    EXPECT_EQ(ge::DT_FLOAT, data_type_vec[2]);
}

TEST_F(FormatDtypeSelectorManagerSTest, op_formatagnostic_selector_query_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "FormatAgnosticOp");
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);
    OpKernelInfoPtr op_kernel_info_ptr;
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                               false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
    vector<ge::Format> format_vec = format_dtype_info.format_map.at(inputs[0]->GetUniqueName());
    vector<ge::DataType> data_type_vec = format_dtype_info.data_type_map.at(inputs[0]->GetUniqueName());
    string new_formats_str = GetStrByFormatVec(format_vec);
    string new_data_types_str = GetStrByDataTypeVec(data_type_vec);
    string expect_format_str = "NC1HWC0_C04,NC1HWC0,C1HWNCoC0,FRACTAL_Z,FRACTAL_NZ,NDC1HWC0,FRACTAL_Z_3D,FRACTAL_Z_3D_TRANSPOSE,FRACTAL_Z_C04,FRACTAL_Z_WINO,C1HWC0,FRACTAL_NZ_C0_2,FRACTAL_NZ_C0_4,FRACTAL_NZ_C0_8,FRACTAL_NZ_C0_16,FRACTAL_NZ_C0_32,NCHW,NCHW,NHWC,NHWC,HWCN,HWCN,CHWN,CHWN,NDHWC,NDHWC,NCDHW,NCDHW,DHWCN,DHWCN,DHWNC,DHWNC,ND,ND";
    EXPECT_EQ(expect_format_str, new_formats_str);
    string expect_data_type_str = "DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT,DT_FLOAT16,DT_FLOAT";
    EXPECT_EQ(expect_data_type_str, new_data_types_str);
    format_vec = format_dtype_info.format_map.at(outputs[0]->GetUniqueName());
    data_type_vec = format_dtype_info.data_type_map.at(outputs[0]->GetUniqueName());
    new_formats_str = GetStrByFormatVec(format_vec);
    new_data_types_str = GetStrByDataTypeVec(data_type_vec);
    EXPECT_EQ(expect_format_str, new_formats_str);
    EXPECT_EQ("DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32,DT_INT8,DT_INT32",        
              new_data_types_str);
}

TEST_F(FormatDtypeSelectorManagerSTest, op_customize_selector_query_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
    vector<int64_t> dim_data = {1, 2, 3, 4};
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, FORMAT_NCHW, DT_FLOAT);
    op_desc_ptr->AddInputDesc("x", data_desc);
    op_desc_ptr->AddOutputDesc("y", data_desc);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_op_store_adapter_ptr->SelectTbeOpFormat = SelectTbeOpFormatStub;

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                               false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
    vector<ge::Format> format_vec = format_dtype_info.format_map.at(inputs[0]->GetUniqueName());
    vector<ge::DataType> data_type_vec = format_dtype_info.data_type_map.at(inputs[0]->GetUniqueName());
    EXPECT_EQ(ge::FORMAT_ND, format_vec[0]);
    EXPECT_EQ(ge::FORMAT_NCHW, format_vec[1]);
    EXPECT_EQ(ge::DT_FLOAT, data_type_vec[0]);
    EXPECT_EQ(ge::DT_FLOAT16, data_type_vec[1]);

    // check outputs
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    format_vec = format_dtype_info.format_map.at(outputs[0]->GetUniqueName());
    data_type_vec = format_dtype_info.data_type_map.at(outputs[0]->GetUniqueName());
    EXPECT_EQ(ge::FORMAT_ND, format_vec[0]);
    EXPECT_EQ(ge::FORMAT_NCHW, format_vec[1]);
    EXPECT_EQ(ge::DT_FLOAT, data_type_vec[0]);
    EXPECT_EQ(ge::DT_FLOAT16, data_type_vec[1]);
}

TEST_F(FormatDtypeSelectorManagerSTest, ConvertFormatDtype_success)
{
  string format_vec_str = "ND, ND";
  string sub_format_vec_str = "test1, test2";
  string data_type_vec_str = "float16, float16";
  uint32_t format_size_of_first_input_or_output = 2;
  vector<ge::Format> format_vec;
  vector<uint32_t> sub_format_vec;
  vector<ge::DataType> data_type_vec;
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
        std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  Status status = customize_selector_ptr->ConvertFormatDtype(format_vec_str, sub_format_vec_str, data_type_vec_str,
                                                             format_size_of_first_input_or_output, format_vec,
                                                             sub_format_vec, data_type_vec);
  EXPECT_EQ(status, fe::SUCCESS);
}


TEST_F(FormatDtypeSelectorManagerSTest, op_customize_selector_query_failed)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);
    vector<int64_t> dim_data = {1, 2, 3, 4};
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, FORMAT_NCHW, DT_FLOAT);
    op_desc_ptr->AddInputDesc("x", data_desc);
    op_desc_ptr->AddOutputDesc("y", data_desc);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_op_store_adapter_ptr->SelectTbeOpFormat = SelectTbeOpFormatStub1;

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                                  false, format_dtype_info);
    EXPECT_EQ(fe::FAILED, result);
}

// grads/x1: 4d, x2: scalar
// grads/x1: 5HD/6D/FZ/NZ, x2: ND
// y1:5HD/6D/FZ/NZ, y2: ND
TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_selector_partscalar_success)
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
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

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

TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_enhanced_success_1)
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

TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_enhanced_success_2)
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

TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_enhanced_success_3)
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
  EXPECT_EQ(fe::SUCCESS, result1);;
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
TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_selector_5hd_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
    vector<int64_t> dim_data1 = {16, 32, 8, 8};
    vector<int64_t> dim_data2 = {1, 32, 16, 16};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

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
TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_selector_6d_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
    vector<int64_t> dim_data1 = {16, 32, 8, 8};
    vector<int64_t> dim_data2 = {16, 32, 16, 16};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

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
TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_selector_nz_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
    vector<int64_t> dim_data1 = {16, 32, 16, 16};
    vector<int64_t> dim_data2 = {16,16};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

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

// x: 4d, y: 2d (the last two axis are not 16 aligned)
// x: null, y:null, z:null
TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_selector_no_heavyformat_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("equal", "Equal");
    vector<int64_t> dim_data1 = {16, 32, 16, 16};
    vector<int64_t> dim_data2 = {16, 8};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddInputDesc("y", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddOutputDesc("z", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    // check inputs
    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                                 false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
    EXPECT_EQ(EQUAL_ORIGIN_FORMAT, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[0]->GetUniqueName())));
    EXPECT_EQ(EQUAL_ORIGIN_FORMAT, GetStrByFormatVec(format_dtype_info.format_map.at(inputs[1]->GetUniqueName())));
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(EQUAL_ORIGIN_FORMAT, GetStrByFormatVec(format_dtype_info.format_map.at(outputs[0]->GetUniqueName())));
}


TEST_F(FormatDtypeSelectorManagerSTest, op_broadcast_selector_dynamicinput_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("addN", "AddN");
    vector<int64_t> dim_data1 = {16,16,128,64};
    vector<int64_t> dim_data2 = {16,16,128,64};
    vector<int64_t> dim_data3 = {16,16,128,64};
    op_desc_ptr->AddInputDesc("x0", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddInputDesc("x1", CreateTensorDesc(dim_data2, FORMAT_NCHW, DT_FLOAT));
    op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data3, FORMAT_NCHW, DT_FLOAT));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

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
TEST_F(FormatDtypeSelectorManagerSTest, op_reduce_selector_5hd_6hd_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 32, 16, 16};
    vector<int64_t> dim_data2 = {16, 8, 1 ,1};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                               false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);

    bool support_flag=false;
    map<string, vector<ge::Format>>::reverse_iterator iter;
    for(iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++){
      vector<ge::Format>::iterator ret;
      ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
      if (ret != iter->second.end()) {
        support_flag=true;
      }
      EXPECT_EQ(support_flag, true);

      support_flag=false;
      ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_C1HWNCoC0);
      if (ret != iter->second.end()) {
        support_flag=true;
      }
      EXPECT_EQ(support_flag, true);

      support_flag=false;
      ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_Z);
      if (ret != iter->second.end()) {
        support_flag=true;
      }
      EXPECT_EQ(support_flag, true);
    }
}

TEST_F(FormatDtypeSelectorManagerSTest, op_reduce_selector_nz_success)
{
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  vector<int64_t> dim_data2 = {16, 8, 1 ,1};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  auto node = graph->AddNode(op_desc_ptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, false);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 1});

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                                 false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag=false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for(iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++){
    vector<ge::Format>::iterator ret;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

TEST_F(FormatDtypeSelectorManagerSTest, op_reduce_selector_5hd_6hd_failed1)
{
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 15};
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  auto node = graph->AddNode(op_desc_ptr);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, false);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  map<string, vector<ge::Format>>::reverse_iterator iter;
  for(iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++){
    bool support_flag=false;
    vector<ge::Format>::iterator ret;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    if (iter->first.find("input") != -1) {
        EXPECT_EQ(support_flag, true);
    } else {
        EXPECT_EQ(support_flag, true);
    }

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);
  }
}

TEST_F(FormatDtypeSelectorManagerSTest, op_reduce_selector_5hd_6hd_failed2)
{
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  GeShape shape_data1(dim_data1);
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  auto node = graph->AddNode(op_desc_ptr);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 1});

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                                 false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  map<string, vector<ge::Format>>::iterator iter;
  for(iter = format_dtype_info.format_map.begin(); iter != format_dtype_info.format_map.end(); iter++){
    bool support_flag=false;
    vector<ge::Format>::iterator ret;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    // EXPECT_EQ(support_flag, true);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

TEST_F(FormatDtypeSelectorManagerSTest, op_reduce_selector_5hd_6hd_failed3)
{
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32};
  GeShape shape_data1(dim_data1);
  op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  auto node = graph->AddNode(op_desc_ptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  map<string, vector<ge::Format>>::reverse_iterator iter;
  for(iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++){
    bool support_flag=false;
    vector<ge::Format>::iterator ret;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag=false;
    ret = std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag=true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_01) {
  std::map<std::string, vector<ge::Format>> format_map;
  std::map<std::string, vector<ge::DataType>> data_type_map;
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test");

  (void) ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index;
  std::vector<uint32_t> expected_index;
  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_02)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<ge::Format> format_vec2 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<ge::Format> format_vec3 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT16};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT16};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);

  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);
  (void) ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {0, 1};
  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_03)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {0};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_03_1)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0, FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z, FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND, FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0, FORMAT_NC1HWC0, FORMAT_NC1HWC0_C04};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1, 2, 3};
  std::vector<uint32_t> expected_index = {0, 2, 3};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_04)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(false);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {1};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_05) {
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(false);

  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {0, 1};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_no_first_layer_conv2d_disable_small_chn)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);

  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(false);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, false);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {1};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_no_first_layer_conv2d_enable_small_chn)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, false);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {0};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_no_first_layer_conv2d_enable_small_chn_only_output_c04)
{
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0_C04};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, false);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {1};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_no_first_layer_normal_op_enable_small_chn)
{
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, false);
  conv_op_->SetType("Normal");
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {0, 1};

  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
  conv_op_->SetType("Conv2D");
}

TEST_F(FormatDtypeSelectorManagerSTest, filter_c04_format_first_layer_normal_op_enable_small_chn)
{
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  std::map<std::string, vector<ge::Format>> format_map;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_NC1HWC0};
  vector<ge::Format> format_vec2 = {FORMAT_FRACTAL_Z_C04, FORMAT_FRACTAL_Z};
  vector<ge::Format> format_vec3 = {FORMAT_ND, FORMAT_ND};
  vector<ge::Format> format_vec4 = {FORMAT_NC1HWC0, FORMAT_NC1HWC0};
  format_map.emplace("input0.x", format_vec1);
  format_map.emplace("input1.y", format_vec2);
  format_map.emplace("input2.z", format_vec3);
  format_map.emplace("output0.o", format_vec4);
  std::map<std::string, vector<ge::DataType>> data_type_map;
  vector<ge::DataType> date_type_vec1 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec2 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec3 = {DT_FLOAT16, DT_FLOAT};
  vector<ge::DataType> date_type_vec4 = {DT_FLOAT16, DT_FLOAT};
  data_type_map.emplace("input0.x", date_type_vec1);
  data_type_map.emplace("input1.y", date_type_vec2);
  data_type_map.emplace("input2.z", date_type_vec3);
  data_type_map.emplace("output0.o", date_type_vec4);
  (void)ge::AttrUtils::SetBool(conv_op_, IS_FIRST_LAYER_CONV, true);
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  conv_op_->SetExtAttr("ext_dynamic_format", format_map);
  conv_op_->SetExtAttr("ext_dynamic_datatype", data_type_map);

  conv_op_->SetType("Normal");
  std::vector<uint32_t> original_index = {0, 1};
  std::vector<uint32_t> expected_index = {0, 1};
  op_format_dtype_judge_ptr_->FilterBySmallChannel(conv_node_, conv_kernel_,
                                                   conv_input_name_map_, conv_output_name_map_, original_index);
  EXPECT_EQ(expected_index, original_index);
  conv_op_->SetType("Conv2D");
}

TEST_F(FormatDtypeSelectorManagerSTest, converage_01) {
  std::shared_ptr<FormatDtypeSelectorBase> base =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  string key = "1";
  vector<ge::Format> result_fm;
  vector<ge::DataType> result_dt;
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test");
  auto node = graph->AddNode(op_desc);
  EXPECT_EQ(fe::FAILED, base->GetFormatFromOpDescByKey(op_desc, key, result_fm));
  EXPECT_EQ(fe::FAILED, base->GetDataTypeFromOpDescByKey(op_desc, key, result_dt));

  map<string, vector<ge::Format>> format_map = {{"2", {ge::FORMAT_ND}}};
  op_desc->SetExtAttr("ext_dynamic_format", format_map);

  map<string, vector<ge::DataType>> dt_map = {{"2", {ge::DT_FLOAT}}};
  op_desc->SetExtAttr("ext_dynamic_datatype", dt_map);

  EXPECT_EQ(fe::FAILED, base->GetFormatFromOpDescByKey(op_desc, key, result_fm));
  EXPECT_EQ(fe::FAILED, base->GetDataTypeFromOpDescByKey(op_desc, key, result_dt));

  OpKernelInfoPtr op_kernel_info_ptr = nullptr;
  HeavyFormatInfo heavy_format_info;
  bool is_dynamic_check = false;

  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  EXPECT_EQ(fe::FAILED, customize_selector_ptr->SetSupportFormatDtype(op_kernel_info_ptr, heavy_format_info,
                                                                      node, is_dynamic_check));
  map<string, vector<ge::Format>> format_map_res = {};
  customize_selector_ptr->GetAllSupportFormat(op_kernel_info_ptr, node, is_dynamic_check, format_map_res);
  EXPECT_EQ(format_map_res, format_map);
}

TEST_F(FormatDtypeSelectorManagerSTest, converage_02) {
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  FormatDtypeInfo format_dtype_info;
  format_dtype_info.format_map = {{"test", {ge::FORMAT_NCHW}}};

  format_dtype_info.sub_format_map = {{"test", {1}}};

  format_dtype_info.data_type_map = {{"test", {ge::DT_FLOAT}}};
  Status ret = customize_selector_ptr->SaveDynamicFormatDtype(format_dtype_info, op_desc);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerSTest, get_reduce_new_format_dtype_test) {
  using FormatDtypeOpBuiltinSelectorPtr = std::shared_ptr<FormatDtypeOpBuiltinSelector>;
  FormatDtypeOpBuiltinSelectorPtr ptr = std::make_shared<FormatDtypeOpBuiltinSelector>();
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["check_subformat.enable"] = "true";
  OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("conv");
  op_kernel_info_ptr->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::OpPattern)] =
          static_cast<int64_t>(OpPattern::OP_PATTERN_REDUCE);
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  string input_key;
  string output_key;
  FormatDtypeInfo format_dtype_info;
  vector<ge::Format> format_vec1 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<ge::Format> format_vec2 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<ge::Format> format_vec3 = {FORMAT_NC1HWC0_C04, FORMAT_FRACTAL_Z_C04};
  vector<uint32_t> sub_format_vec = {0, 0, 0};
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

TEST_F(FormatDtypeSelectorManagerSTest, check_part_nonscalar_inputs_test) {
  FormatProccessInputArg input_arg(FORMAT_NCHW, DT_FLOAT, ge::GeShape({1}), ge::FORMAT_FRACTAL_Z, 2);
  BroadcastProcessFractalZ broadcastprocessfractalz;
  Status ret = broadcastprocessfractalz.CheckPartNonScalarInputs(input_arg);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, check_part_nonscalar_inputs_test1) {
  FormatProccessInputArg input_arg(FORMAT_NCHW, DT_FLOAT, ge::GeShape({1}), ge::FORMAT_FRACTAL_Z_3D, 2);
  BroadcastProcessFractalZ3D broadcastprocessfractalz3d;
  Status ret = broadcastprocessfractalz3d.CheckPartNonScalarInputs(input_arg);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, check_part_nonscalar_inputs_test2) {
  FormatProccessInputArg input_arg(FORMAT_NCHW, DT_FLOAT, ge::GeShape({-1, 32, 32, 16}), ge::FORMAT_FRACTAL_Z_3D, 2);
  BroadcastProcessFractalZ broadcastprocessfractalz;
  Status ret = broadcastprocessfractalz.CheckPartNonScalarInputs(input_arg);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, check_origin_format_and_shape_test) {
  vector<int64_t> dim = {1};
  ge::GeShape shape(dim);
  size_t dim_min = 2;
  vector<ge::Format> input_formats;
  vector<ge::Format> output_formats;
  vector<ge::GeShape> shapes;
  shapes.emplace_back(shape);
  ReduceProcessNz reduceprocessnz;
  bool ret = reduceprocessnz.CheckOriginFormatAndShape(input_formats, output_formats, shapes, dim_min);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, parse_json_fallback)
{
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  nlohmann::json json_object;
  nlohmann::json fallback_json;
  fallback_json["enable"] = "1,1,0,0";
  fallback_json["unknownshape_enable"] = "1,1,0,0";
  json_object["fallback"] = fallback_json;
  FormatDtypeInfo res;
  Status status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::SUCCESS);
  nlohmann::json fallback_fail_json;
  fallback_fail_json["invalid"] = "";
  json_object["fallback"] = fallback_fail_json;
  status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(FormatDtypeSelectorManagerSTest, parse_json_fail)
{
  nlohmann::json json_object;
  FormatDtypeInfo res;
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);

  nlohmann::json fail_json1;
  fail_json1["invalid"] = "";
  json_object["parse_fail"] = fail_json1;
  Status status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::FAILED);

  nlohmann::json fail_json2;
  fail_json2["format"] = "ND";
  json_object["parse_fail"] = fail_json2;
  status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::FAILED);

  nlohmann::json fail_json3;
  fail_json3["format"] = "ND";
  fail_json3["dtype"] = "float16";
  json_object["parse_fail"] = fail_json3;
  status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(FormatDtypeSelectorManagerSTest, op_reduce_selector_5hd_6hd_1)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 32, 16, 16, 1};
    vector<int64_t> dim_data2 = {16, 8, 1 ,1, 1};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_ND, DT_FLOAT16));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                               false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
}

TEST_F(FormatDtypeSelectorManagerSTest, broadcast_check_newshape_support_0) {
  BroadcastProcessFractalZ broadcast_format_pr;
  const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("minimum", "MinimumGrad");


  vector<ge::Format> input_format = {ge::FORMAT_C1HWC0, ge::FORMAT_NCHW};
  vector<ge::DataType> input_datype = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16}), ge::GeShape({2,4,5,45})};
  ge::Format new_format = ge::FORMAT_ND;
  bool ret = broadcast_format_pr.CheckNewShapeSupportBroadcast(*op_desc_ptr, input_format, input_datype, input_shape, new_format, 1);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, broadcast_check_newshape_support_1) {
  BroadcastProcessFractalZ broadcast_format_pr;
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Test");
  vector<ge::Format> input_format = {ge::FORMAT_C1HWC0, ge::FORMAT_NCHW};
  vector<ge::DataType> input_datype = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  vector<ge::GeShape> input_shape = {ge::GeShape({2,4,5,45})};
  ge::Format new_format = ge::FORMAT_ND;
  bool ret = broadcast_format_pr.CheckNewShapeSupportBroadcast(*op_desc, input_format, input_datype, input_shape, new_format, 1);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, broadcast_check_newshape_support_2) {
  BroadcastProcessFractalZ broadcast_format_pr;
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Test");
  vector<ge::Format> input_format;
  vector<ge::DataType> input_datype;
  vector<ge::GeShape> input_shape;
  ge::Format new_format = ge::FORMAT_ND;
  bool ret = broadcast_format_pr.CheckNewShapeSupportBroadcast(*op_desc, input_format, input_datype, input_shape, new_format, 1);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, broadcast_check_newshape_support_3) {
  BroadcastProcessFractalZ broadcast_format_pr;
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16}), ge::GeShape({2,4,5,45})};
  bool ret = broadcast_format_pr.CheckShapeWithSubformatSupportBroadcast(input_shape);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, broadcast_check_newshape_support_4) {
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16})};
  BroadcastProcessFractalZ broadcast_format_pr;
  bool ret = broadcast_format_pr.CheckShapeWithSubformatSupportBroadcast(input_shape);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerSTest, broadcast_check_newshape_support_5) {
  vector<ge::GeShape> input_shape = {ge::GeShape({2,3,4,5,16}), ge::GeShape({2,4,1,5,45})};
  BroadcastProcessFractalZ broadcast_format_pr;
  bool ret = broadcast_format_pr.CheckShapeWithSubformatSupportBroadcast(input_shape);
  EXPECT_EQ(ret, true);
}

TEST_F(FormatDtypeSelectorManagerSTest, coverage_03) {
  const std::string name = "CpuEngine";
  OpStoreAdapterManager::Instance(name);
}

TEST_F(FormatDtypeSelectorManagerSTest, converage_04) {
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