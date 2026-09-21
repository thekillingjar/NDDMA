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
#include "format_selector/builtin/process/format_process_registry.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/config_parser/op_cust_dtypes_config_parser.h"
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


class FormatDtypeSelectorManagerKernelUTest : public testing::Test{
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

        if (Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_ == nullptr) {
          Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_ = std::make_shared<OpCustDtypesConfigParser>();
        }
        op_cust_dtypes_parser_ptr_ =
                std::dynamic_pointer_cast<OpCustDtypesConfigParser>(Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

        OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

        std::map<std::string, std::string> options;
        fe_ops_kernel_info_store_ptr->Initialize(options);
        cout << "a test Set Up" << endl;
    }
    virtual void TearDown(){
        cout << "a test Tear Down" << endl;
        fe_ops_kernel_info_store_ptr->Finalize();

    }

public:
    shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
    OpCustDtypesConfigParserPtr op_cust_dtypes_parser_ptr_;
 };


TEST_F(FormatDtypeSelectorManagerKernelUTest, init_format_success) {
    OpDescPtr opdesc_ptr = std::make_shared<OpDesc>("test", "KernelFormatOp1");
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    EXPECT_EQ(OP_PATTERN_OP_KERNEL, op_kernel_info_ptr->GetOpPattern());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ("input0.x", inputs[0]->GetUniqueName());
    EXPECT_EQ("output0.y", outputs[0]->GetUniqueName());
    EXPECT_EQ(3, inputs[0]->GetFormat().size());
    EXPECT_EQ(3, outputs[0]->GetFormat().size());

    opdesc_ptr = std::make_shared<OpDesc>("test", "KernelFormatOp2");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    EXPECT_EQ(OP_PATTERN_OP_KERNEL, op_kernel_info_ptr->GetOpPattern());

    opdesc_ptr = std::make_shared<OpDesc>("test", "FormatAgnosticOp");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);
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
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    EXPECT_EQ(OP_PATTERN_BROADCAST, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(3, inputs[0]->GetDataType().size());
    EXPECT_EQ(3, outputs[0]->GetDataType().size());

    opdesc_ptr = std::make_shared<OpDesc>("test", "ReduceOp");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", opdesc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    EXPECT_EQ(OP_PATTERN_REDUCE, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(2, inputs[0]->GetDataType().size());
    EXPECT_EQ(2, outputs[0]->GetDataType().size());
}

TEST_F(FormatDtypeSelectorManagerKernelUTest, init_customize_format_success) {
    OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithFormat");
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    EXPECT_EQ(OP_PATTERN_OP_CUSTOMIZE, op_kernel_info_ptr->GetOpPattern());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    vector<InputOrOutputInfoPtr> outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, inputs[0]->GetDataType().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetDataType().size());

    op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
    op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    EXPECT_EQ(OP_PATTERN_OP_CUSTOMIZE, op_kernel_info_ptr->GetOpPattern());
    inputs = op_kernel_info_ptr->GetAllInputInfo();
    outputs = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(0, inputs[0]->GetFormat().size());
    EXPECT_EQ(0, inputs[0]->GetDataType().size());
    EXPECT_EQ(0, outputs[0]->GetFormat().size());
    EXPECT_EQ(0, outputs[0]->GetDataType().size());
}

TEST_F(FormatDtypeSelectorManagerKernelUTest, op_kernel_selector_query_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "KernelFormatOp1");
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    FormatDtypeInfo format_dtype_info;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                             false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
    vector<ge::Format> format_vec = format_dtype_info.format_map.at(inputs[0]->GetUniqueName());
    vector<ge::DataType> data_type_vec = format_dtype_info.data_type_map.at(inputs[0]->GetUniqueName());
    EXPECT_EQ(ge::FORMAT_NCHW, format_vec[0]);
    EXPECT_EQ(ge::FORMAT_NC1HWC0, format_vec[1]);
    EXPECT_EQ(ge::FORMAT_ND, format_vec[2]);
    EXPECT_EQ(ge::DT_FLOAT16, data_type_vec[0]);
    EXPECT_EQ(ge::DT_FLOAT, data_type_vec[1]);
    EXPECT_EQ(ge::DT_FLOAT, data_type_vec[2]);
}

TEST_F(FormatDtypeSelectorManagerKernelUTest, op_formatagnostic_selector_query_success)
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
TEST_F(FormatDtypeSelectorManagerKernelUTest, FormatProcessExists_fail)
{
  FormatProccessArgs args;
  Status result = FormatProcessExists(args);
  EXPECT_EQ(result, false);
}

TEST_F(FormatDtypeSelectorManagerKernelUTest, set_node_format_by_custom_dtypes_1)
{
  OpDescPtr op_desc = std::make_shared<OpDesc>("abc", "ABC");
  vector<int64_t> dims(4, 4);
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  desc.SetOriginDataType(ge::DT_FLOAT);
  desc.SetOriginFormat(ge::FORMAT_ND);
  desc.SetOriginShape(shape);
  op_desc->AddInputDesc("x1", desc);
  op_desc->AddInputDesc("x2", desc);
  op_desc->AddInputDesc("x3", desc);
  op_desc->AddOutputDesc("y", desc);
  ge::AttrUtils::SetInt(op_desc, FE_IMPLY_TYPE, 6);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  OpCustomizeDtype op_cust_dtype;
  op_cust_dtype.input_dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  op_cust_dtype.output_dtypes = {ge::DT_FLOAT16};
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_[op_desc->GetType()] = op_cust_dtype;
  std::shared_ptr<FormatDtypeSetter> format_dtype_setter_ptr =
          std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  format_dtype_setter_ptr->SetNodeDtypeByCustomDtypes(node);
  EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
}

TEST_F(FormatDtypeSelectorManagerKernelUTest, set_node_format_by_custom_dtypes_2)
{
  OpDescPtr op_desc = std::make_shared<OpDesc>("abc", "ABC");
  vector<int64_t> dims(4, 4);
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  desc.SetOriginDataType(ge::DT_FLOAT);
  desc.SetOriginFormat(ge::FORMAT_ND);
  desc.SetOriginShape(shape);
  op_desc->AddInputDesc("x1", desc);
  op_desc->AddInputDesc("x2", desc);
  op_desc->AddInputDesc("x3", desc);
  op_desc->AddOutputDesc("y", desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  OpCustomizeDtype op_cust_dtype;
  op_cust_dtype.input_dtypes = {ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_INT32};
  op_cust_dtype.output_dtypes = {ge::DT_FLOAT16, ge::DT_INT64};
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_[op_desc->GetType()] = op_cust_dtype;
  std::shared_ptr<FormatDtypeSetter> format_dtype_setter_ptr =
          std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  format_dtype_setter_ptr->SetNodeDtypeByCustomDtypes(node);
  EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
}

TEST_F(FormatDtypeSelectorManagerKernelUTest, set_node_format_by_custom_dtypes_3)
{
  OpDescPtr op_desc = std::make_shared<OpDesc>("abc", "ABC");
  vector<int64_t> dims(4, 4);
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  desc.SetOriginDataType(ge::DT_FLOAT);
  desc.SetOriginFormat(ge::FORMAT_ND);
  desc.SetOriginShape(shape);
  op_desc->AddInputDesc("x1", desc);
  op_desc->AddInputDesc("x2", desc);
  op_desc->AddInputDesc("x3", desc);
  op_desc->AddOutputDesc("y", desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  OpCustomizeDtype op_cust_dtype;
  op_cust_dtype.input_dtypes = {ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_INT32};
  op_cust_dtype.output_dtypes = {ge::DT_FLOAT16, ge::DT_INT64};
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_[op_desc->GetType()] = op_cust_dtype;
  OpCustomizeDtype op_cust_dtype2;
  op_cust_dtype2.input_dtypes = {ge::DT_INT32};
  op_cust_dtype2.output_dtypes = {ge::DT_INT64, ge::DT_FLOAT16};
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_[op_desc->GetName()] = op_cust_dtype2;
  std::shared_ptr<FormatDtypeSetter> format_dtype_setter_ptr =
          std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  format_dtype_setter_ptr->SetNodeDtypeByCustomDtypes(node);
  EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_INT32);
  EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_INT64);
}
