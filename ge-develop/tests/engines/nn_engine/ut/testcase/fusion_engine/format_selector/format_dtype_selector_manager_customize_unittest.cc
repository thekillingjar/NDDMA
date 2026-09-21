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
#define private public
#define protected public
#include "common/util/op_info_util.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/platform_utils.h"
#include "graph/ge_tensor.h"
#include "fusion_manager/fusion_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "ops_store/ops_kernel_manager.h"
#include "adapter/common/op_store_adapter_manager.h"
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


class FormatDtypeSelectorManagerCustomizeUTest : public testing::Test {
 protected:

  static void SetUpTestCase() {
    cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
    std::string soc_version = "Ascend310P3";
    PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
    PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
    PlatformUtils::Instance().soc_version_ = soc_version;
  }

  static void TearDownTestCase() {
    cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
  }

  // Some expensive resource shared by all tests.
  virtual void SetUp() {
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
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_->Initialize(options);
    auto reflection_builder_ptr = std::make_shared<ge::RefRelations>();
    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(
        fe::AI_CORE_NAME, reflection_builder_ptr);
    op_format_dtype_judge_ptr_->Initialize();
    conv_input_name_map_ = {
        {0, "x"},
        {1, "y"},
        {2, "z"}
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

  virtual void TearDown() {
    cout << "a test Tear Down" << endl;
    fe_ops_kernel_info_store_ptr_->Finalize();
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
};

bool SelectTbeOpFormatStub(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str)
{
    op_format_dtype_str = "{\"input0\":{\"name\":\"x\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"},\"output0\":{\"name\":\"y\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"}}";
    return true;
}

bool SelectTbeOpFormatStub2(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str)
{
    op_format_dtype_str = "";
    return true;
}

bool SelectTbeOpFormatUnknownStub(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str)
{
    op_format_dtype_str = "{\"input0\":{\"name\":\"x\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"},\"output0\":{\"name\":\"y\",\"format\":\"ND,NCHW\",\"unknownshape_format\":\"ND,NCHW\",\"dtype\":\"float,float16\"}}";
    return true;
}

bool SelectTbeOpFormatStub1(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str)
{
    std::vector<te::TbeOpParam> inputs;
    inputs = tbe_op_info.inputs_;
    if (!inputs.empty()) {
        std::vector<te::TbeOpTensor> tensors;
        inputs[0].GetTensors(tensors);
        if (!tensors.empty()) {
          bool is_first_layer;
          if (tensors[0].GetFirstLayer(is_first_layer) && is_first_layer) {
            // Check if it's the first layer conv
            // the first layer conv can use c04 format
            op_format_dtype_str = "{\"input0\":{\"name\":\"x\",\"format\":\"NC1HWC0_C04,NC1HWC0_C04\", \"dtype\":\"float,float16\"},\"output0\":{\"name\":\"y\",\"format\":\"NC1HWC0_C04,NC1HWC0_C04\", \"dtype\":\"float,float16\"}}";
            return true;
          }
        }
    }
    op_format_dtype_str = "{\"input0\":{\"name\":\"x\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"},\"output0\":{\"name\":\"y\",\"format\":\"ND,NCHW\", \"dtype\":\"float,float16\"}}";
    return true;
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, op_customize_selector_query_success) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
  vector<int64_t> dim_data = {1, 2, 3, 4};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", data_desc);
  op_desc_ptr->AddOutputDesc("y", data_desc);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin",
                                                                                                        op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
  tbe_op_store_adapter_ptr->SelectTbeOpFormat = SelectTbeOpFormatStub;

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, op_customize_selector_query_failed)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
    vector<int64_t> dim_data = {1, 2, 3, 4};
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, FORMAT_NCHW, DT_FLOAT);
    op_desc_ptr->AddInputDesc("x", data_desc);
    op_desc_ptr->AddOutputDesc("y", data_desc);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());

    TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_op_store_adapter_ptr->SelectTbeOpFormat = SelectTbeOpFormatStub2;

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                                  false, format_dtype_info);
    EXPECT_EQ(fe::FAILED, result);
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, unknown_shape_op_customize_selector_query_success)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("test", "DynamicFormatOpWithoutFormat");
    vector<int64_t> dim_data = {1, -1, -1, 4};
    GeShape shape_data(dim_data);
    GeTensorDesc data_desc(shape_data, FORMAT_NCHW, DT_FLOAT);
    op_desc_ptr->AddInputDesc("x", data_desc);
    op_desc_ptr->AddOutputDesc("y", data_desc);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_op_store_adapter_ptr->SelectTbeOpFormat = SelectTbeOpFormatUnknownStub;

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);

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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, ConvertFormatDtype_failed1)
{
  string format_vec_str;
  string sub_format_vec_str;
  string data_type_vec_str = "DT_INT16, DT_INT16";
  uint32_t format_size_of_first_input_or_output;
  vector<ge::Format> format_vec;
  vector<uint32_t> sub_format_vec;
  vector<ge::DataType> data_type_vec;
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
        std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  Status status = customize_selector_ptr->ConvertFormatDtype(format_vec_str, sub_format_vec_str, data_type_vec_str,
                                                             format_size_of_first_input_or_output, format_vec,
                                                             sub_format_vec, data_type_vec);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, ConvertFormatDtype_failed2)
{
  string format_vec_str;
  string sub_format_vec_str;
  string data_type_vec_str;
  uint32_t format_size_of_first_input_or_output = 0;
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, ConvertFormatDtype_success)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, UpdateDTypeAndFormat_suc)
{
  set<uint32_t> remain_index_set;
  std::map<std::string, vector<ge::Format>> format_map;
  std::map<std::string, vector<uint32_t>> sub_format_map;
  std::map<std::string, vector<ge::DataType>> data_type_map;

  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
        std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  Status status = customize_selector_ptr->UpdateDTypeAndFormat(remain_index_set, format_map,
                                                               sub_format_map, data_type_map);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, UpdateDTypeAndFormat_suc2)
{
  set<uint32_t> remain_index_set = {1};
  std::map<std::string, vector<ge::Format>> format_map;
  std::map<std::string, vector<uint32_t>> sub_format_map = {
    {"test1", {}}, {"test2", {2,4,6}}
  };
  std::map<std::string, vector<ge::DataType>> data_type_map;

  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
        std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  Status status = customize_selector_ptr->UpdateDTypeAndFormat(remain_index_set, format_map,
                                                               sub_format_map, data_type_map);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_01) {
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_02)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_03)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_04)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_05) {
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_no_first_layer_conv2d_disable_small_chn)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_no_first_layer_conv2d_enable_small_chn)
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


TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_no_first_layer_conv2d_enable_small_chn_only_output_c04)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_no_first_layer_normal_op_enable_small_chn)
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

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, filter_c04_format_first_layer_normal_op_enable_small_chn)
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
  map<string, vector<ge::Format>> format_map_res = {};
  customize_selector_ptr->GetAllSupportFormat(conv_kernel_, conv_node_, false, format_map_res);
  EXPECT_EQ(format_map_res, format_map);
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, parse_json_fallback)
{
  nlohmann::json json_object;
  nlohmann::json fallback_json;
  fallback_json["enable"] = "1,1,0,0";
  fallback_json["unknownshape_enable"] = "1,1,0,0";
  json_object["fallback"] = fallback_json;
  FormatDtypeInfo res;
  std::shared_ptr<FormatDtypeOpCustomizeSelector> customize_selector_ptr =
      std::make_shared<FormatDtypeOpCustomizeSelector>(AI_CORE_NAME);
  Status status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::SUCCESS);
  nlohmann::json fallback_fail_json;
  fallback_fail_json["invalid"] = "";
  json_object["fallback"] = fallback_fail_json;
  status = customize_selector_ptr->ParseJsonStr(conv_node_, json_object.dump(), true, res);
  EXPECT_EQ(status, fe::FAILED);
  conv_op_->DelExtAttr("ext_dynamic_format");
  conv_op_->DelExtAttr("ext_dynamic_datatype");
  map<string, vector<ge::Format>> format_map_res = {};
  status = customize_selector_ptr->GetAllSupportFormat(conv_kernel_, conv_node_, false, format_map_res);
  EXPECT_EQ(status, fe::SUCCESS);
  std::vector<bool> fallback_res;
  status = customize_selector_ptr->GetFallbackFlags(conv_kernel_, conv_node_, false, fallback_res);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(FormatDtypeSelectorManagerCustomizeUTest, parse_json_fail)
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