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
#include <graph/ge_attr_value.h>
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "common/configuration.h"
#include "ops_store/sub_op_info_store.h"
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
using fe::SubOpsStore;



class OpKernelInfoTest : public testing::Test{
  protected:
    static void SetUpTestCase() {
        cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
    }
    static void TearDownTestCase() {
        cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp(){
        op_desc_ptr = make_shared<ge::OpDesc>();
        input0_desc_ptr = make_shared<ge::GeTensorDesc>();
        input1_desc_ptr = make_shared<ge::GeTensorDesc>();
        input2_desc_ptr = make_shared<ge::GeTensorDesc>();
        output0_desc_ptr = make_shared<ge::GeTensorDesc>();
        std::map<std::string, std::string> options;

        sub_ops_store_ptr = make_shared<fe::SubOpsStore>(fe::AI_CORE_NAME);
        op_kernel_info_ptr = make_shared<fe::OpKernelInfo>("conv");
        // FEOpsKernelInfoStore initialize;
        FEOpsStoreInfo tbe_custom {
                  2,
                  "tbe_custom_opinfo",
                  EN_IMPL_CUSTOM_TBE,
                  GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
                  "xx"};
        sub_ops_store_ptr->SetSubStoreInfo(tbe_custom);
        sub_ops_store_ptr->InitializeSubStore();

        sub_ops_kernel_ptr = std::make_shared<fe::SubOpInfoStore>(tbe_custom);
        sub_ops_kernel_ptr->Initialize(fe::AI_CORE_NAME);


        op_desc_ptr->SetName("tbe_conv");
        ge::OpDescUtilsEx::SetType(op_desc_ptr, "conv");
        ge::DataType set_dtype = ge::DT_FLOAT16;
        ge::Format set_format = ge::FORMAT_ND;
        std::vector<int64_t> shape_vec{256,256,512};
        ge::GeShape shape_desc = GeShape(shape_vec);

        input0_desc_ptr->SetDataType(set_dtype);
        input0_desc_ptr->SetFormat(set_format);
        input0_desc_ptr->SetShape(shape_desc);
        op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

        std::vector<int64_t> shape_vec1{256,256,512};
        ge::GeShape shape_desc1 = GeShape(shape_vec1);
        input1_desc_ptr->SetDataType(set_dtype);
        input1_desc_ptr->SetFormat(set_format);
        input1_desc_ptr->SetShape(shape_desc1);
        op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

        std::vector<int64_t> shape_vec2{256,256,512};
        ge::GeShape shape_desc2 = GeShape(shape_vec2);
        input2_desc_ptr->SetDataType(set_dtype);
        input2_desc_ptr->SetFormat(set_format);
        input2_desc_ptr->SetShape(shape_desc2);
        op_desc_ptr->AddInputDesc("x1", input2_desc_ptr->Clone());

        output0_desc_ptr->SetDataType(set_dtype);
        output0_desc_ptr->SetFormat(set_format);
        output0_desc_ptr->SetShape(shape_desc);
        op_desc_ptr->AddOutputDesc("z", output0_desc_ptr->Clone());

        cout << "a test SetUP" << endl;
    }
    virtual void TearDown(){
        cout << "a test SetUP" << endl;
        sub_ops_store_ptr->FinalizeSubStore();
        sub_ops_store_ptr.reset();
        sub_ops_kernel_ptr->Finalize();
        sub_ops_kernel_ptr.reset();
        Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
        config.is_init_ = false;
        config.content_map_.clear();

        map<string, string> options;
        config.Initialize(options);

    }
public:
    shared_ptr<fe::SubOpsStore> sub_ops_store_ptr;
    shared_ptr<fe::SubOpInfoStore> sub_ops_kernel_ptr;
    shared_ptr<fe::OpKernelInfo> op_kernel_info_ptr;
    shared_ptr<ge::GeTensorDesc> input0_desc_ptr;
    shared_ptr<ge::GeTensorDesc> input1_desc_ptr;
    shared_ptr<ge::GeTensorDesc> input2_desc_ptr;
    shared_ptr<ge::GeTensorDesc> output0_desc_ptr;
    shared_ptr<ge::OpDesc> op_desc_ptr;
};

TEST_F(OpKernelInfoTest, op_input_or_output_finalize_twice)
{
    InputOrOutputInfoPtr InfoPtr = std::make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.FinalizeInputAndOutput(InfoPtr);

    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, op_kernel_info_init_and_finalize_twice)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("conv");
    op_info_ptr->init_flag_ = true;
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret1 = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
    Status ret2 = op_kernel_info_constructor.FinalizeOpKernelInfo(op_info_ptr);
    Status ret3 = op_kernel_info_constructor.FinalizeOpKernelInfo(op_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret1);
    EXPECT_EQ(fe::SUCCESS, ret2);
    EXPECT_EQ(fe::SUCCESS, ret3);
}

TEST_F(OpKernelInfoTest, op_tensor_desc_init)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret =  op_kernel_info_constructor.ParseInputAndOutputFromOpContent(op_content, op_kernel_info_ptr);
    EXPECT_NE(0, op_kernel_info_ptr->input_infos_.size());
    EXPECT_NE(0, op_kernel_info_ptr->output_infos_.size());
    op_kernel_info_ptr->init_flag_ = true;
    /* Clear memory */
    ret = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_EQ("", op_kernel_info_ptr->op_type_);
    EXPECT_EQ(true, op_kernel_info_ptr->input_infos_.empty());
    EXPECT_EQ(true, op_kernel_info_ptr->output_infos_.empty());
    EXPECT_EQ(true, op_kernel_info_ptr->attrs_info_.empty());
    EXPECT_EQ(false, op_kernel_info_ptr->init_flag_);
}

TEST_F(OpKernelInfoTest, op_init_attr_value)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitAttrValue(op_content, op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_NE(0, op_kernel_info_ptr->attrs_info_.size());
    EXPECT_EQ(false, op_kernel_info_ptr->attrs_info_[0]->is_support_all_value_);
    EXPECT_EQ("transposX", op_kernel_info_ptr->attrs_info_[0]->attr_name_);
    EXPECT_EQ(true, op_kernel_info_ptr->attrs_info_[0]->is_required_);
    EXPECT_EQ(false, op_kernel_info_ptr->attrs_info_[0]->is_default_value_defined_);
    EXPECT_EQ(ge::GeAttrValue::VT_INT, op_kernel_info_ptr->attrs_info_[0]->dtype_);
    EXPECT_EQ(ge::GeAttrValue::VT_LIST_LIST_INT, op_kernel_info_ptr->attrs_info_[8]->dtype_);

    vector<vector<int64_t>> exp_value = {{1,2,3},{2,3,4},{3,4,5}};
    vector<vector<int64_t>> real_value;
    op_kernel_info_ptr->attrs_info_[8]->GetDefaultValue().GetValue<vector<vector<int64_t>>>(real_value);
    EXPECT_EQ(exp_value, real_value);


    op_kernel_info_ptr->init_flag_ = true;
    /* Clear memory */
    ret = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_EQ("", op_kernel_info_ptr->op_type_);
    EXPECT_EQ(true, op_kernel_info_ptr->input_infos_.empty());
    EXPECT_EQ(true, op_kernel_info_ptr->output_infos_.empty());
    EXPECT_EQ(true, op_kernel_info_ptr->attrs_info_.empty());
    EXPECT_EQ(false, op_kernel_info_ptr->init_flag_);

}

TEST_F(OpKernelInfoTest, op_init_op_info)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitOpInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    op_kernel_info_ptr->init_flag_ = true;
    /* Clear memory */
    ret = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_EQ("", op_kernel_info_ptr->op_type_);
    EXPECT_EQ(0, op_kernel_info_ptr->input_infos_.size());
    EXPECT_EQ(0, op_kernel_info_ptr->output_infos_.size());
    EXPECT_EQ(0, op_kernel_info_ptr->attrs_info_.size());
    EXPECT_EQ(false, op_kernel_info_ptr->init_flag_);
    EXPECT_EQ(false, op_kernel_info_ptr->IsMultiKernelSupport());
}

TEST_F(OpKernelInfoTest, op_info_initialize)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);

    EXPECT_EQ(fe::SUCCESS, ret);
    /* Clear memory */
    ret = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_EQ("", op_kernel_info_ptr->op_type_);
    EXPECT_EQ(0, op_kernel_info_ptr->input_infos_.size());
    EXPECT_EQ(0, op_kernel_info_ptr->output_infos_.size());
    EXPECT_EQ(0, op_kernel_info_ptr->attrs_info_.size());
    EXPECT_EQ(false, op_kernel_info_ptr->init_flag_);

}

TEST_F(OpKernelInfoTest, input_or_output_info_getter)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    InputOrOutputInfoPtr input0;
    op_kernel_info_ptr->GetInputInfoByName("x", input0);
    input0->GetName();
    input0->GetDataType();
    input0->GetFormat();
    EXPECT_EQ(DYNAMIC, input0->GetParamType());

    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, get_all_input_or_output_info)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    const vector<InputOrOutputInfoPtr> all_input_info = op_kernel_info_ptr->GetAllInputInfo();
    const vector<InputOrOutputInfoPtr> all_output_info = op_kernel_info_ptr->GetAllOutputInfo();
    EXPECT_EQ(3, all_input_info.size());
    EXPECT_EQ(1, all_output_info.size());

    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, get_input_or_output_info_by_name)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    InputOrOutputInfoPtr input_ptr;
    Status ret1 = op_kernel_info_ptr->GetInputInfoByName("x", input_ptr);
    ASSERT_EQ(fe::SUCCESS, ret1);
    EXPECT_EQ("x", input_ptr->GetName());

    InputOrOutputInfoPtr output_ptr;
    Status ret2 = op_kernel_info_ptr->GetOutputInfoByName("z", output_ptr);
    ASSERT_EQ(fe::SUCCESS, ret2);
    EXPECT_EQ("z", output_ptr->GetName());

    InputOrOutputInfoPtr input1_ptr;
    Status ret3 = op_kernel_info_ptr->GetInputInfoByName("x1", input1_ptr);
    EXPECT_NE(fe::SUCCESS, ret3);

    InputOrOutputInfoPtr input2_ptr;
    Status ret4 = op_kernel_info_ptr->GetInputInfoByName("m", input2_ptr);
    EXPECT_NE(fe::SUCCESS, ret4);

    Status ret5 = op_kernel_info_ptr->GetOutputInfoByName("m", input2_ptr);
    EXPECT_NE(fe::SUCCESS, ret5);

    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, get_op_type)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    string op_type = op_kernel_info_ptr->GetOpType();
    EXPECT_NE("abs", op_type);
    EXPECT_EQ("conv", op_type);

    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, get_unknown_shape_format)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    EXPECT_EQ(ret, SUCCESS);
    string op_type = op_kernel_info_ptr->GetOpType();
    for (auto input_info_ptr : op_kernel_info_ptr->GetAllInputInfo()) {
        EXPECT_EQ(input_info_ptr->GetUnknownShapeFormat().size(), 4);
        EXPECT_EQ(input_info_ptr->GetUnknownShapeDataType().size(), 4);
        for (auto dtype : input_info_ptr->GetUnknownShapeDataType()) {
            EXPECT_NE(dtype, ge::DT_FLOAT16);
        }
    }

    for (auto output_info_ptr : op_kernel_info_ptr->GetAllOutputInfo()) {
        EXPECT_EQ(output_info_ptr->GetUnknownShapeFormat().size(), 4);
        EXPECT_EQ(output_info_ptr->GetUnknownShapeDataType().size(), 4);
        for (auto dtype : output_info_ptr->GetUnknownShapeDataType()) {
            EXPECT_NE(dtype, ge::DT_FLOAT);
        }
    }
    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, get_unknown_shape_format_format_agnostic_op)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("formatAgnosticOp", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    EXPECT_EQ(ret, SUCCESS);
    for (auto input_info_ptr : op_kernel_info_ptr->GetAllInputInfo()) {
        EXPECT_GT(input_info_ptr->GetUnknownShapeFormat().size(), 10);
        EXPECT_GT(input_info_ptr->GetUnknownShapeDataType().size(), 10);
    }

    for (auto output_info_ptr : op_kernel_info_ptr->GetAllOutputInfo()) {
        EXPECT_GT(output_info_ptr->GetUnknownShapeFormat().size(), 10);
        EXPECT_GT(output_info_ptr->GetUnknownShapeDataType().size(), 10);
    }
    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, test_finalize)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    Status ret1 = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret1);
}

TEST_F(OpKernelInfoTest, test_get_attr_value)
{
    OpContent op_content;
    sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_NE(0, op_kernel_info_ptr->attrs_info_.size());

    EXPECT_NE(0, op_kernel_info_ptr->attrs_info_[0]->supported_values_.size());
    cout << "====================================the size of attrs_info_ = " << op_kernel_info_ptr->attrs_info_.size() <<endl;
    //vector<ge::GeAttrValue> attr_vec = op_kernel_info_ptr->map_attr_value_.at("attrListInt");
   // vector<ge::GeAttrValue> attr_int_vec = op_kernel_info_ptr->map_attr_value_.at("transposX");
    for (auto el : op_kernel_info_ptr->attrs_info_) {
        cout <<"================================key of attrs_info_  : " << el->attr_name_ << endl;
    }
    // for (auto it:attr_int_vec){
    //     int64_t int_value;
    //     it.GetValue<int64_t>(int_value);
    //     cout <<"the value from op_info "<< int_value << endl;
    // }
    // EXPECT_NE(0, attr_vec.size());
    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}
TEST_F(OpKernelInfoTest, test_dtype_format_special)
{
    OpKernelInfoPtr op_info_ptr = std::make_shared<fe::OpKernelInfo>("conv");
    map<string, string> options;
    // FEOpsKernelInfoStore initialize;
    FEOpsStoreInfo utest_opinfo {
              2,
              "utest_opinfo",
              EN_IMPL_CUSTOM_TBE,
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/utest_opinfo",
              "xx"};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(utest_opinfo);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(fe::AI_CORE_NAME).is_init_ = false;
    OpsKernelManager::Instance(fe::AI_CORE_NAME).Initialize();
    SubOpInfoStorePtr subOpInfoStorePtr = OpsKernelManager::Instance(fe::AI_CORE_NAME).GetSubOpsKernelByStoreName("utest_opinfo");
    OpContent op_content;
    subOpInfoStorePtr->GetOpContentByOpType("conv", op_content);
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_kernel_info_ptr);
    EXPECT_EQ(fe::SUCCESS, ret);
    op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info_ptr);
}

TEST_F(OpKernelInfoTest, test_get_str_from_op_content)
{
    OpContent op_content;
    op_content.op_type_ = "conv";
    map<string, string> cost{{"cost", "10"}};
    map<string, string> flag{{"flag", "true"}};
    op_content.map_kernel_info_.emplace(make_pair("compute", cost));
    op_content.map_kernel_info_.emplace(make_pair("async", flag));
    string value;
   // std::shared_ptr<fe::OpKernelInfo> info_store_ptr = std::make_shared<fe::OpKernelInfo>();
    OpKernelInfoConstructor op_kernel_info_constructor;
    Status ret1 = op_kernel_info_constructor.GetStrFromOpContent(op_content, "partial", "flag", value);
    Status ret2 = op_kernel_info_constructor.GetStrFromOpContent(op_content, "compute", "flag", value);
    EXPECT_NE(fe::SUCCESS, ret1);
    EXPECT_NE(fe::SUCCESS, ret2);
}

TEST_F(OpKernelInfoTest, init_tensor_desc_info_param_type)
{
    map<string, string> map_info{{"name", "x"}, {"paramtype","not_exist"}, {"dtype","float16,float,double,int32"},
                                {"format","NCHW"}, {"shape", "[256,256,512]"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr,temp, op_kernel_info_ptr);
    EXPECT_NE(fe::SUCCESS, ret);

}

TEST_F(OpKernelInfoTest, init_tensor_desc_info_tunformat_switch)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","dynamic"}, {"dtype","float16"},
                                {"format","NCHW"}, {"shape", "[256,256,512]"}, {"tuneformatSwitch", "true"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr,temp, op_kernel_info_ptr);
    EXPECT_EQ(info_ptr->GetTuneFormatSwitch(), true);
    EXPECT_EQ(fe::SUCCESS, ret);
    map_info = {{"name", "x"}, {"paramType","dynamic"}, {"dtype","float16"},
               {"format","NCHW"}, {"shape", "[256,256,512]"}, {"tuneformatSwitch", "not_exist"}};
    ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr,temp, op_kernel_info_ptr);
    EXPECT_NE(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, init_unknownshape_format)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","required"}, {"shape", "all"},
                                 {"dtype","float,int8,bool,float16"}, {"format","FRACTAL_NZ,NHWC,FRACTAL_NZ,NHWC"},
                                 {"unknownshape_format", "NCHW,ND,ND,NC1HWC0"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr, temp, op_kernel_info_ptr);
    std::vector<ge::Format> unknownshape_format = {Format::FORMAT_NCHW,Format::FORMAT_ND,Format::FORMAT_ND,Format::FORMAT_NC1HWC0};
    std::vector<ge::DataType> unknownshape_dtype = {DataType::DT_FLOAT,DataType::DT_INT8,DataType::DT_BOOL,DataType::DT_FLOAT16};
    EXPECT_EQ(info_ptr->GetUnknownShapeFormat(), unknownshape_format);
    EXPECT_EQ(info_ptr->GetUnknownShapeDataType(), unknownshape_dtype);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, init_unknownshape_format_2)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","required"}, {"shape", "all"},
                                 {"dtype","float,int8,bool,float16"}, {"format","FRACTAL_NZ,NHWC,FRACTAL_NZ,NHWC"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr, temp, op_kernel_info_ptr);
    std::vector<ge::Format> unknownshape_format = {Format::FORMAT_FRACTAL_NZ,Format::FORMAT_NHWC,Format::FORMAT_FRACTAL_NZ,Format::FORMAT_NHWC};
    std::vector<ge::DataType> unknownshape_dtype = {DataType::DT_FLOAT,DataType::DT_INT8,DataType::DT_BOOL,DataType::DT_FLOAT16};
    EXPECT_EQ(info_ptr->GetUnknownShapeFormat(), unknownshape_format);
    EXPECT_EQ(info_ptr->GetUnknownShapeDataType(), unknownshape_dtype);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, init_unknownshape_format_3)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","required"}, {"shape", "all"},
                                 {"dtype","float,int8,bool,float16"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr, temp, op_kernel_info_ptr);
    std::vector<ge::Format> unknownshape_format = {};
    std::vector<ge::DataType> unknownshape_dtype = {};
    EXPECT_EQ(info_ptr->GetUnknownShapeFormat(), unknownshape_format);
    EXPECT_EQ(info_ptr->GetUnknownShapeDataType(), unknownshape_dtype);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, init_unknownshape_format_fail_4)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","required"}, {"shape", "all"},
                                 {"format","FRACTAL_NZ,NHWC,FRACTAL_NZ,NHWC"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr, temp, op_kernel_info_ptr);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(OpKernelInfoTest, init_unknownshape_format_fail_5)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","required"}, {"shape", "all"},
                                 {"unknownshape_format", "NCHW,ND,ND,NC1HWC0"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_BROADCAST, "xOp", map_info, 0, info_ptr, temp, op_kernel_info_ptr);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(OpKernelInfoTest, init_unknownshape_format_fail_6)
{
    map<string, string> map_info{{"name", "x"}, {"paramType","required"}, {"shape", "all"},
                                 {"dtype","float,int8,bool,float16"}, {"unknownshape_format", "ND,ND,NC1HWC0"}};

    shared_ptr<InputOrOutputInfo> info_ptr = make_shared<InputOrOutputInfo>("x");
    OpKernelInfoConstructor op_kernel_info_constructor;
    uint32_t temp = 0xffffffff;
    Status ret = op_kernel_info_constructor.InitializeInputAndOutput(OP_PATTERN_OP_KERNEL, "xOp", map_info, 0, info_ptr, temp, op_kernel_info_ptr);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(OpKernelInfoTest, test_convert_list_attr_default_value)
{
  OpContent op_content;
  op_content.op_type_ = "otherNode";
  map<std::string, std::string> in_info_map;
  in_info_map.emplace(std::make_pair("defaultValue", "[1,2]"));
  op_content.map_kernel_info_.emplace(std::make_pair("attr_x1", in_info_map));

  string attr_name = "x1";
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_INT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_support_all_value_ = true;

  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitAttrListTemplate<int64_t>(op_content, attr_name, attr_info_ptr);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, test_convert_list_attr_default_value_failed)
{
  OpContent op_content;
  op_content.op_type_ = "otherNode";
  map<std::string, std::string> in_info_map;
  op_content.map_kernel_info_.emplace(std::make_pair("attr_x1", in_info_map));

  string attr_name = "x1";
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_INT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_support_all_value_ = true;

  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitAttrListTemplate<int64_t>(op_content, attr_name, attr_info_ptr);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_1)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("conv");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "AiCore");
}

TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_2)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("conv3", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("conv3");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "cube");
}

TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_3)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("A", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("A");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "VectorCore");
}

TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_4)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("AA", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("AA");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "");
}

TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_5)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("B", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("B");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "Mix");
}


TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_6)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("BB", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("BB");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "mix");
}

TEST_F(OpKernelInfoTest, op_kernel_info_core_type_test_7)
{
  OpContent op_content;
  sub_ops_kernel_ptr->GetOpContentByOpType("C", op_content);
  OpKernelInfoPtr op_info_ptr = std::make_shared<OpKernelInfo>("C");
  op_info_ptr->init_flag_ = false;
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content, op_info_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_info_ptr->GetCoreType(), "dynamic");
}

TEST_F(OpKernelInfoTest, op_kernel_info_constructor_5hd)
{
  std::vector<ge::DataType> dtypes;
  dtypes.emplace_back(ge::DT_FLOAT);
  dtypes.emplace_back(ge::DT_UINT32);
  dtypes.emplace_back(ge::DT_STRING);
  std::vector<ge::Format> formats;
  formats.emplace_back(ge::FORMAT_NCHW);
  formats.emplace_back(ge::FORMAT_NC1HWC0_C04);
  formats.emplace_back(ge::FORMAT_NC1HWC0);
  InputOrOutputInfoPtr InfoPtr0 = std::make_shared<InputOrOutputInfo>("x0");
  InputOrOutputInfoPtr InfoPtr1 = std::make_shared<InputOrOutputInfo>("x1");
  InputOrOutputInfoPtr InfoPtr2 = std::make_shared<InputOrOutputInfo>("x2");
  InputOrOutputInfoPtr OutPtr = std::make_shared<InputOrOutputInfo>("y");
  InfoPtr0->supported_dtypes_ = dtypes;
  InfoPtr1->supported_dtypes_ = dtypes;
  InfoPtr2->supported_dtypes_ = dtypes;
  OutPtr->supported_dtypes_ = dtypes;
  InfoPtr0->supported_formats_ = formats;
  InfoPtr1->supported_formats_ = formats;
  InfoPtr2->supported_formats_ = formats;
  OutPtr->supported_formats_ = formats;
  std::vector<InputOrOutputInfoPtr> inputs;
  std::vector<InputOrOutputInfoPtr> outputs;
  inputs.emplace_back(InfoPtr0);
  inputs.emplace_back(InfoPtr1);
  inputs.emplace_back(InfoPtr2);
  outputs.emplace_back(OutPtr);
  OpKernelInfoConstructor op_kernel_info_constructor;
  bool ret = op_kernel_info_constructor.RemoveUnknownShapeAndShapeHeavyFormatSupportWhenC0Change(inputs, outputs);
  EXPECT_EQ(ret, false);
}

TEST_F(OpKernelInfoTest, parse_fallback)
{
    OpContent op_content;
    op_content.op_type_ = "TestOp";
    std::map<std::string, std::string> fallback_map;
    fallback_map["enable"] = "1,0,0,1";
    op_content.map_kernel_info_["fallback"] = fallback_map;
    OpKernelInfoConstructor op_kernel_info_constructor;
    std::vector<bool> cfg_fallbacks;
    std::vector<bool> res_fallbacks = {true, false, false, true};
    Status ret = op_kernel_info_constructor.ParseFallbackFlags(op_content, "enable", cfg_fallbacks);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_EQ(res_fallbacks, cfg_fallbacks);
}
