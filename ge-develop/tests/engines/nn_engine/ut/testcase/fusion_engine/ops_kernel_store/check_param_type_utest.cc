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

#define protected public
#define private   public

#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "common/fe_op_info_common.h"
using namespace testing;
using namespace fe;
using namespace std;

using fe::FEOpsKernelInfoStore;
using ge::GeTensorDesc;
using ge::GeTensorDescPtr;
using ge::GeShape;
using ge::OpDescPtr;
using ge::OpDesc;
using ge::AttrUtils;
using ge::Format;
using ge::DataType;
using fe::InputOrOutputInfo;
using fe::SubOpsStore;

enum TestIter
{
    TEST_SUCCESS = 0,
    TEST_SHAPE_NOT_MATCH,
    TEST_FORMAT_NOT_MATCH,
    TEST_DATA_TYPE_NOT_MATCH,
    TEST_REQUIRED_MISS,
    TEST_REQUIRED_MISS_2,
    TEST_DYNAMIC_MISS,
    TEST_DYNAMIC_MISS_2,
    TEST_DYNAMIC_FOUND_2,
    TEST_OPTIONAL_MISS,
};

static map<string, string> IN0_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "x"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {"paramType", "required"},
};

static map<string, string> IN1_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "y"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {"paramType", "dynamic"},
};

static map<string, string> IN2_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "z"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {"paramType", "optional"},
};

static map<string, string> COST_INFO{
    {"cost", "10"},
};

static map<string, string> FLAG_INFO{
    {"flag", "true"},
};
class UTEST_FE_CHECK_PARAM_TYPE : public testing::Test {
protected:
    void SetUp()
    {
        op_content_ = {"FrameworkOp", "", {{"compute", COST_INFO},
                                      {"partial",FLAG_INFO},
                                      {"async",FLAG_INFO},
                                      {"input0", IN0_TENSOR_DESC_INFO_MAP},
                                      {"input1", IN1_TENSOR_DESC_INFO_MAP},
                                      {"input2", IN2_TENSOR_DESC_INFO_MAP},
                                      {"output0", IN0_TENSOR_DESC_INFO_MAP},
                                      {"output1", IN1_TENSOR_DESC_INFO_MAP},
                                      {"output2", IN2_TENSOR_DESC_INFO_MAP}}};
    }

    void TearDown()
    {

    }

    static OpDescPtr GenerateOpDesc(TestIter test_iter)
    {
        OpDescPtr desc_ptr = std::make_shared<OpDesc>("test", "FrameworkOp");
        GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
        /*
        * set TensorDesc
        */
        // TEST_SHAPE_NOT_MATCH
        ge::GeShape shape;
        if (test_iter == TEST_SHAPE_NOT_MATCH) {
            ge::GeShape shape({4, 5, 6});
            general_ge_tensor_desc->SetShape(shape);
        } else {
            ge::GeShape shape({1, 2, 3});
            general_ge_tensor_desc->SetShape(shape);
        }

        // TEST_FORMAT_NOT_MATCH
        if (test_iter == TEST_FORMAT_NOT_MATCH) {
            general_ge_tensor_desc->SetFormat(Format::FORMAT_NHWC);
        } else {
            general_ge_tensor_desc->SetFormat(Format::FORMAT_NCHW);
        }
        // TEST_DATA_TYPE_NOT_MATCH
        if (test_iter == TEST_DATA_TYPE_NOT_MATCH) {
            general_ge_tensor_desc->SetDataType(DataType::DT_INT32);
        } else {
            general_ge_tensor_desc->SetDataType(DataType::DT_FLOAT);
        }
        /*
        * add input desc to OpDesc
        */
        // TEST_REQUIRED_MISS
        if (test_iter != TEST_REQUIRED_MISS) {
            desc_ptr->AddInputDesc("x", general_ge_tensor_desc->Clone());
        }
        // TEST_DYNAMIC_MISS
        if (test_iter == TEST_DYNAMIC_MISS) {
            printf("TEST_DYNAMIC_MISS\n");
        // TEST_DYNAMIC_FOUND_2
        } else if (test_iter == TEST_DYNAMIC_FOUND_2) {
            desc_ptr->AddInputDesc("y1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("y2", general_ge_tensor_desc->Clone());
        } else {
            desc_ptr->AddInputDesc("y1", general_ge_tensor_desc->Clone());
        }
        // TEST_OPTIONAL_MISS
        if (test_iter != TEST_OPTIONAL_MISS) {
            desc_ptr->AddInputDesc("z", general_ge_tensor_desc->Clone());
        }
        /*
        * add output desc to OpDesc
        */
        // TEST_REQUIRED_MISS
        if (test_iter != TEST_REQUIRED_MISS && test_iter != TEST_REQUIRED_MISS_2) {
            desc_ptr->AddOutputDesc("x", general_ge_tensor_desc->Clone());
        }
        // TEST_DYNAMIC_MISS
        if (test_iter == TEST_DYNAMIC_MISS || test_iter == TEST_DYNAMIC_MISS_2) {
            printf("TEST_DYNAMIC_MISS\n");
        // TEST_DYNAMIC_FOUND_2
        } else if (test_iter == TEST_DYNAMIC_FOUND_2) {
            desc_ptr->AddOutputDesc("y1", general_ge_tensor_desc->Clone());
            desc_ptr->AddOutputDesc("y2", general_ge_tensor_desc->Clone());
        } else {
            desc_ptr->AddOutputDesc("y1", general_ge_tensor_desc->Clone());
        }
        // TEST_OPTIONAL_MISS
        if (test_iter != TEST_OPTIONAL_MISS) {
            desc_ptr->AddOutputDesc("z", general_ge_tensor_desc->Clone());
        }

        return desc_ptr;
    }

    fe::OpContent op_content_;
};

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_success)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_SUCCESS);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(true, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, ParseBasicParameterOutputInplace)
{
    fe::OpContent op_content;
    op_content.op_type_ = "InplaceA";
    op_content.ops_path_name_prefix_ = "";
    op_content.map_kernel_info_ = {{"outputInplaceAbility", {{"flag", "{{0, 0}, {0, 1}}"}}}};
                                
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("InplaceA");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.ParseOpOutputInplaceAbility(op_content, op_kernel_info_ptr);
    std::vector<std::vector<int64_t>> output_inplace = op_kernel_info_ptr->GetOutputIplaceInfo();
    std::vector<std::vector<int64_t>> golden = {{0, 0}, {0, 1}};
    EXPECT_EQ(output_inplace, golden);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_required_miss)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_REQUIRED_MISS);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(false, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_required_miss_2)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_REQUIRED_MISS_2);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(false, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_dynamic_miss)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_DYNAMIC_MISS);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(false, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_dynamic_miss_2)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_DYNAMIC_MISS_2);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(false, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_dynamic_found_2)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_DYNAMIC_FOUND_2);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(true, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_param_type_optional_miss)
{
    auto op_desc_ptr = GenerateOpDesc(TEST_OPTIONAL_MISS);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_, op_kernel_info_ptr);

    SubOpsStore test_subject(AI_CORE_NAME);

    map<uint32_t, string> input_index_name_map;
    GetInputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), input_index_name_map);

    map<uint32_t, string> output_index_name_map;
    GetOutputIndexNameMap(*op_desc_ptr, *(op_kernel_info_ptr.get()), output_index_name_map);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    bool result = test_subject.CheckParamType(test_node, op_kernel_info_ptr,
                                             input_index_name_map, output_index_name_map, reason);

    EXPECT_EQ(true, result);
}

TEST_F(UTEST_FE_CHECK_PARAM_TYPE, check_sub_string_function_1)
{
    EXPECT_EQ(true, fe::CheckInputSubString("123", "12"));
    EXPECT_EQ(false, fe::CheckInputSubString("123x", "12"));
    EXPECT_EQ(false, fe::CheckInputSubString("1", "12"));
    EXPECT_EQ(false, fe::CheckInputSubString("x", "12"));

    EXPECT_EQ(true, fe::CheckInputSubString("x1", "x"));
    EXPECT_EQ(false, fe::CheckInputSubString("xx", "x"));
    EXPECT_EQ(false, fe::CheckInputSubString("xx1", "x"));
    EXPECT_EQ(true, fe::CheckInputSubString("xx1", "xx"));
    EXPECT_EQ(false, fe::CheckInputSubString("xx1", "xxxx"));
    EXPECT_EQ(false, fe::CheckInputSubString("1xx1", "xx"));
    EXPECT_EQ(false, fe::CheckInputSubString("xx1", "ww"));
    EXPECT_EQ(false, fe::CheckInputSubString("wwxx1", "ww"));
    EXPECT_EQ(false, fe::CheckInputSubString("wwx", "ww"));
    EXPECT_EQ(true, fe::CheckInputSubString("asdv123", "asdv"));
    EXPECT_EQ(false, fe::CheckInputSubString("as", "asdv"));

}


