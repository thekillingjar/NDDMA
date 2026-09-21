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
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/sub_ops_store.h"
using namespace testing;
using namespace fe;
using namespace std;

using fe::FEOpsKernelInfoStore;
using fe::SubOpsStore;
using ge::GeTensorDesc;
using ge::GeTensorDescPtr;
using ge::GeShape;
using ge::OpDescPtr;
using ge::OpDesc;
using ge::AttrUtils;
using ge::Format;
using ge::DataType;
using fe::InputOrOutputInfoPtr;

enum TestIter
{
    TEST_SUCCESS = 0,
    TEST_SHAPE_NOT_MATCH,
    TEST_FORMAT_NOT_MATCH,
    TEST_DATA_TYPE_NOT_MATCH,
    TEST_REQUIRED_MISS,
    TEST_DYNAMIC_MISS,
    TEST_DYNAMIC_FOUND_2,
    TEST_OPTIONAL_MISS,
    TEST_REQUIRED_DUPLICATE_NAME,
    TEST_REQUIRED_DUPLICATE_NAME_XX,
    TEST_DYNAMIC_DUPLICATE_NAME_XX,
    TEST_DYNAMIC_DUPLICATE_NAME_XX2,
};

static const std::string STR_PARAM_TYPE = "paramType";

static map<string, string> IN0_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "x"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> IN1_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "y"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "dynamic"},
};

static map<string, string> IN2_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "z"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "optional"},
};

static map<string, string> IN3_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "x1"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> IN4_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "x2"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> IN5_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "x3"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> IN6_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "x4"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> IN7_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "xx"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> IN8_TENSOR_DESC_INFO_MAP{
    {STR_NAME, "xxx"},
    {STR_DTYPE, "float"},
    {STR_FORMAT, "NCHW"},
    {STR_PARAM_TYPE, "required"},
};

static map<string, string> COST_INFO{
    {"cost", "10"},
};

static map<string, string> FLAG_INFO{
    {"flag", "true"},
};
class STEST_FE_CHECK_PARAM_TYPE : public testing::Test {
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
        if (test_iter == TEST_REQUIRED_DUPLICATE_NAME) {
            desc_ptr->AddInputDesc("x1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("x2", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("x3", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("x4", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_REQUIRED_DUPLICATE_NAME_XX) {
            desc_ptr->AddInputDesc("y", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx2", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_DYNAMIC_DUPLICATE_NAME_XX) {
            desc_ptr->AddInputDesc("y", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx2", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_DYNAMIC_DUPLICATE_NAME_XX2) {
            desc_ptr->AddInputDesc("x1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xxx2", general_ge_tensor_desc->Clone());
        }
        // TEST_REQUIRED_MISS
        if (test_iter != TEST_REQUIRED_MISS) {
            desc_ptr->AddInputDesc("x", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_REQUIRED_DUPLICATE_NAME) {
            desc_ptr->AddInputDesc("x1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("x2", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("x3", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("x4", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_REQUIRED_DUPLICATE_NAME_XX) {
            desc_ptr->AddInputDesc("y", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx2", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_DYNAMIC_DUPLICATE_NAME_XX) {
            desc_ptr->AddInputDesc("y", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx2", general_ge_tensor_desc->Clone());
        }
        if (test_iter == TEST_DYNAMIC_DUPLICATE_NAME_XX2) {
            desc_ptr->AddInputDesc("x1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xx1", general_ge_tensor_desc->Clone());
            desc_ptr->AddInputDesc("xxx2", general_ge_tensor_desc->Clone());
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
        if (test_iter != TEST_REQUIRED_MISS) {
            desc_ptr->AddOutputDesc("x", general_ge_tensor_desc->Clone());
        }
        // TEST_DYNAMIC_MISS
        if (test_iter == TEST_DYNAMIC_MISS) {
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

TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_success)
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

TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_required_miss)
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

TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_dynamic_miss)
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

TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_dynamic_found_2)
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

TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_optional_miss)
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

TEST_F(STEST_FE_CHECK_PARAM_TYPE, ParseBasicParameterOutputInplace)
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

/* Test CheckParamType, 5 input with closed name. x, x1, x2, x3, x4 separately
 * x1/x2/x3/x4/ are different opdesc input of input type name x.
 * Expected Result: False
 * Reason: x is required but in this case we have 5 input named after x*/
TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_success_more_than_one_input)
{
    fe::OpContent op_content_temp = {"FrameworkOp", "", {{"compute", COST_INFO},
                                  {"partial",FLAG_INFO},
                                  {"async",FLAG_INFO},
                                  {"input0", IN0_TENSOR_DESC_INFO_MAP},
                                  {"input1", IN1_TENSOR_DESC_INFO_MAP},
                                  {"input2", IN2_TENSOR_DESC_INFO_MAP},
                                  {"input3", IN3_TENSOR_DESC_INFO_MAP},
                                  {"input4", IN4_TENSOR_DESC_INFO_MAP},
                                  {"input5", IN5_TENSOR_DESC_INFO_MAP},
                                  {"input6", IN6_TENSOR_DESC_INFO_MAP},
                                  {"output0", IN0_TENSOR_DESC_INFO_MAP},
                                  {"output1", IN1_TENSOR_DESC_INFO_MAP},
                                  {"output2", IN2_TENSOR_DESC_INFO_MAP}}};

    auto op_desc_ptr = GenerateOpDesc(TEST_REQUIRED_DUPLICATE_NAME);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_temp, op_kernel_info_ptr);

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

/* Test CheckParamType, 4 input with closed name. x, y, xx1, xx2 separately,
 * xx1,xx2 would be judged as the first one of input type: xx
 * Expected result: false;
 * Reason: Because input name xx is required, which should only have one input,
 * this case have xx1,xx2 which exceeds one. */
TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_success_more_than_one_input_2)
{
    fe::OpContent op_content_temp = {"FrameworkOp", "", {{"compute", COST_INFO},
                                                   {"partial",FLAG_INFO},
                                                   {"async",FLAG_INFO},
                                                   {"input0", IN0_TENSOR_DESC_INFO_MAP},
                                                   {"input1", IN1_TENSOR_DESC_INFO_MAP},
                                                   {"input2", IN2_TENSOR_DESC_INFO_MAP},
                                                   {"input6", IN7_TENSOR_DESC_INFO_MAP},
                                                   {"output0", IN0_TENSOR_DESC_INFO_MAP},
                                                   {"output1", IN1_TENSOR_DESC_INFO_MAP},
                                                   {"output2", IN2_TENSOR_DESC_INFO_MAP}}};

    auto op_desc_ptr = GenerateOpDesc(TEST_REQUIRED_DUPLICATE_NAME_XX);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_temp, op_kernel_info_ptr);

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

/* Test CheckParamType, 4 input with closed name. x, y, xx1, xx2 separately,
 * xx1,xx2 would be judged as the first one of input type: xx
 * Expected result: true;
 * Reason: Because input name xx is dynamic, which should have one or more input,
 * this case have xx1,xx2 which is satisfied. */
TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_success_more_than_one_input_3)
{
    map<string, string> IN_TENSOR_DESC_INFO_MAP_XX_DYNAMIC = IN7_TENSOR_DESC_INFO_MAP;
    IN_TENSOR_DESC_INFO_MAP_XX_DYNAMIC[STR_PARAM_TYPE] = "dynamic";
    fe::OpContent op_content_temp = {"FrameworkOp", "", {{"compute", COST_INFO},
                                                   {"partial",FLAG_INFO},
                                                   {"async",FLAG_INFO},
                                                   {"input0", IN0_TENSOR_DESC_INFO_MAP},
                                                   {"input1", IN1_TENSOR_DESC_INFO_MAP},
                                                   {"input2", IN2_TENSOR_DESC_INFO_MAP},
                                                   {"input6", IN_TENSOR_DESC_INFO_MAP_XX_DYNAMIC},
                                                   {"output0", IN0_TENSOR_DESC_INFO_MAP},
                                                   {"output1", IN1_TENSOR_DESC_INFO_MAP},
                                                   {"output2", IN2_TENSOR_DESC_INFO_MAP}}};

    auto op_desc_ptr = GenerateOpDesc(TEST_DYNAMIC_DUPLICATE_NAME_XX);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_temp, op_kernel_info_ptr);

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

/* Test CheckParamType, 4 input with closed name. x, x1, xx1, xxx2 separately,
 * xx1,xx2 would be judged as the first one of input type: xx
 * Expected result: false;
 * Reason: xxx2 has no input type in ops_kernel_info.
 */
TEST_F(STEST_FE_CHECK_PARAM_TYPE, check_param_type_success_more_than_one_input_4)
{
    map<string, string> IN0_TENSOR_DESC_INFO_MAP_XX_DYNAMIC = IN0_TENSOR_DESC_INFO_MAP;
    IN0_TENSOR_DESC_INFO_MAP_XX_DYNAMIC[STR_PARAM_TYPE] = "dynamic";

    map<string, string> IN7_TENSOR_DESC_INFO_MAP_XX_DYNAMIC = IN7_TENSOR_DESC_INFO_MAP;
    IN7_TENSOR_DESC_INFO_MAP_XX_DYNAMIC[STR_PARAM_TYPE] = "dynamic";

    fe::OpContent op_content_temp = {"FrameworkOp", "", {{"compute", COST_INFO},
                                                   {"partial",FLAG_INFO},
                                                   {"async",FLAG_INFO},
                                                   {"input0", IN0_TENSOR_DESC_INFO_MAP_XX_DYNAMIC},
                                                   {"input1", IN1_TENSOR_DESC_INFO_MAP},
                                                   {"input2", IN2_TENSOR_DESC_INFO_MAP},
                                                   {"input6", IN7_TENSOR_DESC_INFO_MAP_XX_DYNAMIC},
                                                   {"output0", IN0_TENSOR_DESC_INFO_MAP},
                                                   {"output1", IN1_TENSOR_DESC_INFO_MAP},
                                                   {"output2", IN2_TENSOR_DESC_INFO_MAP}}};

    auto op_desc_ptr = GenerateOpDesc(TEST_DYNAMIC_DUPLICATE_NAME_XX2);
    OpKernelInfoPtr op_kernel_info_ptr = std::make_shared<OpKernelInfo>("FrameworkOp");
    OpKernelInfoConstructor op_kernel_info_constructor;
    op_kernel_info_constructor.InitializeOpKernelInfo(fe::AI_CORE_NAME, op_content_temp, op_kernel_info_ptr);

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
