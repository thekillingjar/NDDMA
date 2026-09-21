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

#include "graph/ge_attr_value.h"
#include "fe_llt_utils.h"
#define protected public
#define private   public
#include "ops_kernel_store/sub_ops_store.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_store/op_kernel_info.h"

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
using fe::InputOrOutputInfoPtr ;
using ge::GeAttrValue;
using std::vector;
using std::map;

static const string ATTR_NAME_INT = "attr_name_int";
static const string ATTR_NAME_FLOAT = "attr_name_float";
static const string ATTR_NAME_STR = "attr_name_str";
static const string ATTR_NAME_BOOL = "attr_name_bool";
static const string ATTR_NAME_LIST_INT = "attr_name_list_int";
static const string ATTR_NAME_LIST_FLOAT = "attr_name_list_float";
static const string ATTR_NAME_LIST_STR = "attr_name_list_str";
static const string ATTR_NAME_LIST_BOOL = "attr_name_list_bool";
static const string ATTR_NAME_DEFAULT = "attr_name_default";

enum TestIter {
    TEST_SUCCESS = 0,
    TEST_HAVE_ALL,        // have one "all" type for attr check
    TEST_ATTR_NOT_FOUND,  // can not found attr ATTR_NAME_STR in OpDesc
    TEST_NOT_SUPPORT_DATA_TYPE,  // exit not support ValueType
    TEST_CHECK_FAILED,    // have one not match iter (ATTR_NAME_FLOAT)
};

class STEST_FE_CHECK_ATTR_SUPPORT : public testing::Test {
protected:
    /* 0 : int
     * 1 : float
     * 2 : string
     * 3 : bool
     * 4 : list_int
     * 5 : list_float
     * 6 : list_string
     * 7 : list_bool
     */
    void SetUp()
    {
        test_subject_ptr_ = std::make_shared<SubOpsStore>(AI_CORE_NAME);
        test_subject_ptr_->format_dtype_querier_ptr_ =
            std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);
        const int64_t size = 3;
        std::vector<AttrInfoPtr> empty_attrs_info;
        test_attrs_info_.swap(empty_attrs_info);
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_INT));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_FLOAT));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_STR));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_BOOL));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_LIST_INT));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_LIST_FLOAT));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_LIST_STR));
        test_attrs_info_.emplace_back(std::make_shared<AttrInfo>(ATTR_NAME_LIST_BOOL));


        for (int64_t i = 0; i < size; i++) {
            int64_t list_a = 0 + size * i;
            int64_t list_b = 1 + size * i;
            int64_t list_c = 2 + size * i;
            test_attrs_info_[0]->supported_values_.emplace_back(GeAttrValue::CreateFrom<int64_t>(i));
            test_attrs_info_[1]->supported_values_.emplace_back (GeAttrValue::CreateFrom<float>((float)(i + 0.1)));
            test_attrs_info_[2]->supported_values_.emplace_back(GeAttrValue::CreateFrom<string>(std::to_string(i)));
            test_attrs_info_[2]->is_required_ = true;
            test_attrs_info_[3]->supported_values_.emplace_back(GeAttrValue::CreateFrom<bool>((i % 2) == 1));

            vector<int64_t> tmp_int_vec;
            tmp_int_vec.emplace_back(list_a);
            tmp_int_vec.emplace_back(list_b);
            tmp_int_vec.emplace_back(list_c);
            GeAttrValue tmp_list_int = GeAttrValue::CreateFrom<vector<int64_t>>(tmp_int_vec);
            test_attrs_info_[4]->supported_values_.emplace_back(tmp_list_int);

            vector<float> tmp_float_vec({(float)(list_a + 0.1), (float)(list_b + 0.1), (float)(list_c + 0.1)});
            test_attrs_info_[5]->supported_values_.emplace_back(GeAttrValue::CreateFrom<vector<float>>(tmp_float_vec));

            vector<string> tmp_str_vec({std::to_string(list_a), std::to_string(list_b), std::to_string(list_c)});
            test_attrs_info_[6]->supported_values_.emplace_back(GeAttrValue::CreateFrom<vector<string>>(tmp_str_vec));

            vector<bool> tmp_bool_vec({(list_a % 2) == 1, (list_b % 2) == 1, (list_c % 2) == 1});
            test_attrs_info_[7]->supported_values_.emplace_back(GeAttrValue::CreateFrom<vector<bool>>(tmp_bool_vec));
        }
    }

    void TearDowm()
    {
        test_subject_ptr_->FinalizeSubStore();
        test_attrs_info_.clear();
        test_attr_value_.clear();
    }

    OpDescPtr CreateOpDescPtr(TestIter test_iter)
    {
        OpDescPtr desc_ptr = std::make_shared<OpDesc>("test_op_desc", "FrameworkOP");
        AttrUtils::SetInt(desc_ptr, ATTR_NAME_INT, 1);
        if (test_iter == TEST_CHECK_FAILED) {
            AttrUtils::SetFloat(desc_ptr, ATTR_NAME_FLOAT, 3.1415);
        } else {
            AttrUtils::SetFloat(desc_ptr, ATTR_NAME_FLOAT, 1.1);
        }
        if (test_iter != TEST_ATTR_NOT_FOUND) {
            AttrUtils::SetStr(desc_ptr, ATTR_NAME_STR, "1");
        }
        AttrUtils::SetBool(desc_ptr, ATTR_NAME_BOOL, true);
        if (test_iter == TEST_HAVE_ALL) {
            AttrUtils::SetListInt(desc_ptr, ATTR_NAME_LIST_INT, { 100, 101, 103});
        } else {
            AttrUtils::SetListInt(desc_ptr, ATTR_NAME_LIST_INT, { 0, 1, 2 });
        }

        AttrUtils::SetListFloat(desc_ptr, ATTR_NAME_LIST_FLOAT, { 0.1, 1.1, 2.1 });
        AttrUtils::SetListStr(desc_ptr, ATTR_NAME_LIST_STR, { "0", "1", "2" });
        AttrUtils::SetListBool(desc_ptr, ATTR_NAME_LIST_BOOL, { true, false, true });

        return desc_ptr;
    }

    void GenerateOpKernelInfo(TestIter test_iter)
    {
        map<string,string> options;
        FEOpsStoreInfo cce_custom {
        1,
        "cce_custom_opinfo",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_custom_opinfo",
        ""};

        test_subject_ptr_->SetSubStoreInfo(cce_custom);
        test_subject_ptr_->InitializeSubStore();

        /* 0 : int
         * 1 : float
         * 2 : string
         * 3 : bool
         * 4 : list_int
         * 5 : list_float
         * 6 : list_string
         * 7 : list_bool
         */
        std::vector<GeAttrValue::ValueType> attr_value_type_array = {
        GeAttrValue::VT_INT,    GeAttrValue::VT_FLOAT,      GeAttrValue::VT_STRING,      GeAttrValue::VT_BOOL,
        GeAttrValue::VT_LIST_INT, GeAttrValue::VT_LIST_FLOAT, GeAttrValue::VT_LIST_STRING, GeAttrValue::VT_LIST_BOOL
        };

        for(uint32_t i = 0; i < test_attrs_info_.size(); i++) {
            test_attrs_info_[i]->dtype_ = attr_value_type_array[i];
        }

        if(test_iter == TEST_NOT_SUPPORT_DATA_TYPE) {
            test_attrs_info_[0]->dtype_ = GeAttrValue::VT_BYTES;
        }
        if (test_iter == TEST_HAVE_ALL) {
            test_attrs_info_[4]->is_support_all_value_ = true;
        }
    }

    std::vector<AttrInfoPtr> test_attrs_info_;
    std::vector<vector<GeAttrValue>> test_attr_value_;
    SubOpsStorePtr test_subject_ptr_;
};

TEST_F(STEST_FE_CHECK_ATTR_SUPPORT, check_attr_support_success)
{
    TestIter test_iter = TEST_SUCCESS;

    OpDescPtr test_op_desc_ptr = CreateOpDescPtr(test_iter);
    GenerateOpKernelInfo(test_iter);

    OpKernelInfo op_kernel_info("FrameworkOP");
    op_kernel_info.attrs_info_ = test_attrs_info_;
    cout<<"1"<<endl;
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(test_op_desc_ptr);
    bool result = test_subject_ptr_->CheckAttrSupport(test_node, op_kernel_info, reason);

    EXPECT_EQ(true, result);
}

TEST_F(STEST_FE_CHECK_ATTR_SUPPORT, check_attr_support_have_all_flag)
{
    TestIter test_iter = TEST_HAVE_ALL;

    OpDescPtr test_op_desc_ptr = CreateOpDescPtr(test_iter);
    GenerateOpKernelInfo(test_iter);

    OpKernelInfo op_kernel_info("FrameworkOP");
    op_kernel_info.attrs_info_ = test_attrs_info_;
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(test_op_desc_ptr);
    bool result = test_subject_ptr_->CheckAttrSupport(test_node, op_kernel_info, reason);

    EXPECT_EQ(true, result);
}

TEST_F(STEST_FE_CHECK_ATTR_SUPPORT, check_attr_support_failed)
{
    TestIter test_iter = TEST_CHECK_FAILED;

    OpDescPtr test_op_desc_ptr = CreateOpDescPtr(test_iter);
    GenerateOpKernelInfo(test_iter);

    OpKernelInfo op_kernel_info("FrameworkOP");
    op_kernel_info.attrs_info_ = test_attrs_info_;
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(test_op_desc_ptr);
    bool result = test_subject_ptr_->CheckAttrSupport(test_node, op_kernel_info, reason);

    EXPECT_EQ(false, result);
}

TEST_F(STEST_FE_CHECK_ATTR_SUPPORT, check_attr_support_attr_not_found)
{
    TestIter test_iter = TEST_ATTR_NOT_FOUND;

    OpDescPtr test_op_desc_ptr = CreateOpDescPtr(test_iter);
    GenerateOpKernelInfo(test_iter);

    OpKernelInfo op_kernel_info("FrameworkOP");
    op_kernel_info.attrs_info_ = test_attrs_info_;
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(test_op_desc_ptr);
    bool result = test_subject_ptr_->CheckAttrSupport(test_node, op_kernel_info, reason);

    EXPECT_EQ(false, result);
}

TEST_F(STEST_FE_CHECK_ATTR_SUPPORT, check_attr_support_attr_not_supported_data_type)
{
    TestIter test_iter = TEST_NOT_SUPPORT_DATA_TYPE;

    OpDescPtr test_op_desc_ptr = CreateOpDescPtr(test_iter);
    GenerateOpKernelInfo(test_iter);

    OpKernelInfo op_kernel_info("FrameworkOP");
    op_kernel_info.attrs_info_ = test_attrs_info_;
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(test_op_desc_ptr);
    bool result = test_subject_ptr_->CheckAttrSupport(test_node, op_kernel_info, reason);

    EXPECT_EQ(false, result);
}
