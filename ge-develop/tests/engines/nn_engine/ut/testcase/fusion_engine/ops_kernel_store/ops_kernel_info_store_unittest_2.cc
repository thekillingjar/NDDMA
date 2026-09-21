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
#include <memory>
#include <string>
#include <map>
#include <utility>
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "common/configuration.h"
#include "graph/ge_tensor.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"

#include "ops_store/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace fe;
using ge::AttrUtils;


FEOpsStoreInfo cce_custom_opinfo {
      0,
      "cce-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_custom_opinfo",
      ""
};
FEOpsStoreInfo tik_custom_opinfo {
      1,
      "tik-custom",
      EN_IMPL_CUSTOM_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_custom_opinfo",
      ""
};
FEOpsStoreInfo tbe_custom_opinfo {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""
};
FEOpsStoreInfo cce_constant_opinfo {
      3,
      "cce-constant",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_constant_opinfo",
      ""
};
FEOpsStoreInfo cce_general_opinfo {
      4,
      "cce-general",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_general_opinfo",
      ""
};
FEOpsStoreInfo tik_opinfo {
      5,
      "tik-builtin",
      EN_IMPL_HW_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_opinfo",
      ""
};
FEOpsStoreInfo tbe_opinfo {
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      ""
};
FEOpsStoreInfo tbe_opinfo_wrong_precision_reduce_flag {
    6,
    "tbe-builtin",
    EN_IMPL_HW_TBE,
    GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo_wrong_precision_reduce_flag",
    ""
};
FEOpsStoreInfo rl_opinfo {
      7,
      "rl-builtin",
      EN_IMPL_RL,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/rl_opinfo",
      ""
};

FEOpsStoreInfo tbe_opinfo_wrong_path {
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo_wrong_path",
      ""
};

FEOpsStoreInfo tbe_opinfo_wrong_1 {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo_wrong_1",
        ""
};

FEOpsStoreInfo tbe_opinfo_wrong_2 {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo_wrong_2",
        ""
};

FEOpsStoreInfo tbe_opinfo_wrong_3 {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo_wrong_3",
        ""
};

FEOpsStoreInfo rl_opinfo_wrong_path {
      7,
      "rl-builtin",
      EN_IMPL_RL,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/rl_opinfo_wrong_path",
      ""
};
std::vector<FEOpsStoreInfo> all_fe_ops_store_info{
      cce_custom_opinfo,
      tik_custom_opinfo,
      tbe_custom_opinfo,
      cce_constant_opinfo,
      cce_general_opinfo,
      tik_opinfo,
      tbe_opinfo,
      rl_opinfo
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_wrong_path{
      cce_custom_opinfo,
      tik_custom_opinfo,
      tbe_custom_opinfo,
      cce_constant_opinfo,
      cce_general_opinfo,
      tik_opinfo,
      tbe_opinfo_wrong_path,
      rl_opinfo_wrong_path
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_less_than8{
    cce_custom_opinfo,
    tik_custom_opinfo,
    tbe_custom_opinfo,
    cce_constant_opinfo,
    cce_general_opinfo,
    tik_opinfo,
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_wrong_json1{
        tbe_opinfo_wrong_1
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_wrong_json2{
        tbe_opinfo_wrong_2
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_wrong_json3{
        tbe_opinfo_wrong_3
};

std::vector<FEOpsStoreInfo> all_fe_ops_store_info_wrong_precision_reduce_flag{
    tbe_opinfo_wrong_precision_reduce_flag
};



class FEOpsKernelInfoStoreTest_2 : public testing::Test{
  protected:
    static void SetUpTestCase() {
        std::cout << "OpsKernelInfoStoreTest SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "OpsKernelInfoStoreTest TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp(){
        map<string, string> options;
        fe_ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info);
        fe_ops_kernel_info_store_ptr->Initialize(options);
        fe_ops_kernel_info_store_ptr->GetAllSubOpsStore(sub_ops_stores);
        for (auto& el : sub_ops_stores){
            el.second->InitializeSubStore();
        }
        cout << fe_ops_kernel_info_store_ptr->map_all_sub_store_info_.size() << endl;
        std::cout << "One Test SetUP" << std::endl;
    }
    virtual void TearDown(){
        for (auto& el : sub_ops_stores){
            el.second->FinalizeSubStore();
        }
        std::cout << "a test SetUP" << std::endl;
        Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
        config.is_init_ = false;
        config.content_map_.clear();

        map<string, string> options;
        config.Initialize(options);

    }
    void print_vec(std::vector<FEOpsStoreInfo> StoreVec)
    {
        auto it = StoreVec.begin();
        while(it!=StoreVec.end()) {
            cout << "store name: "<< (*it).fe_ops_store_name <<" --- priority = "<<(*it).priority << endl;
            it++;
        }
    }
public:
    shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
    std::map<std::string, SubOpsStorePtr> sub_ops_stores;
};


TEST_F(FEOpsKernelInfoStoreTest_2, init_from_jsonfile_succ){
    map<string, string> options;
    shared_ptr<FEOpsKernelInfoStore> ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    Status ret = ops_kernel_info_store_ptr->Initialize(options);
    EXPECT_EQ(8, ops_kernel_info_store_ptr->map_all_sub_store_info_.size());
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, init_from_jsonfile_less_than_8_store){
    map<string, string> options;
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_less_than8);
    shared_ptr<FEOpsKernelInfoStore> ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    Status ret = ops_kernel_info_store_ptr->Initialize(options);
    EXPECT_EQ(6, ops_kernel_info_store_ptr->map_all_sub_store_info_.size());
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, init_from_wrong_jsonfile_1){
    map<string, string> options;
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_wrong_json1);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    Status ret = OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, init_from_wrong_jsonfile_2){
    map<string, string> options;
    Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_wrong_json2);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    Status ret = OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, init_from_wrong_jsonfile_3){
    map<string, string> options;
    Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_wrong_json3);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    Status ret = OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, init_from_wrong_precision_reduce_flag){
    map<string, string> options;
    Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_wrong_precision_reduce_flag);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    Status ret = OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, get_sub_ops_stores){

    std::map<std::string, SubOpsStorePtr> sub_ops_stores;
    Status ret = fe_ops_kernel_info_store_ptr->GetAllSubOpsStore(sub_ops_stores);
    EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(FEOpsKernelInfoStoreTest_2, query_op_highprio_fail){

    OpImplType op_impl_type;
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv_not_exit");
    ge::DataType set_dtype = ge::DT_FLOAT16;
    ge::Format set_format = ge::FORMAT_ND;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetFormat(set_format);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc(input0_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
    output_desc_ptr->SetDataType(set_dtype);
    output_desc_ptr->SetFormat(set_format);
    output_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddOutputDesc(output_desc_ptr->Clone());

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    OpKernelInfoPtr op_kernel_ptr;
    Status ret = fe_ops_kernel_info_store_ptr->QueryHighPrioOpImplType(test_node, op_impl_type, op_kernel_ptr);
    EXPECT_EQ(fe::OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, query_op_highp_rio_succ){

    OpImplType op_impl_type;
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv");

    int64_t int_value = 1;
    float float_value = 2.0;
    bool bool_value = false;
    string str_value = "abc";
    vector<int64_t> int_vec{1, 2, 3};
    vector<int64_t> rint_vec;
    vector<float> float_vec{4.0, 5.0, 6.0};
    vector<float> rfloat_vec;
    vector<bool> bool_vec{false, true, true};
    vector<bool> rbool_vec;
    std::vector<string> str_vec{"a", "b", "c"};
    AttrUtils::SetInt(op_desc_ptr, "transposX", int_value);
    AttrUtils::SetFloat(op_desc_ptr, "transposY", float_value);
    AttrUtils::SetBool(op_desc_ptr,"attrBool", bool_value);
    AttrUtils::SetStr(op_desc_ptr,"attrStr", str_value);
    AttrUtils::SetListInt(op_desc_ptr, "attrListInt", int_vec);
    AttrUtils::SetListFloat(op_desc_ptr, "attrListFloat", float_vec);
    AttrUtils::SetListBool(op_desc_ptr, "attrListBool", bool_vec);
    AttrUtils::SetListStr(op_desc_ptr, "attrListStr", str_vec);

    ge::DataType set_dtype = ge::DT_FLOAT16;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    input0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    input0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetShape(shape_desc);
    input1_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    input1_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
    output_desc_ptr->SetDataType(set_dtype);
    output_desc_ptr->SetShape(shape_desc);
    output_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    output_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr->AddOutputDesc("z", output_desc_ptr->Clone());

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    OpKernelInfoPtr op_kernel_ptr;
    Status ret = fe_ops_kernel_info_store_ptr->QueryHighPrioOpImplType(test_node, op_impl_type, op_kernel_ptr);
    // tbe_custom_opinfo
    EXPECT_EQ(2, (int)op_impl_type);
    EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(FEOpsKernelInfoStoreTest_2, update_op_priority_from_cfg){

    std::vector<FEOpsStoreInfo> fe_ops_store_info_vec = Configuration::Instance(fe::AI_CORE_NAME).GetOpsStoreInfo();
    print_vec(fe_ops_store_info_vec);
    int32_t tbe_custom_prio = 0;
    int32_t cce_custm_prio = 0;
    for(auto store_info : fe_ops_store_info_vec) {
      if(store_info.fe_ops_store_name == "tbe-custom") {
          tbe_custom_prio = store_info.priority;
      } else if (store_info.fe_ops_store_name == "cce-custom") {
          cce_custm_prio = store_info.priority;
      }
    }

    EXPECT_EQ(2, tbe_custom_prio);
    EXPECT_EQ(0, cce_custm_prio);
}

// attr_list_float not supported by all the OpsKernelInfoStore
TEST_F(FEOpsKernelInfoStoreTest_2, query_op_highp_rio_fail){

    OpImplType op_impl_type;
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv");

    int64_t int_value = 1;
    float float_value = 2.0;
    bool bool_value = false;
    string str_value = "abc";
    vector<int64_t> int_vec{1, 2, 3};
    vector<int64_t> rint_vec;
    // list_float value not supported
    vector<float> float_vec{44.0, 55.0, 88.0};
    vector<float> rfloat_vec;
    vector<bool> bool_vec{false, true, true};
    std::vector<string> str_vec{"a", "b", "c"};
    vector<bool> rbool_vec;
    AttrUtils::SetInt(op_desc_ptr, "transposX", int_value);
    AttrUtils::SetFloat(op_desc_ptr, "transposY", float_value);
    AttrUtils::SetBool(op_desc_ptr,"attrBool", bool_value);
    AttrUtils::SetStr(op_desc_ptr,"attrStr", str_value);
    AttrUtils::SetListInt(op_desc_ptr, "attrListInt", int_vec);
    AttrUtils::SetListFloat(op_desc_ptr, "attrListFloat", float_vec);
    AttrUtils::SetListBool(op_desc_ptr, "attrListBool", bool_vec);
    AttrUtils::SetListStr(op_desc_ptr, "attrListStr", str_vec);

    ge::DataType set_dtype = ge::DT_FLOAT16;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

    shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
    output_desc_ptr->SetDataType(set_dtype);
    output_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddOutputDesc("z", output_desc_ptr->Clone());
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    OpKernelInfoPtr op_kernel_ptr;
    Status ret = fe_ops_kernel_info_store_ptr->QueryHighPrioOpImplType(test_node, op_impl_type, op_kernel_ptr);
    EXPECT_EQ(fe::OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, test_op_info_mgr_finalize)
{
    Status ret = fe_ops_kernel_info_store_ptr->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, init_from_jsonfile_not_exist_succ){
    map<string, string> options;
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_= (all_fe_ops_store_info);
    Status ret = fe_ops_kernel_info_store_ptr->Initialize(options);
    EXPECT_EQ(8, fe_ops_kernel_info_store_ptr->map_all_sub_store_info_.size());
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, multiple_op_merge_fusion_graph){
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv_not_exit");
    ge::DataType set_dtype = ge::DT_FLOAT16;
    ge::Format set_format = ge::FORMAT_ND;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetFormat(set_format);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc(input0_desc_ptr->Clone());
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
    vector<ge::NodePtr> node_vec;
    node_vec.push_back(test_node);
    Status ret = fe_ops_kernel_info_store_ptr->MultipleOpMergeFusionGraph(node_vec, node_vec, false);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FEOpsKernelInfoStoreTest_2, get_all_atomic_clean_node_suc){
    ge::OpDescPtr conv_op = std::make_shared<ge::OpDesc>("Conv", "conv");
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node_ptr = graph->AddNode(conv_op);
    vector<ge::NodePtr> atomic_node_vec;
    Status ret = fe_ops_kernel_info_store_ptr->GetAllAtomicCleanNode(node_ptr, atomic_node_vec);
    EXPECT_EQ(fe::SUCCESS, ret);
}

// attr_list_float not supported by all the OpsKernelInfoStore
TEST_F(FEOpsKernelInfoStoreTest_2, set_atomic_op_attr){
    shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv_not_exit");
    ge::DataType set_dtype = ge::DT_FLOAT16;
    ge::Format set_format = ge::FORMAT_ND;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);

    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetFormat(set_format);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc(input0_desc_ptr->Clone());

    std::vector<uint32_t> tmp_output_index = {0, 1};
    ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);
    std::vector<int64_t> params = {1, 0, 1, 0, 0, 1, 0, 1};
    ge::AttrUtils::SetListInt(op_desc_ptr, "tbe_op_atomic_workspace_index", params);
    bool atomic_node_flag = false;
    Status ret = fe_ops_kernel_info_store_ptr->SetAtomicOpAttr(op_desc_ptr, atomic_node_flag);
    EXPECT_EQ(fe::SUCCESS, ret);
}
