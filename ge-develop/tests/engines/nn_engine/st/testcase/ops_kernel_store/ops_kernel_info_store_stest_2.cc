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
#include "graph/ge_tensor.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"

#include "ops_kernel_store/sub_ops_store.h"
#include "ops_store/op_kernel_info.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "common/configuration.h"

#include "graph/ge_tensor.h"
#include "graph/types.h"
#include "register/op_registry.h"
using namespace std;
using namespace testing;
using namespace fe;
using namespace ge;
using fe::SubOpsStore;
using ge::AttrUtils;
using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;
using SubOpsStorePtr = std::shared_ptr<SubOpsStore>;

FEOpsStoreInfo cce_custom_opinfo {
      0,
      "cce-custom",
      EN_IMPL_CUSTOM_CONSTANT_CCE,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_custom_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo tik_custom_opinfo {
      1,
      "tik-custom",
      EN_IMPL_CUSTOM_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tik_custom_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo tbe_custom_opinfo {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_custom_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo cce_constant_opinfo {
      3,
      "cce-constant",
      EN_IMPL_HW_CONSTANT_CCE,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_constant_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo cce_general_opinfo {
      4,
      "cce-general",
      EN_IMPL_HW_GENERAL_CCE,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_general_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo tik_opinfo {
      5,
      "tik-builtin",
      EN_IMPL_HW_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tik_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo tbe_opinfo {
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
};
FEOpsStoreInfo rl_opinfo {
      7,
      "rl-builtin",
      EN_IMPL_RL,
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/rl_opinfo",
      "",
      false,
      false,
      false
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

static const std::vector<std::string> vec_type = {
    "add",
    "del",
    "mul",
    "dev"};
                
static const std::unordered_map<std::string, std::vector<std::vector<ge::Format>>> op_input_supported_formats = {
    {"add", {{FORMAT_NCHW, FORMAT_NHWC}, {FORMAT_NCHW, FORMAT_NHWC}}},
    {"del", {{FORMAT_ND, FORMAT_NC1HWC0}, {FORMAT_FRACTAL_Z, FORMAT_NC1C0HWPAD}}},
    {"mul", {{FORMAT_NHWC1C0}, {FORMAT_NHWC1C0}}},
    {"dev", {{FORMAT_FSR_NCHW}, {FORMAT_FRACTAL_DECONV}}}
};

static const std::unordered_map<std::string, std::vector<std::vector<ge::Format>>> op_output_supported_formats = {
    {"add", {{FORMAT_NCHW}}},
    {"del", {{FORMAT_ND, FORMAT_NC1HWC0}}},
    {"mul", {{FORMAT_NHWC1C0}}},
    {"dev", {{FORMAT_FSR_NCHW, FORMAT_FRACTAL_DECONV}}}
};

static const std::unordered_map<std::string, std::vector<std::vector<ge::DataType>>> op_input_supported_data_types = {
    {"add", {{DT_FLOAT, DT_FLOAT16}, {DT_FLOAT, DT_FLOAT16}}},
    {"del", {{DT_INT8, DT_INT16}, {DT_UINT16, DT_UINT8}}},
    {"mul", {{DT_INT32}, {DT_INT32}}},
    {"dev", {{DT_INT64}, {DT_UINT32}}}, 
};

static const std::unordered_map<std::string, std::vector<std::vector<ge::DataType>>> op_output_supported_data_types = {
    {"add", {{DT_FLOAT}}},
    {"del", {{DT_INT8, DT_INT16}}},
    {"mul", {{DT_INT32}}},
    {"dev", {{DT_INT64, DT_UINT32}}}, 
};

std::unordered_map<std::string, std::vector<std::vector<ge::TensorDescInfo>>> op_input_limited_tensor_descs;

std::unordered_map<std::string, std::vector<std::vector<ge::TensorDescInfo>>> op_output_limited_tensor_descs;

class STEST_OPS_KERNEL_INFO_STORE_2 : public testing::Test{
  protected:
    static void SetUpTestCase() {
        std::cout << "STEST_OPS_KERNEL_INFO_STORE_2 SetUP" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "STEST_OPS_KERNEL_INFO_STORE_2 TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp(){
        map<string, string> options;
        fe_ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>();
        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info);
        OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

        Status result = fe_ops_kernel_info_store_ptr->Initialize(options);
        cout << fe_ops_kernel_info_store_ptr->map_all_sub_store_info_.size() << endl;
        TensorDescInfo tensor_i1;
        tensor_i1.format_ = FORMAT_ND;
        tensor_i1.dataType_ = DT_INT16;

        TensorDescInfo tensor_i2;
        tensor_i2.format_ = FORMAT_NHWC;
        tensor_i2.dataType_ = DT_UINT32;

        TensorDescInfo tensor_i3;
        tensor_i2.format_ = FORMAT_ND;
        tensor_i2.dataType_ = DT_INT8;

        TensorDescInfo tensor_i4;
        tensor_i2.format_ = FORMAT_NHWC1C0;
        tensor_i2.dataType_ = DT_INT16;

        TensorDescInfo tensor_i5;
        tensor_i2.format_ = FORMAT_ND;
        tensor_i2.dataType_ = DT_INT8;

        TensorDescInfo tensor_i6;
        tensor_i2.format_ = FORMAT_NHWC;
        tensor_i2.dataType_ = DT_UINT32;

        TensorDescInfo tensor_o1;
        tensor_i2.format_ = FORMAT_ND;
        tensor_i2.dataType_ = DT_INT16;

        TensorDescInfo tensor_o2;
        tensor_i2.format_ = FORMAT_NHWC1C0;
        tensor_i2.dataType_ = DT_UINT32;

        std::vector<ge::TensorDescInfo> vec_i1 = {tensor_i1, tensor_i2};
        std::vector<ge::TensorDescInfo> vec_i2 = {tensor_i3};
        std::vector<ge::TensorDescInfo> vec_i3 = {tensor_i4};
        std::vector<ge::TensorDescInfo> vec_i4 = {tensor_i5, tensor_i6};
        std::vector<ge::TensorDescInfo> vec_o1 = {tensor_o1};
        std::vector<ge::TensorDescInfo> vec_o2 = {tensor_o2};
        std::unordered_map<std::string, std::vector<std::vector<ge::TensorDescInfo>>> op_input_limited_tensor_descs_ {
            {"del", {vec_i1, vec_i2}},
            {"mul", {vec_i3, vec_i4}}
        };

        std::unordered_map<std::string, std::vector<std::vector<ge::TensorDescInfo>>> op_output_limited_tensor_descs_ {
            {"del", {vec_o1}},
            {"mul", {vec_o2}}
        };

        op_input_limited_tensor_descs = op_input_limited_tensor_descs_;
        op_output_limited_tensor_descs = op_output_limited_tensor_descs_;

        std::cout << "A ops kernel info store test is setting up; Result = " << result << std::endl;
    }
    virtual void TearDown(){
        Status result = fe_ops_kernel_info_store_ptr->Finalize();
        std::cout << "A ops kernel info store test is tearing down; Result = " << result << std::endl;

    }
public:
    FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr;
};

static void GetOpTypeByImplyTypeStub(domi::OpRegistry *This, std::vector<std::string>& vec_op_type, const domi::ImplyType& imply_type) {
    vec_op_type = vec_type;
    return;
}


static void GetSupportedInputFormatsStub(domi::OpRegistry *This, const std::string& op_type, std::vector<std::vector<ge::Format>>& suported_input_formats) {
    auto find_ret = op_input_supported_formats.find(op_type);
    if (find_ret == op_input_supported_formats.end()) {
        suported_input_formats.clear();
    } else {
        suported_input_formats = find_ret->second;
    }
    return;
}

static void GetSupportedOutputFormatsStub(domi::OpRegistry *This, const std::string& op_type, std::vector<std::vector<ge::Format>>& supported_output_formats)
{
    auto find_ret = op_output_supported_formats.find(op_type);
    if (find_ret == op_output_supported_formats.end()) {
        supported_output_formats.clear();
    } else {
        supported_output_formats = find_ret->second;
    }
}

static void GetSupportedInputTypesStub(domi::OpRegistry *This, const std::string& op_type, std::vector<std::vector<ge::DataType>>& suported_input_data_types)
{
    auto find_ret = op_input_supported_data_types.find(op_type);
    if (find_ret == op_input_supported_data_types.end()) {
        suported_input_data_types.clear();
    } else {
        suported_input_data_types = find_ret->second;
    }
}

static void GetSupportedOutputTypesStub(domi::OpRegistry *This, const std::string& op_type, std::vector<std::vector<ge::DataType>>& supported_output_data_types)
{
    auto find_ret = op_output_supported_data_types.find(op_type);
    if (find_ret == op_output_supported_data_types.end()) {
        supported_output_data_types.clear();
    } else {
        supported_output_data_types = find_ret->second;
    }
}

static void GetLimitedInputTensorDescsStub(domi::OpRegistry *This, const std::string& op_type, std::vector<std::vector<ge::TensorDescInfo>>& input_limited_tensor_descs)
{
    auto find_ret = op_input_limited_tensor_descs.find(op_type);
    if (find_ret == op_input_limited_tensor_descs.end()) {
        input_limited_tensor_descs.clear();
    } else {
        input_limited_tensor_descs = find_ret->second;
    }
}

static void GetLimitedOutputTensorDescsStub(domi::OpRegistry *This, const std::string& op_type, std::vector<std::vector<ge::TensorDescInfo>>& output_limited_tensor_descs)
{
    auto find_ret = op_output_limited_tensor_descs.find(op_type);
    if (find_ret == op_output_limited_tensor_descs.end()) {
        output_limited_tensor_descs.clear();
    } else {
        output_limited_tensor_descs = find_ret->second;
    }
}

TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, init_from_jsonfile_succ){
    map<string, string> options;
    shared_ptr<FEOpsKernelInfoStore> ops_store = make_shared<FEOpsKernelInfoStore>();
    Status ret = ops_store->Initialize(options);
    EXPECT_EQ(8, ops_store->map_all_sub_store_info_.size());
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, get_ops_kernel_info_stores){

    std::map<std::string, SubOpsStorePtr > all_sub_stores;
    Status ret = fe_ops_kernel_info_store_ptr->GetAllSubOpsStore(all_sub_stores);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, query_op_highprio_fail){

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

TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, query_op_highp_rio_succ){

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

// attr_list_float not supported by all the OpsKernelInfoStore
TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, query_op_highp_rio_fail){

    OpImplType op_impl_type = EN_RESERVED;
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
    cout <<"the op_impl_type of conv is "<< (int)op_impl_type << endl;
    EXPECT_EQ(fe::OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL, ret);
}

TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, multiple_op_merge_fusion_graph){
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

// attr_list_float not supported by all the OpsKernelInfoStore
TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, set_atomic_op_attr){
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
    std::vector<int64_t> params = {1, 0, 1, 0, 0, 1, 0, 1};
    ge::AttrUtils::SetListInt(op_desc_ptr, "tbe_op_atomic_workspace_index", params);
    bool atomic_node_flag = false;
    Status ret = fe_ops_kernel_info_store_ptr->SetAtomicOpAttr(op_desc_ptr, atomic_node_flag);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OPS_KERNEL_INFO_STORE_2, test_op_info_mgr_finalize)
{
    Status ret = fe_ops_kernel_info_store_ptr->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}
