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
#include <iostream>
#include <list>
#include "fe_llt_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "common/fe_utils.h"
#define protected public
#define private public
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/fe_op_info_common.h"
#include "common/math_util.h"
#include "ops_store/ops_kernel_manager.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class op_info_common_ut: public testing::Test
{
public:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
  OpKernelInfoPtr op_kernel_info_ptr;

protected:
    void SetUp()
    {
        fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
        FEOpsStoreInfo cce_builtin {
            1,
            "cce-builtin",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_general_opinfo",
            "",
            false,
            false,
            false};
        vector<FEOpsStoreInfo> store_info;
        store_info.emplace_back(cce_builtin);
        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
        OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

        std::map<std::string, std::string> options;
        fe_ops_kernel_info_store_ptr->Initialize(options);
        op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-builtin", "ApplyMomentum");
    }

    void TearDown()
    {
        Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
        config.is_init_ = false;

        map<string, string> options;
        config.Initialize(options);

    }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

namespace{

}

TEST_F(op_info_common_ut, GetInputOrOutputIndexNameMap_name_match)
{
    OpDescPtr apply_mom = std::make_shared<OpDesc>("xxx", "ApplyMomentum");
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);

    GeTensorDesc in_desc(shape);
    in_desc.SetFormat(FORMAT_NCHW);
    in_desc.SetDataType(DT_FLOAT16);

    GeTensorDesc out_desc(shape);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);

    apply_mom->AddInputDesc("accumulation", in_desc);
    apply_mom->AddOutputDesc("accumulationUpdate", out_desc);

    map<uint32_t, string> input_index_name_map;
    Status status = GetInputIndexNameMap(*(apply_mom.get()), *(op_kernel_info_ptr.get()), input_index_name_map);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(input_index_name_map.size(), 1);
    EXPECT_EQ(input_index_name_map[0], "accumulation");

    map<uint32_t, string> output_index_name_map;
    status = GetOutputIndexNameMap(*(apply_mom.get()), *(op_kernel_info_ptr.get()), output_index_name_map);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(output_index_name_map.size(), 1);
    EXPECT_EQ(output_index_name_map[0], "accumulationUpdate");
}

TEST_F(op_info_common_ut, GetInputOrOutputIndexNameMap_index_match)
{
    OpDescPtr apply_mom = std::make_shared<OpDesc>("xxx", "ApplyMomentum");
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);

    GeTensorDesc in_desc0(shape);
    in_desc0.SetFormat(FORMAT_NCHW);
    in_desc0.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc5(shape);
    in_desc5.SetFormat(FORMAT_NCHW);
    in_desc5.SetDataType(DT_FLOAT16);

    GeTensorDesc out_desc0(shape);
    out_desc0.SetFormat(FORMAT_NCHW);
    out_desc0.SetDataType(DT_FLOAT16);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);

    apply_mom->AddInputDesc("z", in_desc0);
    apply_mom->AddInputDesc("s", in_desc1);
    apply_mom->AddInputDesc("c", in_desc2);
    apply_mom->AddInputDesc("v", in_desc3);
    apply_mom->AddInputDesc("b", in_desc4);
    apply_mom->AddInputDesc("n", in_desc5);
    apply_mom->AddOutputDesc("k", out_desc0);
    apply_mom->AddOutputDesc("l", out_desc1);

    map<uint32_t, string> input_index_name_map;
    Status status = GetInputIndexNameMap(*(apply_mom.get()), *(op_kernel_info_ptr.get()), input_index_name_map);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(input_index_name_map.size(), 6);
    EXPECT_EQ(input_index_name_map[0], "variable");
    EXPECT_EQ(input_index_name_map[1], "accumulation");
    EXPECT_EQ(input_index_name_map[2], "learning_rate");
    EXPECT_EQ(input_index_name_map[3], "gradient");
    EXPECT_EQ(input_index_name_map[4], "momentum");
    EXPECT_EQ(input_index_name_map[5], "x");

    map<uint32_t, string> output_index_name_map;
    status = GetOutputIndexNameMap(*(apply_mom.get()), *(op_kernel_info_ptr.get()), output_index_name_map);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(output_index_name_map.size(), 2);
    EXPECT_EQ(output_index_name_map[0], "variableUpdate");
    EXPECT_EQ(output_index_name_map[1], "accumulationUpdate");
}

TEST_F(op_info_common_ut, GetInputOrOutputIndexNameMap_optional_match)
{
    OpDescPtr apply_mom = std::make_shared<OpDesc>("xxx", "ApplyMomentum");
    vector<int64_t> dim(4, 1);
    GeShape shape(dim);

    GeTensorDesc in_desc0(shape);
    in_desc0.SetFormat(FORMAT_NCHW);
    in_desc0.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);

    GeTensorDesc in_desc4(shape);
    in_desc4.SetFormat(FORMAT_NCHW);
    in_desc4.SetDataType(DT_FLOAT16);

    GeTensorDesc out_desc(shape);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_FLOAT16);

    apply_mom->AddInputDesc("variable", in_desc0);
    apply_mom->AddInputDesc("accumulation", in_desc1);
    apply_mom->AddInputDesc("learning_rate", in_desc2);
    apply_mom->AddInputDesc("gradient", in_desc3);
    apply_mom->AddInputDesc("qwer", in_desc4);
    apply_mom->AddOutputDesc("asdf", out_desc);

    map<uint32_t, string> input_index_name_map;
    Status status = GetInputIndexNameMap(*(apply_mom.get()), *(op_kernel_info_ptr.get()), input_index_name_map);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(input_index_name_map.size(), 5);
    EXPECT_EQ(input_index_name_map[4], "momentum");

    map<uint32_t, string> output_index_name_map;
    status = GetOutputIndexNameMap(*(apply_mom.get()), *(op_kernel_info_ptr.get()), output_index_name_map);
    EXPECT_EQ(status, fe::SUCCESS);
    EXPECT_EQ(output_index_name_map.size(), 1);
    EXPECT_EQ(output_index_name_map[0], "variableUpdate");
}

TEST_F(op_info_common_ut, CheckAddSize_t_success)
{
    size_t a = 1234;
    size_t b = 8765;
    Status status = CheckSizetAddOverFlow(a, b);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(op_info_common_ut, CheckAddSize_t_fail)
{
    size_t a = 18446744073709551000;
    size_t b = 18446744073709551111;
    Status status = CheckSizetAddOverFlow(a, b);
    EXPECT_EQ(status, fe::FAILED);
}

TEST_F(op_info_common_ut, GenerateUnionFormatDtype_format_dtype_empty)
{
    vector<ge::Format> old_formats;
    vector<ge::DataType> old_data_types;
    vector<ge::Format> new_formats;
    vector<ge::DataType> new_data_types;

    Status status = GenerateUnionFormatDtype(old_formats, old_data_types, new_formats, new_data_types);
    EXPECT_EQ(status, fe::SUCCESS);

    old_data_types.clear();
    ge::Format format = ge::FORMAT_FRACTAL_NZ;
    old_formats.push_back(format);
    status = GenerateUnionFormatDtype(old_formats, old_data_types, new_formats, new_data_types);
    EXPECT_EQ(status, fe::SUCCESS);

    old_formats.clear();
    ge::DataType data_type = ge::DT_FLOAT16;
    old_data_types.push_back(data_type);

    status = GenerateUnionFormatDtype(old_formats, old_data_types, new_formats, new_data_types);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(op_info_common_ut, test_get_impl_type_string) {
  string impl_type1 = GetImplTypeString(EN_IMPL_CUSTOM_TBE);
  string impl_type2 = GetImplTypeString(EN_RESERVED);
  EXPECT_EQ(impl_type1, "tbe-custom");
  EXPECT_EQ(impl_type2, "unknown-type 13");
}

TEST_F(op_info_common_ut, test_get_pass_type_string) {
  string pass_type1 = GetPassTypeString(BUILT_IN_GRAPH_PASS);
  string pass_type2 = GetPassTypeString(CUSTOM_AI_CORE_GRAPH_PASS);

  string pass_type3 = GetPassTypeString(GRAPH_FUSION_PASS_TYPE_RESERVED);
  EXPECT_EQ(pass_type1, "built-in-ai-core-graph-pass");
  EXPECT_EQ(pass_type2, "custom-ai-core-graph-pass");
  EXPECT_EQ(pass_type3, "unknown-pass-type 19");
}

TEST_F(op_info_common_ut, test_get_buffer_fusion_pass_type_string) {
  string pass_type1 = GetBufferFusionPassTypeString(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
  string pass_type2 = GetBufferFusionPassTypeString(BUFFER_FUSION_PASS_TYPE_RESERVED);

  EXPECT_EQ(pass_type1, "build-in-ai-core-buffer_fusion-pass");
  EXPECT_EQ(pass_type2, "unknown-buffer-fusion-pass-type 4");
}

TEST_F(op_info_common_ut, get_fusion_scope_attr_1) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  int64_t scope_id = 0;
  bool ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, false);

  ge::AttrUtils::SetInt(op_desc, L1_SCOPE_ID_ATTR, 123);
  ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, 123);
}

TEST_F(op_info_common_ut, get_fusion_scope_attr_2) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  int64_t scope_id = 0;
  bool ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, false);

  ge::AttrUtils::SetInt(op_desc, SCOPE_ID_ATTR, 234);
  ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, 234);
}

TEST_F(op_info_common_ut, get_fusion_scope_attr_3) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  int64_t scope_id = 0;
  bool ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, false);

  ge::AttrUtils::SetInt(op_desc, SCOPE_ID_ATTR, 234);
  ge::AttrUtils::SetInt(op_desc, L1_SCOPE_ID_ATTR, 345);
  ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, 345);
}

TEST_F(op_info_common_ut, get_fusion_scope_attr_4) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  int64_t scope_id = 0;
  bool is_l1_fusion = false;
  bool ret = GetFusionScopeAttr(op_desc, scope_id, is_l1_fusion);
  EXPECT_EQ(ret, false);

  ge::AttrUtils::SetInt(op_desc, SCOPE_ID_ATTR, 789);
  ge::AttrUtils::SetInt(op_desc, L1_SCOPE_ID_ATTR, 123);
  ret = GetFusionScopeAttr(op_desc, scope_id, is_l1_fusion);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, 123);
  EXPECT_EQ(is_l1_fusion, true);
}

TEST_F(op_info_common_ut, get_fusion_scope_attr_5) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  int64_t scope_id = 0;
  bool is_l1_fusion = false;
  bool ret = GetFusionScopeAttr(op_desc, scope_id, is_l1_fusion);
  EXPECT_EQ(ret, false);

  ge::AttrUtils::SetInt(op_desc, SCOPE_ID_ATTR, 234);
  ret = GetFusionScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, 234);
  EXPECT_EQ(is_l1_fusion, false);
}

TEST_F(op_info_common_ut, remove_l1fusion_scope_attr_1) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  int64_t scope_id = 123;
  ge::AttrUtils::SetInt(op_desc, L1_SCOPE_ID_ATTR, scope_id);
  RemoveL1FusionScopeAttr(op_desc);
  EXPECT_EQ(op_desc->HasAttr(L1_SCOPE_ID_ATTR), false);

  int64_t fail_scope_id = 0;
  bool has_fail_ret = ge::AttrUtils::GetInt(op_desc, FAILED_L1_SCOPE_ID_ATTR, fail_scope_id);
  EXPECT_EQ(has_fail_ret, true);
  EXPECT_EQ(fail_scope_id, scope_id);
}

TEST_F(op_info_common_ut, remove_l1fusion_scope_attr_2) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("op_desc", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  op_desc->AddInputDesc(tenosr_desc);
  op_desc->AddOutputDesc(tenosr_desc);
  RemoveL1FusionScopeAttr(op_desc);
  EXPECT_EQ(op_desc->HasAttr(L1_SCOPE_ID_ATTR), false);
  EXPECT_EQ(op_desc->HasAttr(FAILED_L1_SCOPE_ID_ATTR), false);
}

TEST_F(op_info_common_ut, get_default_reshape_type) {
  ge::Format original_format;
  size_t old_dims_size = 5; 
  std::string reshape_type;
  Status ret = GetDefaultReshapeType(original_format, old_dims_size, reshape_type);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(op_info_common_ut, copy_weight_attr_to_placeholder_test) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", "Reshape");
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetOutputOffset(std::vector<int64_t>{4});
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = compute_graph->AddNode(op_desc);
  CopyWeightAttrToPlaceHolder(node);
  EXPECT_EQ(node->GetOpDesc()->GetOutputOffset()[0], 4);
}

TEST_F(op_info_common_ut, is_thread1_node) {
  EXPECT_EQ(IsThread1Node(nullptr), false);
  ge::OpDescPtr op_desc_0 = std::make_shared<OpDesc>("relu0", "Relu");
  op_desc_0->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc_0->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));

  ge::OpDescPtr op_desc_1 = std::make_shared<OpDesc>("relu1", "Relu");
  op_desc_1->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc_1->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node_0 = compute_graph->AddNode(op_desc_0);
  ge::NodePtr node_1 = compute_graph->AddNode(op_desc_1);
  EXPECT_EQ(IsThread1Node(node_0), false);
  node_0->GetOpDesc()->SetExtAttr(kAttrThread1Node, node_1);
  EXPECT_EQ(IsThread1Node(node_0), true);
}
