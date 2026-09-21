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
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/fe_op_info_common.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class op_info_common_st: public testing::Test
{
public:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
  OpKernelInfoPtr op_kernel_info_ptr;

protected:
    void SetUp()
    {
        fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>();
        FEOpsStoreInfo cce_builtin {
            1,
            "cce-builtin",
            EN_IMPL_HW_GENERAL_CCE,
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
        string soc_version = "Ascend910B";
        config.Initialize(options);

    }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

namespace{

}

TEST_F(op_info_common_st, GetInputOrOutputIndexNameMap_name_match)
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

TEST_F(op_info_common_st, GetInputOrOutputIndexNameMap_index_match)
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

TEST_F(op_info_common_st, GetInputOrOutputIndexNameMap_optional_match)
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

TEST_F(op_info_common_st, GenerateUnionFormatDtype_format_dtype_empty)
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

TEST_F(op_info_common_st, get_default_reshape_type) {
  ge::Format original_format;
  size_t old_dims_size = 5; 
  std::string reshape_type;
  Status ret = GetDefaultReshapeType(original_format, old_dims_size, reshape_type);
  EXPECT_EQ(ret, ge::FAILED);
}

// TEST_F(op_info_common_st, expand_by_reshape_type){
//   std::vector<int64_t> dims;
//   std::string op_name;
//   ge::Format original_format = ge::FORMAT_NCHW;
//   size_t full_size = 1;
//   uint32_t tensor_index;
//   std::string reshape_type = "NCHW";
//   fe::ExpandByReshapeType(dims, op_name, original_format, full_size, tensor_index, reshape_type);
// }

TEST_F(op_info_common_st, copy_weight_attr_to_placeholder_test) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", "Reshape");
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetOutputOffset(std::vector<int64_t>{4});
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = compute_graph->AddNode(op_desc);
  EXPECT_EQ(compute_graph->GetDirectNode().size(), 1);
  fe::CopyWeightAttrToPlaceHolder(node);
}