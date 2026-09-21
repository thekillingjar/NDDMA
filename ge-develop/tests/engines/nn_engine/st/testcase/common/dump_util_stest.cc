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
#include <string>
#include <vector>

#define protected public
#define private public
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph/utils/graph_utils.h"
#include "graph/op_kernel_bin.h"
#include "common/graph_comm.h"
#include "common/dump_util.h"
#include "common/util/op_info_util.h"
#include "tensor_engine/tbe_op_info.h"
#include "common/format/axis_util.h"
#include "common/format/range_axis_util.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class STEST_dump_util_stest : public testing::Test
{
protected:
    void SetUp()
    {
    }

    void TearDown()
    {
    }
};

TEST_F(STEST_dump_util_stest, case_dump_op_info_tensor)
{
    std::vector<te::TbeOpParam> puts;
    te::TbeOpParam tbe_op_param;
    std::vector<te::TbeOpTensor> tensors;
    te::TbeOpTensor tbe_op_tensor;
    std::vector<int64_t> slice_shape = {1, 2, 3, 4};
    std::vector<int64_t> offset = {256};
    tbe_op_tensor.SetShape(slice_shape);
    tbe_op_tensor.SetOriginShape(slice_shape);
    tbe_op_tensor.SetSgtSliceShape(slice_shape);
    tbe_op_tensor.SetValidShape(slice_shape);
    tbe_op_tensor.SetSliceOffset(offset);
    tensors.push_back(tbe_op_tensor);
    tbe_op_param.SetTensors(tensors);
    puts.push_back(tbe_op_param);
    std::string debug_str;
    bool flag = false;
    DumpOpInfoTensor(puts, debug_str);
    if (!debug_str.empty()) {
        flag = true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(STEST_dump_util_stest, dump_op_info)
{
    te::TbeOpInfo op_info = te::TbeOpInfo("", "", "", "");
    std::vector<te::TbeOpParam> puts;
    te::TbeOpParam tbe_op_param;
    std::vector<te::TbeOpTensor> tensors;
    te::TbeOpTensor tbe_op_tensor;
    std::vector<int64_t> slice_shape = {1, 2, 3, 4};
    std::vector<int64_t> offset = {256};
    tbe_op_tensor.SetShape(slice_shape);
    tbe_op_tensor.SetOriginShape(slice_shape);
    tbe_op_tensor.SetSgtSliceShape(slice_shape);
    tbe_op_tensor.SetValidShape(slice_shape);
    tbe_op_tensor.SetSliceOffset(offset);
    tensors.push_back(tbe_op_tensor);
    EXPECT_EQ(tensors.size(), 1);
    tbe_op_param.SetTensors(tensors);
    puts.push_back(tbe_op_param);
    op_info.SetInputs(puts);
    op_info.SetOutputs(puts);
    DumpOpInfo(op_info);
}

TEST_F(STEST_dump_util_stest, dump_l1_info)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    ge::OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
    std::vector<int64_t> in_memory_type_list = {2, 2};
    std::vector<int64_t> out_memory_type_list = {2, 2};
    (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type_list);
    (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type_list);
    ToOpStructPtr lx_info = std::make_shared<ToOpStruct>();
    lx_info->slice_input_shape = {{2,2,2,2}};
    lx_info->slice_output_shape = {{2,2,2,2}};
    lx_info->slice_input_offset = {{2,2,2,2}};
    lx_info->slice_output_offset = {{2,2,2,2}};
    op_desc->SetExtAttr(ge::ATTR_NAME_L1_FUSION_EXTEND_PTR, lx_info);
    ge::NodePtr node = graph->AddNode(op_desc);
    EXPECT_EQ(graph.get(), node->GetOwnerComputeGraph().get());
    DumpL1Attr(node.get());
}

TEST_F(STEST_dump_util_stest, dump_l2_info)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    ge::OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
    std::vector<int64_t> in_memory_type_list = {2, 2};
    std::vector<int64_t> out_memory_type_list = {2, 2};
    (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type_list);
    (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type_list);
    ToOpStructPtr lx_info = std::make_shared<ToOpStruct>();
    lx_info->slice_input_shape = {{2,2,2,2}};
    lx_info->slice_output_shape = {{2,2,2,2}};
    lx_info->slice_input_offset = {{2,2,2,2}};
    lx_info->slice_output_offset = {{2,2,2,2}};
    op_desc->SetExtAttr(ATTR_NAME_L2_FUSION_EXTEND_PTR, lx_info);
    ge::NodePtr node = graph->AddNode(op_desc);
    EXPECT_EQ(graph.get(), node->GetOwnerComputeGraph().get());
    DumpL2Attr(node.get());
}

TEST_F(STEST_dump_util_stest, check_params_test)
{
  vector<int64_t> original_dim_vec = {0, 1, 2};
  uint32_t c0 = 1;
  vector<int64_t> nd_value = {0, 1, 2, 3};
  size_t dim_default_size = DIM_DEFAULT_SIZE;
  Status status = AxisUtil::CheckParams(original_dim_vec, c0, nd_value, dim_default_size);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(STEST_dump_util_stest, CheckParamValue_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  size_t min_size = DIM_DEFAULT_SIZE;
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.CheckParamValue(original_range_vec, original_dim_vec, c0, range_value, min_size);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_dump_util_stest, GetRangeAxisValueByND_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByND(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_dump_util_stest, GetRangeAxisValueByNCHW_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0 = 16;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByNCHW(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_dump_util_stest, GetRangeAxisValueByNHWC_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0 = 16;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByNHWC(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_dump_util_stest, GetRangeAxisValueByNC1HWC0_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0 = 16;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByNC1HWC0(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_dump_util_stest, GetRangeAxisValueByHWCN_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0 = 16;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByHWCN(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_dump_util_stest, GetRangeAxisValueByCHWN_test) {
  vector<std::pair<int64_t, int64_t>> original_range_vec = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<int64_t> original_dim_vec = {1, 1, 1, 1};
  uint32_t c0 = 16;
  vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                     {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  RangeAxisUtil range_axis_util;
  Status ret = range_axis_util.GetRangeAxisValueByCHWN(original_range_vec, original_dim_vec, c0, range_value);
  EXPECT_EQ(ret, fe::SUCCESS);
}
