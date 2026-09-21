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
#define protected public
#define private public
#include "common/lxfusion_json_util.h"
#undef private
#undef protected

using namespace std;
using namespace ge;
using namespace fe;

class lxfusion_json_util_st : public testing::Test
{
protected:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(lxfusion_json_util_st, get_l1_info_from_json)
{
  OpDescPtr op_desc_ptr = nullptr;
  Status ret = GetL1InfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, FAILED);

  op_desc_ptr = std::make_shared<OpDesc>("relu1", "Relu");
  ret = GetL1InfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, FAILED);

  std::string str_l1_info = "";
  ge::AttrUtils::SetStr(op_desc_ptr, L1_FUSION_TO_OP_STRUCT, str_l1_info);
  ret = GetL1InfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(lxfusion_json_util_st, get_l2_info_from_json)
{
  OpDescPtr op_desc_ptr = nullptr;
  Status ret = GetL2InfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, FAILED);

  op_desc_ptr = std::make_shared<OpDesc>("relu1", "Relu");
  ret = GetL2InfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, FAILED);

  std::string str_l2_info = "";
  ge::AttrUtils::SetStr(op_desc_ptr, L2_FUSION_TO_OP_STRUCT, str_l2_info);
  ret = GetL2InfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(lxfusion_json_util_st, write_graph_infoto_json) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc;
  op_desc_ptr->AddInputDesc(tensor_desc);
  op_desc_ptr->AddOutputDesc(tensor_desc);
  fe::ToOpStructPtr l1_info_ptr = std::make_shared<fe::ToOpStruct_t>();
  l1_info_ptr->op_l1_fusion_type = {1};
  l1_info_ptr->op_l1_space = 2;
  l1_info_ptr->op_l1_workspace_flag = 3;
  l1_info_ptr->op_l1_workspace_size = 4;
  l1_info_ptr->slice_input_shape = {{5}};
  l1_info_ptr->slice_input_offset = {{6}};
  l1_info_ptr->slice_output_shape = {{7}};
  l1_info_ptr->slice_output_offset = {{8}};
  l1_info_ptr->total_shape = {9};
  l1_info_ptr->split_index = 0;
  op_desc_ptr->SetExtAttr("_l1_fusion_extend_content", l1_info_ptr);
  op_desc_ptr->SetExtAttr("l2_fusion_extend_content", l1_info_ptr);

  L2FusionInfoPtr l2_fusion_info_ptr = std::make_shared<fe::TaskL2FusionInfo_t>();
  l2_fusion_info_ptr->node_name = "some_node";
  l2_fusion_info_ptr->is_used = 1;

  L2FusionData_t l2_data;
  l2_data.l2Addr = 1;
  l2_data.l2Index = 2;
  l2_data.l2PageNum = 3;
  L2FusionDataMap_t l2_input_data_map;
  l2_input_data_map.emplace(0, l2_data);
  L2FusionDataMap_t l2_output_data_map;
  l2_output_data_map.emplace(0, l2_data);
  l2_fusion_info_ptr->input = l2_input_data_map;
  l2_fusion_info_ptr->output = l2_output_data_map;

  rtSmData_t sm_data;
  sm_data.L2_data_section_size = 1;
  sm_data.L2_load_to_ddr = 2;
  sm_data.L2_mirror_addr = 3;
  sm_data.L2_page_offset_base = 4;
  sm_data.L2_preload = 5;
  sm_data.modified = 6;
  sm_data.prev_L2_page_offset_base = 7;
  sm_data.priority = 8;
  rtSmDesc_t sm_desc;
  sm_desc.size = 8;
  sm_desc.data[0] = sm_data;
  sm_desc.l2_in_main = 9;
  fe_sm_desc_t fe_sm_desc;
  fe_sm_desc.l2ctrl = sm_desc;
  l2_fusion_info_ptr->l2_info = fe_sm_desc;
  op_desc_ptr->SetExtAttr("task_l2_fusion_info_extend_content", l2_fusion_info_ptr);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("test");
  NodePtr node_ptr = graph_ptr->AddNode(op_desc_ptr);
  Status ret = WriteGraphInfoToJson(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);

  ret = ReadGraphInfoFromJson(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);

  op_desc_ptr->SetExtAttr(fe::ATTR_NAME_L2_FUSION_EXTEND_PTR, nullptr);
  op_desc_ptr->SetExtAttr("_l1_fusion_extend_content", nullptr);
  ToOpStructPtr tmp_l2_info_ptr = nullptr;
  GetL2ToOpStructFromJson(op_desc_ptr, tmp_l2_info_ptr);
  ToOpStructPtr tmp_l1_info_ptr = nullptr;
  GetL1ToOpStructFromJson(op_desc_ptr, tmp_l1_info_ptr);
}

TEST_F(lxfusion_json_util_st, L2_fusion_info)
{
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("relu1", "Relu");
  L2FusionInfoPtr l2_fusion_info_ptr = nullptr;
  SetL2FusionInfoToNode(op_desc_ptr, l2_fusion_info_ptr);

  l2_fusion_info_ptr = std::make_shared<fe::TaskL2FusionInfo_t>();
  l2_fusion_info_ptr->node_name = "some_node";
  l2_fusion_info_ptr->is_used = 1;

  L2FusionData_t l2_data;
  l2_data.l2Addr = 1;
  l2_data.l2Index = 2;
  l2_data.l2PageNum = 3;
  L2FusionDataMap_t l2_input_data_map;
  l2_input_data_map.emplace(0, l2_data);
  L2FusionDataMap_t l2_output_data_map;
  l2_output_data_map.emplace(0, l2_data);
  l2_fusion_info_ptr->input = l2_input_data_map;
  l2_fusion_info_ptr->output = l2_output_data_map;

  rtSmData_t sm_data;
  sm_data.L2_data_section_size = 1;
  sm_data.L2_load_to_ddr = 2;
  sm_data.L2_mirror_addr = 3;
  sm_data.L2_page_offset_base = 4;
  sm_data.L2_preload = 5;
  sm_data.modified = 6;
  sm_data.prev_L2_page_offset_base = 7;
  sm_data.priority = 8;
  rtSmDesc_t sm_desc;
  sm_desc.size = 8;
  sm_desc.data[0] = sm_data;
  sm_desc.l2_in_main = 9;
  fe_sm_desc_t fe_sm_desc;
  fe_sm_desc.l2ctrl = sm_desc;
  l2_fusion_info_ptr->l2_info = fe_sm_desc;

  SetL2FusionInfoToNode(op_desc_ptr, l2_fusion_info_ptr);

  L2FusionInfoPtr l2_fusion_info_ptr2 = GetL2FusionInfoFromJson(op_desc_ptr);

  op_desc_ptr->DelAttr(ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR);
  op_desc_ptr->SetExtAttr(fe::ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, nullptr);
  L2FusionInfoPtr l2_fusion_info_ptr3 = GetL2FusionInfoFromJson(op_desc_ptr);

  Status ret = GetTaskL2FusionInfoFromJson(op_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(lxfusion_json_util_st, write_op_slice_info_to_json)
{
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc;
  op_desc_ptr->AddInputDesc(tensor_desc);
  op_desc_ptr->AddOutputDesc(tensor_desc);
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("test");
  NodePtr node_ptr = graph_ptr->AddNode(op_desc_ptr);

  Status ret = WriteOpSliceInfoToJson(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  ret = ReadOpSliceInfoFromJson(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);

  fe::OpCalcInfoPtr op_calc_info_ptr = std::make_shared<fe::OpCalcInfo>();
  op_calc_info_ptr->Initialize();
  AxisReduceMapPtr axis_reduce_map_ptr = std::make_shared<fe::AxisReduceMap>();
  axis_reduce_map_ptr->Initialize();
  std::vector<int64_t> axis = {1,2,3,4};
  InputReduceInfoPtr input_reduce_ptr = std::make_shared<fe::InputReduceInfo>();
  input_reduce_ptr->Initialize();
  input_reduce_ptr->SetIndex(0);
  input_reduce_ptr->SetAxis(axis);
  vector<InputReduceInfoPtr> input_reduce_vec = {input_reduce_ptr};
  OutputReduceInfoPtr output_reduce_ptr = std::make_shared<fe::OutputReduceInfo>();
  output_reduce_ptr->Initialize();
  output_reduce_ptr->SetIndex(0);
  output_reduce_ptr->SetIsAtomic(true);
  output_reduce_ptr->SetReduceType(REDUCE_MEAN);
  vector<OutputReduceInfoPtr> output_reduce_vec = {output_reduce_ptr};
  axis_reduce_map_ptr->SetInputReduceInfos(input_reduce_vec);
  axis_reduce_map_ptr->SetOutputReduceInfos(output_reduce_vec);
  vector<AxisReduceMapPtr> reduce_vec = {axis_reduce_map_ptr};
  AxisSplitMapPtr axis_split_map_ptr = std::make_shared<fe::AxisSplitMap>();
  axis_split_map_ptr->Initialize();
  InputSplitInfoPtr input_split_ptr = std::make_shared<fe::InputSplitInfo>();
  input_split_ptr->Initialize();
  input_split_ptr->SetIndex(0);
  input_split_ptr->SetAxis(axis);
  input_split_ptr->SetHeadOverLap(axis);
  input_split_ptr->SetTailOverLap(axis);
  vector<InputSplitInfoPtr> input_split_vec = {input_split_ptr};
  OutputSplitInfoPtr output_split_ptr = std::make_shared<fe::OutputSplitInfo>();
  output_split_ptr->Initialize();
  output_split_ptr->SetIndex(0);
  output_split_ptr->SetAxis(axis);
  vector<OutputSplitInfoPtr> output_split_vec = {output_split_ptr};
  axis_split_map_ptr->SetInputSplitInfos(input_split_vec);
  axis_split_map_ptr->SetOutputSplitInfos(output_split_vec);
  vector<AxisSplitMapPtr> split_vec = {axis_split_map_ptr};
  op_calc_info_ptr->SetAxisReduceMaps(reduce_vec);
  op_calc_info_ptr->SetAxisSplitMaps(split_vec);
  op_desc_ptr->SetExtAttr(FUSION_OP_SLICE_INFO, op_calc_info_ptr);

  std::string op_slice_info_str;
  fe::OpCalcInfo op_calc_info;
  SetOpSliceInfoToJson(*op_calc_info_ptr, op_slice_info_str);
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  SetFusionOpSliceInfoToJson(*op_calc_info_ptr, op_slice_info_str);
  GetFusionOpSliceInfoFromJson(op_calc_info, op_slice_info_str);

  ret = WriteOpSliceInfoToJson(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  ret = ReadOpSliceInfoFromJson(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
}
