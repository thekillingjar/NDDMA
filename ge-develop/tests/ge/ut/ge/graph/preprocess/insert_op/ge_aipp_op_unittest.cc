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
#include <iostream>
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/graph_builder_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "common/context/local_context.h"
#include "ge_running_env/path_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/preprocess/insert_op/aipp_op.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class UtestGeAipp : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
  NodePtr MyAddNode(ut::GraphBuilder &builder, const string &name, const string &type, int in_cnt, int out_cnt,
                    Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT) {
    auto node = builder.AddNode(name, type, in_cnt, out_cnt);
    auto op_desc = node->GetOpDesc();
    op_desc->SetInputOffset(std::vector<int64_t>(in_cnt));
    op_desc->SetOutputOffset(std::vector<int64_t>(out_cnt));
    return node;
  };
};
class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

ComputeGraphPtr BuildGraph() {
  auto builder = ut::GraphBuilder("mbatch_Case");

  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetStr(data_desc, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  AttrUtils::SetStr(data_desc, "mbatch-switch-name", "case1");
  AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, 0);

  auto mapindex1 = builder.AddNode("mapindex1", "MapIndex", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(mapindex1, 0, case1, 0);
  builder.AddDataEdge(data1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput1, 0);

  return builder.GetGraph();
}

TEST_F(UtestGeAipp, test_Cinvert_related_input_name_to_rank) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  domi::AippOpParams params;
  params.set_related_input_name("data0");
  AippOp aipp_op;
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op.ConvertRelatedInputNameToRank();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(aipp_op.aipp_params_->related_input_rank(), 0);

  domi::AippOpParams params1;
  params1.set_related_input_name("data1");
  AippOp aipp_op1;
  auto ret1 = aipp_op1.Init(&params1);
  ASSERT_EQ(ret1, SUCCESS);
  ret1 = aipp_op1.ConvertRelatedInputNameToRank();
  EXPECT_EQ(ret1, PARAM_INVALID);
  EXPECT_EQ(aipp_op1.aipp_params_->related_input_rank(), 0);
  // Reset global context
  domi::GetContext().data_tensor_names = data_tensor_names_old;

  NodePtr aipp_node = nullptr;
  ret = aipp_op.AddNodeToGraph(aipp_node, 0);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGeAipp, test_Insert_Aipp_ToGraph) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  {
    AippOp aipp_op;
    domi::AippOpParams params;
    std::string aippConfigPath;
    const uint32_t index = 1;
    params.set_related_input_name("data1");
    ASSERT_EQ(aipp_op.Init(&params), SUCCESS);
    aipp_op.InsertAippToGraph(graph, aippConfigPath, index);
  }

  {
    AippOp aipp_op1;
    domi::AippOpParams params1;
    std::string aippConfigPath1;
    const uint32_t index1 = 1;
    params1.set_related_input_name("data0");
    ASSERT_EQ(aipp_op1.Init(&params1), SUCCESS);
    aipp_op1.InsertAippToGraph(graph, aippConfigPath1, index1);
  }

  // Reset global context
  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Validate_Params) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  int32_t val = 10;
  ::domi::AippOpParams_AippMode *value = (::domi::AippOpParams_AippMode *)(&val);
  params.set_aipp_mode(*value);
  auto ret = aipp_op.ValidateParams();
  ASSERT_EQ(ret, PARAM_INVALID);

  params.set_aipp_mode(domi::AippOpParams::undefined);
  ret = aipp_op.ValidateParams();
  ASSERT_EQ(ret, PARAM_INVALID);

  AippOp aipp_op1;
  params.set_aipp_mode(domi::AippOpParams::dynamic);
  ret = aipp_op1.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op1.ValidateParams();
  ASSERT_EQ(ret, PARAM_INVALID);

  AippOp aipp_op2;
  params.set_aipp_mode(domi::AippOpParams::static_);
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  ret = aipp_op2.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op2.ValidateParams();
  ASSERT_EQ(ret, SUCCESS);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Set_Csc_Default_Value) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  aipp_op.SetCscDefaultValue();

  AippOp aipp_op1;
  params.set_input_format(domi::AippOpParams::XRGB8888_U8);
  ret = aipp_op1.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  aipp_op1.SetCscDefaultValue();

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Set_Dtc_Default_Value) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  aipp_op.SetDtcDefaultValue();

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Convert_Param_To_Attr) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  params.set_aipp_mode(domi::AippOpParams::static_);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ge::NamedAttrs aipp_attrs;
  aipp_op.ConvertParamToAttr(aipp_attrs);

  AippOp aipp_op1;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  params.set_aipp_mode(domi::AippOpParams::dynamic);
  ret = aipp_op1.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  aipp_op1.ConvertParamToAttr(aipp_attrs);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Set_Default_Params) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  params.set_aipp_mode(domi::AippOpParams::static_);
  params.set_csc_switch(true);
  params.set_crop(false);
  params.set_resize(false);
  params.set_padding(false);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ge::NamedAttrs aipp_attrs;
  aipp_op.ConvertParamToAttr(aipp_attrs);
  aipp_op.SetDefaultParams();
  ASSERT_EQ(ret, SUCCESS);
  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_convert_dynamic_Params_to_json) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_aipp_mode(domi::AippOpParams::dynamic);
  params.set_related_input_rank(2);
  params.set_related_input_name("aipp_dynamic_node");
  params.set_max_src_image_size(5);
  params.set_support_rotation(true);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  auto json_str = aipp_op.ConvertParamToJson();
  std::cout << json_str << std::endl;

  ASSERT_NE(json_str.find("aipp_mode"), json_str.npos);
  ASSERT_NE(json_str.find("max_src_image_size"), json_str.npos);
  ASSERT_EQ(json_str.find("related_input_name"), json_str.npos);
  ASSERT_NE(json_str.find("related_input_rank"), json_str.npos);
  ASSERT_EQ(json_str.find("support_rotation"), json_str.npos);
  ASSERT_EQ(json_str.find("raw_rgbir_to_f16_n"), json_str.npos);
  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_convert_static_Params_to_json) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");


  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_aipp_mode(domi::AippOpParams::static_);
  params.set_related_input_rank(2);
  params.set_related_input_name("aipp_dynamic_node");
  for (int i = 0; i < 3; i++) {
    params.add_input_edge_idx(i);
  }
  // set static data
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  params.set_csc_switch(true);
  params.set_rbuv_swap_switch(true);
  params.set_ax_swap_switch(false);
  params.set_single_line_mode(true);
  params.set_src_image_size_w(20);
  params.set_src_image_size_h(20);
  params.set_crop(true);
  params.set_load_start_pos_w(10);
  params.set_load_start_pos_h(10);
  params.set_crop_size_w(20);
  params.set_crop_size_h(20);
  params.set_resize(true);
  params.set_resize_output_w(5);
  params.set_resize_output_h(5);
  params.set_padding(true);
  params.set_left_padding_size(8);
  params.set_right_padding_size(8);
  params.set_top_padding_size(8);
  params.set_bottom_padding_size(8);
  params.set_padding_value(2.0);
  params.set_mean_chn_0(2);
  params.set_mean_chn_1(2);
  params.set_mean_chn_2(2);
  params.set_mean_chn_3(2);
  params.set_min_chn_0(2);
  params.set_min_chn_1(2);
  params.set_min_chn_2(2);
  params.set_min_chn_3(2);
  params.add_var_reci_chn_0(2.0);
  params.add_var_reci_chn_1(2.0);
  params.add_var_reci_chn_2(2.0);
  params.add_var_reci_chn_3(2.0);
  params.add_matrix_r0c0(3);
  params.add_matrix_r0c1(3);
  params.add_matrix_r0c2(3);
  params.add_matrix_r1c0(3);
  params.add_matrix_r1c1(3);
  params.add_matrix_r1c2(3);
  params.add_matrix_r2c0(3);
  params.add_matrix_r2c1(3);
  params.add_matrix_r2c2(3);
  params.add_output_bias_0(3);
  params.add_output_bias_1(3);
  params.add_output_bias_2(3);
  params.add_input_bias_0(3);
  params.add_input_bias_1(3);
  params.add_input_bias_2(3);

  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  auto json_str = aipp_op.ConvertParamToJson();
  std::cout << json_str << std::endl;
  
  ASSERT_NE(json_str.find("aipp_mode"), json_str.npos);
  ASSERT_EQ(json_str.find("ax_swap_switch"), json_str.npos);
  ASSERT_NE(json_str.find("bottom_padding_size"), json_str.npos);
  ASSERT_EQ(json_str.find("cpadding_value"), json_str.npos);
  ASSERT_NE(json_str.find("crop"), json_str.npos);
  ASSERT_NE(json_str.find("crop_size_h"), json_str.npos);
  ASSERT_NE(json_str.find("crop_size_w"), json_str.npos);
  ASSERT_NE(json_str.find("csc_switch"), json_str.npos);
  ASSERT_NE(json_str.find("input_bias_0"), json_str.npos);
  ASSERT_NE(json_str.find("input_bias_1"), json_str.npos);
  ASSERT_NE(json_str.find("input_bias_2"), json_str.npos);
  ASSERT_EQ(json_str.find("input_edge_idx"), json_str.npos);
  ASSERT_NE(json_str.find("input_format"), json_str.npos);
  ASSERT_NE(json_str.find("left_padding_size"), json_str.npos);
  ASSERT_NE(json_str.find("load_start_pos_h"), json_str.npos);
  ASSERT_NE(json_str.find("load_start_pos_w"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r0c0"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r0c1"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r0c2"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r1c0"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r1c1"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r1c2"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r2c0"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r2c1"), json_str.npos);
  ASSERT_NE(json_str.find("matrix_r2c2"), json_str.npos);
  ASSERT_NE(json_str.find("mean_chn_0"), json_str.npos);
  ASSERT_NE(json_str.find("mean_chn_1"), json_str.npos);
  ASSERT_NE(json_str.find("mean_chn_2"), json_str.npos);
  ASSERT_NE(json_str.find("mean_chn_3"), json_str.npos);
  ASSERT_NE(json_str.find("min_chn_0"), json_str.npos);
  ASSERT_NE(json_str.find("min_chn_1"), json_str.npos);
  ASSERT_NE(json_str.find("min_chn_2"), json_str.npos);
  ASSERT_NE(json_str.find("min_chn_3"), json_str.npos);
  ASSERT_NE(json_str.find("output_bias_0"), json_str.npos);
  ASSERT_NE(json_str.find("output_bias_1"), json_str.npos);
  ASSERT_NE(json_str.find("output_bias_2"), json_str.npos);
  ASSERT_NE(json_str.find("padding"), json_str.npos);
  ASSERT_NE(json_str.find("padding_value"), json_str.npos);
  ASSERT_EQ(json_str.find("raw_rgbir_to_f16_n"), json_str.npos);
  ASSERT_NE(json_str.find("rbuv_swap_switch"), json_str.npos);
  ASSERT_EQ(json_str.find("related_input_name"), json_str.npos);
  ASSERT_NE(json_str.find("related_input_rank"), json_str.npos);
  ASSERT_NE(json_str.find("resize"), json_str.npos);
  ASSERT_NE(json_str.find("resize_output_h"), json_str.npos);
  ASSERT_NE(json_str.find("resize_output_w"), json_str.npos);
  ASSERT_NE(json_str.find("right_padding_size"), json_str.npos);
  ASSERT_EQ(json_str.find("single_line_mode"), json_str.npos);
  ASSERT_NE(json_str.find("src_image_size_h"), json_str.npos);
  ASSERT_NE(json_str.find("src_image_size_w"), json_str.npos);
  ASSERT_NE(json_str.find("top_padding_size"), json_str.npos);
  ASSERT_NE(json_str.find("var_reci_chn_3"), json_str.npos);
  ASSERT_NE(json_str.find("var_reci_chn_2"), json_str.npos);
  ASSERT_NE(json_str.find("var_reci_chn_1"), json_str.npos);
  ASSERT_NE(json_str.find("var_reci_chn_0"), json_str.npos);
  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Generate_OpDesc) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  aipp_op.GenerateOpDesc(desc_ptr);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Add_Attr_To_AippData) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  AippOp aipp_op;
  domi::AippOpParams params;
  params.set_input_format(domi::AippOpParams::YUV420SP_U8);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  aipp_op.AddAttrToAippData(desc_ptr);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Find_Data_By_Index) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  AippOp aipp_op;
  domi::AippOpParams params;
  const uint32_t index = 1;
  std::string aippConfigPath;
  params.set_related_input_name("data1");
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op.InsertAippToGraph(graph, aippConfigPath, index);
  ge::NodePtr data2 = nullptr;
  data2 = aipp_op.FindDataByIndex(graph, 0);
  if (data2 != nullptr) {
      ret = SUCCESS;
      ASSERT_EQ(ret, SUCCESS);
  }

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Add_Aipp_Attrbutes) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  AippOp aipp_op;
  domi::AippOpParams params;
  const uint32_t index = 0;
  params.set_related_input_name("data1");
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  const std::string aipp_cfg_path = "/root/";
  ret = aipp_op.AddAippAttrbutes(desc_ptr, index);
  ASSERT_EQ(ret, SUCCESS);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Get_Target_Position) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr data1 = NodeBuilder("data1", DATA)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .Build(graph);
  ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AippOp aipp_op;
  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_related_input_name("data1");
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op.GetTargetPosition(graph, data1, target_edges);
  ASSERT_EQ(ret, SUCCESS);

  ge::NodePtr data2 = NodeBuilder("case1", CASE)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .Build(graph);
  AippOp aipp_op1;
  domi::AippOpParams params1;
  params1.set_related_input_name("data1");
  ret = aipp_op1.Init(&params1);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op1.GetTargetPosition(graph, data2, target_edges);
  ASSERT_EQ(ret, SUCCESS);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Get_Static_TargetNode) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr data1 = NodeBuilder("data1", DATA)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .Build(graph);
  ge::NodePtr data2 = NodeBuilder("case1", CASE)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                    .Build(graph);
  AippOp aipp_op;
  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op.GetStaticTargetNode(data1, data2);
  ASSERT_EQ(ret, SUCCESS);

  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_Insert_Aipp_ToGraph1) {
  // Set global context
  auto data_tensor_names_old = domi::GetContext().data_tensor_names;
  domi::GetContext().data_tensor_names.push_back("data0");
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  AippOp aipp_op;
  domi::AippOpParams params;
  const uint32_t index = 1;
  std::string aippConfigPath;
  params.set_related_input_name("data0");
  auto ret = aipp_op.Init(&params);
  ASSERT_EQ(ret, SUCCESS);
  ret = aipp_op.ConvertRelatedInputNameToRank();
  EXPECT_EQ(ret, SUCCESS);
  aipp_op.InsertAippToGraph(graph, aippConfigPath, index);
  // Reset global context
  domi::GetContext().data_tensor_names = data_tensor_names_old;
}

TEST_F(UtestGeAipp, test_CreateAippData5) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_NHWC, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NHWC, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  (void)aipp_op.Init(&params);
  int32_t rank = 0;
  NodePtr target;
  std::set<uint32_t> edge_indexes;
  aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes);
  EXPECT_EQ(aipp_op.CreateAippData(aipp), PARAM_INVALID);
}

TEST_F(UtestGeAipp, test_CreateAippData1) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NHWC;
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_NHWC, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NHWC, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  (void)aipp_op.Init(&params);
  int32_t rank = 0;
  NodePtr target;
  std::set<uint32_t> edge_indexes;
  aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes);
  EXPECT_EQ(aipp_op.CreateAippData(aipp), INTERNAL_ERROR);
}

TEST_F(UtestGeAipp, test_CreateAippData2) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224}), FORMAT_ND, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  std::vector<int64_t> origin_input_dims = {224};
  AttrUtils::SetListInt(data1->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;

  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::dynamic);
  (void)aipp_op.Init(&params);
  int32_t rank = 0;
  NodePtr target;
  std::set<uint32_t> edge_indexes;
  aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes);
  EXPECT_EQ(aipp_op.CreateAippData(aipp), PARAM_INVALID);
}

TEST_F(UtestGeAipp, test_CreateAippData3) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{-1,3,224,224}), FORMAT_NCHW, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{-1,3,224,224}), FORMAT_ND, DT_FLOAT));
  std::vector<int64_t> origin_input_dims = {-1,3,224,224};
  AttrUtils::SetListInt(data1->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;

  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  (void)aipp_op.Init(&params);
  int32_t rank = 0;
  NodePtr target;
  std::set<uint32_t> edge_indexes;
  aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes);
  EXPECT_EQ(aipp_op.CreateAippData(aipp), SUCCESS);
}

TEST_F(UtestGeAipp, test_CreateAippData4) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2,3,224,224}), FORMAT_NCHW, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2,3,224,224}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  std::vector<int64_t> origin_input_dims = {2,3,224,224};
  AttrUtils::SetListInt(data1->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);

  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  (void)aipp_op.Init(&params);
  int32_t rank = 0;
  NodePtr target;
  std::set<uint32_t> edge_indexes;
  aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes);
  EXPECT_EQ(aipp_op.CreateAippData(aipp), INTERNAL_ERROR);
}

TEST_F(UtestGeAipp, test_GetStaticTargetNode1) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{3, 224, 224}), FORMAT_NCHW, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0,
                                       GeTensorDesc(GeShape(std::vector<int64_t>{3, 224, 224}), FORMAT_NCHW, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
  std::vector<int64_t> origin_input_dims = {3, 224, 224};
  AttrUtils::SetStr(data1->GetOpDesc(), "mbatch-switch-name", "");

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;

  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  EXPECT_EQ(aipp_op.Init(&params), SUCCESS);
  NodePtr target;
  EXPECT_EQ(aipp_op.GetStaticTargetNode(data1, target), SUCCESS);
}

TEST_F(UtestGeAipp, test_GetAndCheckTarget1) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Const", 1, 1);

  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  int32_t rank = 0;
  NodePtr target;
  std::set<uint32_t> edge_indexes;
  EXPECT_EQ(aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes), PARAM_INVALID);
}

TEST_F(UtestGeAipp, test_GetAndCheckTarget2) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{3, 224, 224}), FORMAT_NCHW, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0,
                                       GeTensorDesc(GeShape(std::vector<int64_t>{3, 224, 224}), FORMAT_NCHW, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
  AttrUtils::SetBool(data1->GetOpDesc(), "mbatch-inserted-node", false);
  AttrUtils::SetBool(data2->GetOpDesc(), "mbatch-inserted-node", true);
  std::string test = "test";
  AttrUtils::SetStr(data1->GetOpDesc(), "_user_defined_data_type", test);
  AttrUtils::SetStr(data2->GetOpDesc(), "_user_defined_data_type", test);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;

  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::dynamic);
  (void)aipp_op.Init(&params);
  NodePtr target;
  int32_t rank = 0;
  std::set<uint32_t> edge_indexes;
  EXPECT_EQ(aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes), PARAM_INVALID);
}

TEST_F(UtestGeAipp, test_GetAndCheckTarget3) {
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  ge::AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{3, 224, 224}), FORMAT_NCHW, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0,
                                       GeTensorDesc(GeShape(std::vector<int64_t>{3, 224, 224}), FORMAT_NCHW, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;

  domi::AippOpParams params;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::dynamic);
  (void)aipp_op.Init(&params);
  NodePtr target;
  int32_t rank = 0;
  std::set<uint32_t> edge_indexes;
  EXPECT_EQ(aipp_op.GetAndCheckTarget(computeGraph, rank, target, edge_indexes), SUCCESS);
}

/*
 *         NetOutput
 *            |
 *  + -------------------------- +
 *  |        |                  |
 *  |  Subgraph-1-NetOutput     |
 *  |        |                  |
 *  |      conv                 | -------> (partitionedcall | subgraph)
 *  |     /    \                |
 *  |  data00 data01            |
 *  + --------------------------+
 *    /    \
 *  data1 data2
 */
TEST_F(UtestGeAipp, test_FindDataNodeInSubgraph1) {
  ComputeGraphPtr root_graph;
  NodePtr partitioned_call;
  ComputeGraphPtr subgraph;
  {  // build root_graph
    auto builder_root = ut::GraphBuilder("root_graph");
    auto data1 = MyAddNode(builder_root, "data1", DATA, 1, 1);
    auto data2 = MyAddNode(builder_root, "data2", DATA, 1, 1);
    auto net_output = MyAddNode(builder_root, "NetOutput", NETOUTPUT, 1, 1);
    partitioned_call = MyAddNode(builder_root, "PartitionedCall", PARTITIONEDCALL, 2, 1);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);

    builder_root.AddDataEdge(data1, 0, partitioned_call, 0);
    builder_root.AddDataEdge(data2, 0, partitioned_call, 1);
    builder_root.AddDataEdge(partitioned_call, 0, net_output, 0);
    root_graph = builder_root.GetGraph();
  }

  {  // build subgraph1
    auto builder_subgraph = ut::GraphBuilder("subgraph");
    auto data00 = MyAddNode(builder_subgraph, "data00", DATA, 1, 1);
    auto data01 = MyAddNode(builder_subgraph, "data01", DATA, 1, 1);
    auto conv_node = MyAddNode(builder_subgraph, "conv_node", CONV2D, 2, 1);
    auto net_output = MyAddNode(builder_subgraph, "Subgraph-1-NetOutput", NETOUTPUT, 1, 1);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

    builder_subgraph.AddDataEdge(data00, 0, conv_node, 0);
    builder_subgraph.AddDataEdge(data01, 0, conv_node, 1);
    builder_subgraph.AddDataEdge(conv_node, 0, net_output, 0);
    subgraph = builder_subgraph.GetGraph();

    subgraph->SetParentNode(partitioned_call);
    subgraph->SetParentGraph(root_graph);
    partitioned_call->GetOpDesc()->AddSubgraphName("subgraph");
    partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    root_graph->AddSubgraph(subgraph);
  }

  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 0)->GetName(), "data00");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 1)->GetName(), "data01");
}

/*
 *               NetOutput
 *                  |
 *  + --------------------------- +
 *  |              |              |
 *  |     Subgraph-1-NetOutput    |
 *  |             |               |
 *  |        c   o   n   v        | -------> (partitionedcall | subgraph0)
 *  |     /    \     \    \       |
 *  | data00 data01 data10 data11 |
 *  + --------------------------- +
 *    /    \       /    \
 *  data1 data2  data3 data4
 */
TEST_F(UtestGeAipp, test_FindDataNodeInSubgraph2) {
  NodePtr partitioned_call0;
  ComputeGraphPtr root_graph;
  ComputeGraphPtr subgraph0;

  {  // build root_graph
    auto builder_root = ut::GraphBuilder("root_graph");
    auto data0 = MyAddNode(builder_root, "data0", DATA, 1, 1);
    auto data1 = MyAddNode(builder_root, "data1", DATA, 1, 1);
    auto data2 = MyAddNode(builder_root, "data2", DATA, 1, 1);
    auto data3 = MyAddNode(builder_root, "data3", DATA, 1, 1);
    partitioned_call0 = MyAddNode(builder_root, "partitioned_call0", PARTITIONEDCALL, 4, 1);
    auto net_output_root = MyAddNode(builder_root, "net_output_root", NETOUTPUT, 1, 1);

    AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_INDEX, 3);

    builder_root.AddDataEdge(data0, 0, partitioned_call0, 0);
    builder_root.AddDataEdge(data1, 0, partitioned_call0, 1);
    builder_root.AddDataEdge(data2, 0, partitioned_call0, 2);
    builder_root.AddDataEdge(data3, 0, partitioned_call0, 3);
    builder_root.AddDataEdge(partitioned_call0, 0, net_output_root, 0);
    root_graph = builder_root.GetGraph();
  }

  {  // build subgraph0
    auto builder_subgraph0 = ut::GraphBuilder("subgraph0");
    auto data00 = MyAddNode(builder_subgraph0, "data00", DATA, 1, 1);
    auto data01 = MyAddNode(builder_subgraph0, "data01", DATA, 1, 1);
    auto data10 = MyAddNode(builder_subgraph0, "data10", DATA, 1, 1);
    auto data11 = MyAddNode(builder_subgraph0, "data11", DATA, 1, 1);
    auto conv_node0 = MyAddNode(builder_subgraph0, "conv0_node", CONV2D, 2, 1);
    auto conv_node1 = MyAddNode(builder_subgraph0, "conv1_node", CONV2D, 2, 1);
    auto net_output_subgraph0 = MyAddNode(builder_subgraph0, "Subgraph0-NetOutput", NETOUTPUT, 2, 1);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data10->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data11->GetOpDesc(), ATTR_NAME_INDEX, 3);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(data10->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
    AttrUtils::SetInt(data11->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);

    builder_subgraph0.AddDataEdge(data00, 0, conv_node0, 0);
    builder_subgraph0.AddDataEdge(data01, 0, conv_node0, 1);
    builder_subgraph0.AddDataEdge(data10, 0, conv_node1, 0);
    builder_subgraph0.AddDataEdge(data11, 0, conv_node1, 1);

    builder_subgraph0.AddDataEdge(conv_node0, 0, net_output_subgraph0, 0);
    builder_subgraph0.AddDataEdge(conv_node1, 0, net_output_subgraph0, 1);
    subgraph0 = builder_subgraph0.GetGraph();

    subgraph0->SetParentNode(partitioned_call0);
    subgraph0->SetParentGraph(root_graph);
    partitioned_call0->GetOpDesc()->AddSubgraphName("subgraph0");
    partitioned_call0->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph0");
    root_graph->AddSubgraph("subgraph0", subgraph0);
  }
  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 0)->GetName(), "data00");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 1)->GetName(), "data01");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 2)->GetName(), "data10");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 3)->GetName(), "data11");
}

/*
 *+ -------------------------- +
 *|        |                  |
 *|  Subgraph00-NetOutput     |
 *|        |                  |
 *|   conv_node00             |                net_output_root
 *|     /    \                |                   |
 *|  data000 data001          |   + -------------------------------- +
 *+ --------------------------+   |              |                  |
 *                  \             |     Subgraph0-NetOutput         |
 *                   \            |           |          \          |
 *               subgraph00<-------partitioned_call00 conv_node1    | -------> (partitionedcall | subgraph0)
 *                                |    /     \          /   \       |
 *                                | data00 data01    data10 data11    |
 *                                + ------------------------------- +
 *                                  /    \            /    \
 *                                data1 data2       data3 data4
 */
TEST_F(UtestGeAipp, test_FindDataNodeInSubgraph3) {
  NodePtr partitioned_call0;
  NodePtr partitioned_call00;
  ComputeGraphPtr root_graph;
  ComputeGraphPtr subgraph0;
  ComputeGraphPtr subgraph00;

  {  // build root_graph
    auto builder_root = ut::GraphBuilder("root_graph");
    auto data0 = MyAddNode(builder_root, "data0", DATA, 1, 1);
    auto data1 = MyAddNode(builder_root, "data1", DATA, 1, 1);
    auto data2 = MyAddNode(builder_root, "data2", DATA, 1, 1);
    auto data3 = MyAddNode(builder_root, "data3", DATA, 1, 1);
    partitioned_call0 = MyAddNode(builder_root, "partitioned_call0", PARTITIONEDCALL, 4, 1);
    auto net_output_root = MyAddNode(builder_root, "net_output_root", NETOUTPUT, 1, 1);

    AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_INDEX, 3);

    builder_root.AddDataEdge(data0, 0, partitioned_call0, 0);
    builder_root.AddDataEdge(data1, 0, partitioned_call0, 1);
    builder_root.AddDataEdge(data2, 0, partitioned_call0, 2);
    builder_root.AddDataEdge(data3, 0, partitioned_call0, 3);
    builder_root.AddDataEdge(partitioned_call0, 0, net_output_root, 0);
    root_graph = builder_root.GetGraph();
  }

  {  // build subgraph0
    auto builder_subgraph0 = ut::GraphBuilder("subgraph0");
    auto data00 = MyAddNode(builder_subgraph0, "data00", DATA, 1, 1);
    auto data01 = MyAddNode(builder_subgraph0, "data01", DATA, 1, 1);
    auto data10 = MyAddNode(builder_subgraph0, "data10", DATA, 1, 1);
    auto data11 = MyAddNode(builder_subgraph0, "data11", DATA, 1, 1);
    partitioned_call00 = MyAddNode(builder_subgraph0, "partitioned_call00", PARTITIONEDCALL, 2, 1);
    auto conv_node1 = MyAddNode(builder_subgraph0, "conv_node1", CONV2D, 2, 1);
    auto net_output_subgraph0 = MyAddNode(builder_subgraph0, "Subgraph0-NetOutput", NETOUTPUT, 2, 1);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data10->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data11->GetOpDesc(), ATTR_NAME_INDEX, 3);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(data10->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
    AttrUtils::SetInt(data11->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);

    builder_subgraph0.AddDataEdge(data00, 0, partitioned_call00, 0);
    builder_subgraph0.AddDataEdge(data01, 0, partitioned_call00, 1);
    builder_subgraph0.AddDataEdge(data10, 0, conv_node1, 0);
    builder_subgraph0.AddDataEdge(data11, 0, conv_node1, 1);

    builder_subgraph0.AddDataEdge(partitioned_call00, 0, net_output_subgraph0, 0);
    builder_subgraph0.AddDataEdge(conv_node1, 0, net_output_subgraph0, 1);
    subgraph0 = builder_subgraph0.GetGraph();

    subgraph0->SetParentNode(partitioned_call0);
    subgraph0->SetParentGraph(root_graph);
    partitioned_call0->GetOpDesc()->AddSubgraphName("subgraph0");
    partitioned_call0->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph0");
    root_graph->AddSubgraph("subgraph0", subgraph0);
  }

  {  // build subgraph00
    auto builder_subgraph00 = ut::GraphBuilder("subgraph00");
    auto data000 = MyAddNode(builder_subgraph00, "data000", DATA, 1, 1);
    auto data001 = MyAddNode(builder_subgraph00, "data001", DATA, 1, 1);
    auto conv_node00 = MyAddNode(builder_subgraph00, "conv_node00", CONV2D, 2, 1);
    auto net_output_subgraph00 = MyAddNode(builder_subgraph00, "Subgraph00-NetOutput", NETOUTPUT, 1, 1);

    AttrUtils::SetInt(data000->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data001->GetOpDesc(), ATTR_NAME_INDEX, 1);

    AttrUtils::SetInt(data000->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(data001->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

    builder_subgraph00.AddDataEdge(data000, 0, conv_node00, 0);
    builder_subgraph00.AddDataEdge(data001, 0, conv_node00, 1);
    builder_subgraph00.AddDataEdge(conv_node00, 0, net_output_subgraph00, 0);

    subgraph00 = builder_subgraph00.GetGraph();
    subgraph00->SetParentNode(partitioned_call00);
    subgraph00->SetParentGraph(subgraph0);
    partitioned_call00->GetOwnerComputeGraph()->AddSubgraph(subgraph00);
    root_graph->AddSubgraph(subgraph00);
  }

  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 0)->GetName(), "data000");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 1)->GetName(), "data001");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 2)->GetName(), "data10");
  EXPECT_EQ(aipp_op.FindDataByIndex(root_graph, 3)->GetName(), "data11");
}

/*
 *               NetOutput
 *                  |
 *  + --------------------------- +
 *  |              |              |
 *  |     Subgraph-1-NetOutput    |
 *  |             |               |
 *  |        c   o   n   v        | -------> (partitionedcall | subgraph0)
 *  |     /    \     \    \       |
 *  | data00 data01 data10 data11 |
 *  + --------------------------- +
 *    /    \       /    \
 *  data1 data2  data3 data4
 */
TEST_F(UtestGeAipp, test_InsertAippInSubGraph) {
  NodePtr partitioned_call0;
  ComputeGraphPtr root_graph;
  ComputeGraphPtr subgraph0;

  {  // build root_graph
    auto builder_root = ut::GraphBuilder("root_graph");
    auto data0 = MyAddNode(builder_root, "data0", DATA, 1, 1);
    auto data1 = MyAddNode(builder_root, "data1", DATA, 1, 1);
    auto data2 = MyAddNode(builder_root, "data2", DATA, 1, 1);
    auto data3 = MyAddNode(builder_root, "data3", DATA, 1, 1);
    partitioned_call0 = MyAddNode(builder_root, "partitioned_call0", PARTITIONEDCALL, 4, 1);
    auto net_output_root = MyAddNode(builder_root, "net_output_root", NETOUTPUT, 1, 1);

    AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_INDEX, 3);

    builder_root.AddDataEdge(data0, 0, partitioned_call0, 0);
    builder_root.AddDataEdge(data1, 0, partitioned_call0, 1);
    builder_root.AddDataEdge(data2, 0, partitioned_call0, 2);
    builder_root.AddDataEdge(data3, 0, partitioned_call0, 3);
    builder_root.AddDataEdge(partitioned_call0, 0, net_output_root, 0);
    root_graph = builder_root.GetGraph();
  }

  {  // build subgraph0
    auto builder_subgraph0 = ut::GraphBuilder("subgraph0");
    auto data00 = MyAddNode(builder_subgraph0, "data00", DATA, 1, 1);
    auto data01 = MyAddNode(builder_subgraph0, "data01", DATA, 1, 1);
    auto data10 = MyAddNode(builder_subgraph0, "data10", DATA, 1, 1);
    auto data11 = MyAddNode(builder_subgraph0, "data11", DATA, 1, 1);
    auto conv_node0 = MyAddNode(builder_subgraph0, "conv0_node", CONV2D, 2, 1);
    auto conv_node1 = MyAddNode(builder_subgraph0, "conv1_node", CONV2D, 2, 1);
    auto net_output_subgraph0 = MyAddNode(builder_subgraph0, "Subgraph0-NetOutput", NETOUTPUT, 2, 1);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data10->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data11->GetOpDesc(), ATTR_NAME_INDEX, 3);

    AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(data10->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
    AttrUtils::SetInt(data11->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);

    builder_subgraph0.AddDataEdge(data00, 0, conv_node0, 0);
    builder_subgraph0.AddDataEdge(data01, 0, conv_node0, 1);
    builder_subgraph0.AddDataEdge(data10, 0, conv_node1, 0);
    builder_subgraph0.AddDataEdge(data11, 0, conv_node1, 1);

    builder_subgraph0.AddDataEdge(conv_node0, 0, net_output_subgraph0, 0);
    builder_subgraph0.AddDataEdge(conv_node1, 0, net_output_subgraph0, 1);
    subgraph0 = builder_subgraph0.GetGraph();

    subgraph0->SetParentNode(partitioned_call0);
    subgraph0->SetParentGraph(root_graph);
    partitioned_call0->GetOpDesc()->AddSubgraphName("subgraph0");
    partitioned_call0->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph0");
    root_graph->AddSubgraph("subgraph0", subgraph0);
  }

  AippOp aipp_op;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;
  GetLocalOmgContext().format = domi::DOMI_TENSOR_NCHW;
  domi::AippOpParams params;
  std::string conf_path = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_subgraph_conf.cfg";
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  params.set_aipp_mode(domi::AippOpParams::static_);
  EXPECT_EQ(aipp_op.Init(&params), SUCCESS);

  auto data_node0 = aipp_op.FindDataByIndex(root_graph, 0);
  aipp_op.InsertAippToGraph(root_graph, conf_path, 0);
  NodePtr is_aipp;
  for (auto node : root_graph->GetAllNodes()) {
    if (node->GetType() != AIPP) {
      continue;
    }
    is_aipp = node;
  }
  EXPECT_EQ(data_node0->GetName(), is_aipp->GetInNodes().at(0)->GetName());
}
}
