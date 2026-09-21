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
#include "faker/fake_value.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "runtime/dev.h"
#include "stub/gert_runtime_stub.h"
#include "graph/debug/ge_attr_define.h"
#include "proto/insert_op.pb.h"
#include "faker/ge_model_builder.h"
#include "faker/model_data_faker.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/space_registry_faker.h"

using namespace ge;
namespace gert {
namespace {}  // namespace
class AippSystemTest : public bg::BgTest {
  void SetUp() override {
    gert::CreateVersionInfo();
  }

  void TearDown() override {
    gert::DestroyVersionInfo();
  }
};

/**
 * 用例描述：Aipp场景下，模型加载完后，获取aipp相关信息成功
 *
 * 预置条件：
 * 1. fake一个aipp算子的图，设置aipp data节点上相关属性
 *
 * 测试步骤：
 * 1. 按照预制条件构造Aipp编译图
 * 2. lowering、加载计算图
 * 3. 获取Aipp相关信息
 *
 * 预期结果：
 * 1. Aipp执行图加载成功
 * 2. Aipp相关信息获取成功，同时相关数据获取准确
 */
TEST_F(AippSystemTest, TestAippShapeV2SetAndGetSuccess) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();

  ge::NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", ge::GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_dynamic));
  aipp_attr.SetAttr("related_input_rank", ge::GeAttrValue::CreateFrom<int64_t>(0));
  aipp_attr.SetAttr("max_src_image_size", ge::GeAttrValue::CreateFrom<int64_t>(2048));
  aipp_attr.SetAttr("support_rotation", ge::GeAttrValue::CreateFrom<int64_t>(1));

  auto aippData1 = graph->FindNode("aippData1");
  auto op_desc = aippData1->GetOpDesc();
  const std::vector<int64_t> input_dims{-1, -1, -1, -1};
  ge::GeTensorDesc tensor_desc(ge::GeShape(input_dims), ge::FORMAT_NHWC, ge::DT_FLOAT);
  tensor_desc.SetShapeRange({{1, -1}, {1, -1}, {1, -1}, {1, -1}});
  op_desc->AddInputDesc(tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetOpKernelLibName("GeLocal");
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);
  ge::AttrUtils::SetNamedAttrs(op_desc, ge::ATTR_NAME_AIPP, aipp_attr);
  (void)ge::AttrUtils::GetNamedAttrs(op_desc, ge::ATTR_NAME_AIPP, aipp_attr);
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_DATA_AIPP_DATA_NAME_MAP, "aippData1");

  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_DYNAMIC_AIPP_INPUT_DIMS, input_dims);

  std::vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:100:3:1,2,8" };
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:100:3:1,2,8" };
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_AIPP_OUTPUTS, outputs);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(stream_executor, nullptr);

  ge::AippConfigInfo aipp_info;
  ge::InputAippType aipp_type;
  size_t aipp_index = 0;
  ge::Status ret;
  // Has been set
  ret = stream_executor->GetAippInfo(0, aipp_info);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(aipp_info.aipp_mode, domi::AippOpParams_AippMode_dynamic);
  ret = stream_executor->GetAippType(0, aipp_type, aipp_index);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(aipp_type, ge::DATA_WITH_DYNAMIC_AIPP);
  EXPECT_TRUE(aipp_index == 0);

  Shape tmp_shape({-1, -1, -1, -1});

  EXPECT_EQ(stream_executor->GetModelDesc().GetInputDesc(0)->GetAippShape(), tmp_shape);

  ge::OriginInputInfo orig_input_info;
  std::vector<ge::InputOutputDims> input_dims_aipp;
  std::vector<ge::InputOutputDims> output_dims_aipp;
  ASSERT_EQ(ge::SUCCESS, stream_executor->GetOriginAippInputInfo(0, orig_input_info));
  ASSERT_EQ(orig_input_info.format, ge::FORMAT_NCHW);
  ASSERT_EQ(orig_input_info.data_type, ge::DT_FLOAT);
  ASSERT_EQ(orig_input_info.dim_num, 3);

  ASSERT_EQ(ge::SUCCESS, stream_executor->GetAllAippInputOutputDims(0, input_dims_aipp, output_dims_aipp));
  ASSERT_EQ(input_dims_aipp.size(), 1);
  ASSERT_EQ(input_dims_aipp[0].name, "TensorName");
  ASSERT_EQ(input_dims_aipp[0].size, 100);
  ASSERT_EQ(input_dims_aipp[0].dim_num, 3);
  std::vector<int64_t> verify_vec{1,2,8};
  ASSERT_EQ(input_dims_aipp[0].dims, verify_vec);

  ASSERT_EQ(output_dims_aipp.size(), 1);
  ASSERT_EQ(output_dims_aipp[0].name, "TensorName");
  ASSERT_EQ(output_dims_aipp[0].size, 100);
  ASSERT_EQ(output_dims_aipp[0].dim_num, 3);
  ASSERT_EQ(output_dims_aipp[0].dims, verify_vec);
}

/**
 * 用例描述：Aipp场景下，模型加载完后，获取aipp相关信息失败场景，
 *
 * 预置条件：
 * 1. fake一个aipp算子的图，设置aipp data节点上相关属性
 *
 * 测试步骤：
 * 1. 按照预制条件构造Aipp编译图
 * 2. lowering、加载计算图
 * 3. 获取Aipp相关信息
 *
 * 预期结果：
 * 1. Aipp执行图加载成功
 * 2. Aipp相关信息获取失敗，range与dims维度不匹配导致失败
 */
TEST_F(AippSystemTest, TestAippShapeV2SetAndGetFailed_When_Shape_And_Range_Not_Match) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();
  auto aippData1 = graph->FindNode("aippData1");
  auto op_desc = aippData1->GetOpDesc();

  const std::vector<int64_t> input_dims{-1, -1, -1, -1};
  const std::vector<std::pair<int64_t, int64_t>> input_ranges{{1, -1}, {1, -1}, {1, -1}, {1, -1}};

  ge::GeTensorDesc tensor_desc(ge::GeShape(input_dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginShapeRange(input_ranges);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetOpKernelLibName("GeLocal");
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);

  // 非正常场景，dims与range维度不匹配
  const std::vector<int64_t> err_input_dims{-1, -1, -1};
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_DYNAMIC_AIPP_INPUT_DIMS, err_input_dims);
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_DIMS, err_input_dims);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_EQ(error_code, ge::SUCCESS);
  ASSERT_NE(stream_executor, nullptr);
  // 校验失败, aipp shape与range不相等
  ASSERT_NE(stream_executor->GetModelDesc().GetInputDesc(0)->GetAippShape().GetDimNum(),
            stream_executor->GetModelDesc().GetInputDesc(0)->GetOriginShapeRange().GetMax().GetDimNum());
  ASSERT_NE(stream_executor->GetModelDesc().GetInputDesc(0)->GetOriginShape().GetDimNum(),
            stream_executor->GetModelDesc().GetInputDesc(0)->GetOriginShapeRange().GetMax().GetDimNum());
  ASSERT_NE(stream_executor->GetModelDesc().GetInputDesc(0)->GetStorageShape().GetDimNum(),
            stream_executor->GetModelDesc().GetInputDesc(0)->GetOriginShapeRange().GetMax().GetDimNum());
}
}  // namespace gert
