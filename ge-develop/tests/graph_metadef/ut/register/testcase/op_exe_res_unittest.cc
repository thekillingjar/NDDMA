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
#include <vector>
#include <string>
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "register/graph_optimizer/graph_fusion/fusion_quant_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/debug/ge_log.h"
#define protected public
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#undef protected

using namespace std;
using namespace ge;
using namespace fe;
using namespace gert;
namespace {
class OpExeResTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

bool IsShapeEqual(gert::Shape shape1, ge::GeShape shape2) {
  if (shape1.GetDimNum() != shape2.GetDimNum()) {
    GELOGD("Dim[%zu] vs [%zu].", shape1.GetDimNum(), shape2.GetDimNum());
    return false;
  }
  for (size_t i = 0; i < shape1.GetDimNum(); ++i) {
    GELOGD("Dim val [%ld] vs [%ld].", shape1.GetDim(i), shape2.GetDim(i));
    if (shape1.GetDim(i) != shape2.GetDim(i)) {
      return false;
    }
  }
  return true;
}

TEST_F(OpExeResTest, OpResAPITest) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr x = std::make_shared<OpDesc>("x", "Data");
  OpDescPtr weight = std::make_shared<OpDesc>("weight", "Const");
  OpDescPtr atquant_scale = std::make_shared<OpDesc>("atquant_scale", "Const");
  OpDescPtr quant_scale = std::make_shared<OpDesc>("quant_scale", "Const");
  OpDescPtr quant_offset = std::make_shared<OpDesc>("quant_offset", "Const");
  OpDescPtr mm = std::make_shared<OpDesc>("mm", "WeightQuantBatchMatmulV2");
  OpDescPtr y = std::make_shared<OpDesc>("y", "NetOutput");

  // add descriptor
  ge::GeShape shape1({2,4,9,16});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);

  GeTensorDesc tensor_desc2(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_INT8);
  tensor_desc2.SetOriginShape(shape1);

  ge::GeShape shape2({1, 16});
  GeTensorDesc tensor_desc3(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc3.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc3.SetOriginShape(shape2);

  x->AddOutputDesc(tensor_desc1);
  weight->AddOutputDesc(tensor_desc2);
  atquant_scale->AddOutputDesc(tensor_desc1);
  quant_scale->AddOutputDesc(tensor_desc3);
  quant_offset->AddOutputDesc(tensor_desc3);

  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc2);
  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc3);
  mm->AddInputDesc(tensor_desc3);
  mm->AddOutputDesc(tensor_desc2);
  y->AddInputDesc(tensor_desc2);

  // create nodes
  NodePtr x_node = graph->AddNode(x);
  NodePtr weight_node = graph->AddNode(weight);
  NodePtr atquant_scale_node = graph->AddNode(atquant_scale);
  NodePtr quant_scale_node = graph->AddNode(quant_scale);
  NodePtr quant_offset_node = graph->AddNode(quant_offset);
  NodePtr mm_node = graph->AddNode(mm);
  NodePtr y_node = graph->AddNode(y);

  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(atquant_scale_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(quant_scale_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(3));

  ge::GraphUtils::AddEdge(mm_node->GetOutDataAnchor(0), y_node->GetInDataAnchor(0));

  mm->SetStreamId(12);
  mm->SetId(34);
  std::vector<int64_t> ori_work_sizes{22,33,44};
  mm->SetWorkspaceBytes(ori_work_sizes);

  ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*mm_node);
  EXPECT_NE(res_ptr_holder, nullptr);
  auto op_exe_res_ctx = reinterpret_cast<ExeResGenerationContext *>(res_ptr_holder->context_);
  auto op_check_ctx = reinterpret_cast<OpCheckContext *>(res_ptr_holder->context_);
  auto node_ptr = op_exe_res_ctx->MutableInputPointer<ge::Node>(0);
  EXPECT_NE(node_ptr, nullptr);
  auto stream_id = op_exe_res_ctx->GetStreamId();
  EXPECT_EQ(stream_id, 12);
  auto op_id = op_exe_res_ctx->GetOpId();
  EXPECT_EQ(op_id, 34);

  auto work_sizes = op_exe_res_ctx->GetWorkspaceBytes();
  EXPECT_EQ(work_sizes, ori_work_sizes);
  std::vector<int64_t> n_work_sizes{02,03,04};
  op_exe_res_ctx->SetWorkspaceBytes(n_work_sizes);
  work_sizes = op_exe_res_ctx->GetWorkspaceBytes();
  EXPECT_EQ(work_sizes, n_work_sizes);

  // test shape
  auto in_3_shape = op_exe_res_ctx->GetInputShape(3);
  EXPECT_NE(in_3_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(in_3_shape->GetOriginShape(), shape2), true);
  EXPECT_EQ(IsShapeEqual(in_3_shape->GetStorageShape(), shape2), true);

  auto in_5_shape = op_exe_res_ctx->GetInputShape(4);
  EXPECT_EQ(in_5_shape, nullptr);

  auto out_0_shape = op_exe_res_ctx->GetOutputShape(0);
  EXPECT_NE(out_0_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(out_0_shape->GetOriginShape(), shape1), true);
  EXPECT_EQ(IsShapeEqual(out_0_shape->GetStorageShape(), shape1), true);

  graph->SetGraphUnknownFlag(true);
  auto mode = op_exe_res_ctx->GetExecuteMode();
  EXPECT_EQ(mode, ExecuteMode::kDynamicExecute);

  ge::GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(0));
  ge::AscendString ir_name = "__input0";
  auto is_const = op_exe_res_ctx->IsConstInput(ir_name);
  EXPECT_EQ(is_const, false);

  is_const = op_check_ctx->IsConstInput(ir_name);
  EXPECT_EQ(is_const, false);

  ge::AscendString ir_name1 = "__input_invalid";
  is_const = op_exe_res_ctx->IsConstInput(ir_name1);
  EXPECT_EQ(is_const, false);

  is_const = op_check_ctx->IsConstInput(ir_name1);
  EXPECT_EQ(is_const, false);

  std::vector<StreamInfo> stream_info_vec;
  StreamInfo si_1;
  si_1.name = "tiling";
  si_1.reuse_key = "tiling_key";
  si_1.depend_value_input_indices = {1, 2};
  stream_info_vec.emplace_back(si_1);
  si_1.name = "tiling1";
  si_1.reuse_key = "tiling_key1";
  si_1.depend_value_input_indices = {0};
  stream_info_vec.emplace_back(si_1);
  auto ret = op_exe_res_ctx->SetAttachedStreamInfos(stream_info_vec);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<ge::GeAttrValue::NAMED_ATTRS> stream_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(mm_node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
    stream_info_attrs);
  EXPECT_EQ(stream_info_attrs.size(), stream_info_vec.size());
  stream_info_vec.clear();
  stream_info_vec = op_exe_res_ctx->GetAttachedStreamInfos();
  EXPECT_EQ(stream_info_vec.size(), stream_info_attrs.size());

  std::vector<SyncResInfo> sync_info_vec;
  SyncResInfo sync_info;
  sync_info.type = SyncResType::SYNC_RES_EVENT;
  sync_info.name = "tiling";
  sync_info.reuse_key = "tiling_key";
  sync_info_vec.emplace_back(sync_info);
  ret = op_exe_res_ctx->SetSyncResInfos(sync_info_vec);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<ge::GeAttrValue::NAMED_ATTRS> sync_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node_ptr->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
      sync_info_attrs);
  sync_info_vec.clear();
  sync_info_vec = op_exe_res_ctx->GetSyncResInfos();
  EXPECT_EQ(sync_info_vec.size(), sync_info_attrs.size());

  std::string test_key = "test_key";
  std::vector<std::string> test_list = {"test_key"};
  ret = op_exe_res_ctx->SetListStr(test_key, test_list);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OpExeResTest, OpResAPITest_1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr x = std::make_shared<OpDesc>("x", "Data");
  OpDescPtr weight = std::make_shared<OpDesc>("weight", "Const");
  OpDescPtr atquant_scale = std::make_shared<OpDesc>("atquant_scale", "Const");
  OpDescPtr quant_scale = std::make_shared<OpDesc>("quant_scale", "Const");
  OpDescPtr quant_offset = std::make_shared<OpDesc>("quant_offset", "Const");
  OpDescPtr mm = std::make_shared<OpDesc>("mm", "WeightQuantBatchMatmulV2");
  OpDescPtr y = std::make_shared<OpDesc>("y", "NetOutput");

  // add descriptor
  ge::GeShape shape0({2,4,9,16});
  ge::GeShape ori_shape0({1,16});
  GeTensorDesc tensor_desc1(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(ori_shape0);

  ge::GeShape shape1({2,4,9,16});
  GeTensorDesc tensor_desc2(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_INT8);
  tensor_desc2.SetOriginShape(shape1);

  ge::GeShape shape2({1, 16});
  GeTensorDesc tensor_desc3(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc3.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc3.SetOriginShape(shape2);

  x->AddOutputDesc(tensor_desc1);
  weight->AddOutputDesc(tensor_desc2);
  atquant_scale->AddOutputDesc(tensor_desc1);
  quant_scale->AddOutputDesc(tensor_desc3);
  quant_offset->AddOutputDesc(tensor_desc3);

  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc2);
  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc3);
  mm->AddOutputDesc(tensor_desc2);
  y->AddInputDesc(tensor_desc2);

  // create nodes
  NodePtr x_node = graph->AddNode(x);
  NodePtr weight_node = graph->AddNode(weight);
  NodePtr atquant_scale_node = graph->AddNode(atquant_scale);
  NodePtr quant_scale_node = graph->AddNode(quant_scale);
  NodePtr quant_offset_node = graph->AddNode(quant_offset);
  NodePtr mm_node = graph->AddNode(mm);
  NodePtr y_node = graph->AddNode(y);

  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(atquant_scale_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(quant_scale_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(mm_node->GetOutDataAnchor(0), y_node->GetInDataAnchor(0));

  ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpCheckContext(*mm_node);
  EXPECT_NE(res_ptr_holder, nullptr);
  auto op_check_ctx = reinterpret_cast<OpCheckContext *>(res_ptr_holder->context_);
  auto node_ptr = op_check_ctx->MutableInputPointer<ge::Node>(0);
  EXPECT_NE(node_ptr, nullptr);

  // test dtype&format
  auto in_0_desc = op_check_ctx->GetInputDesc(0);
  EXPECT_EQ(in_0_desc->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(in_0_desc->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(in_0_desc->GetOriginFormat(), ge::FORMAT_ND);

  auto in_1_desc = op_check_ctx->GetInputDesc(1);
  EXPECT_EQ(in_1_desc->GetDataType(), ge::DT_INT8);
  EXPECT_EQ(in_1_desc->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(in_1_desc->GetOriginFormat(), ge::FORMAT_NCHW);

  auto in_2_desc = op_check_ctx->GetInputDesc(2);
  EXPECT_EQ(in_2_desc->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(in_2_desc->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(in_2_desc->GetOriginFormat(), ge::FORMAT_ND);

  auto in_3_desc = op_check_ctx->GetInputDesc(3);
  EXPECT_EQ(in_3_desc->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(in_3_desc->GetStorageFormat(), ge::FORMAT_ND);
  EXPECT_EQ(in_3_desc->GetOriginFormat(), ge::FORMAT_ND);

  auto out_0_desc = op_check_ctx->GetOutputDesc(0);
  EXPECT_EQ(out_0_desc->GetDataType(), ge::DT_INT8);
  EXPECT_EQ(out_0_desc->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(out_0_desc->GetOriginFormat(), ge::FORMAT_NCHW);

  // test shape
  auto in_0_shape = op_check_ctx->GetInputShape(0);
  EXPECT_NE(in_0_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(in_0_shape->GetOriginShape(), ori_shape0), true);
  EXPECT_EQ(IsShapeEqual(in_0_shape->GetStorageShape(), shape0), true);

  auto in_1_shape = op_check_ctx->GetInputShape(1);
  EXPECT_NE(in_1_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(in_1_shape->GetOriginShape(), shape1), true);
  EXPECT_EQ(IsShapeEqual(in_1_shape->GetStorageShape(), shape1), true);

  auto in_2_shape = op_check_ctx->GetInputShape(2);
  EXPECT_NE(in_2_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(in_2_shape->GetOriginShape(), ori_shape0), true);
  EXPECT_EQ(IsShapeEqual(in_2_shape->GetStorageShape(), shape0), true);

  auto in_3_shape = op_check_ctx->GetInputShape(3);
  EXPECT_NE(in_3_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(in_3_shape->GetOriginShape(), shape2), true);
  EXPECT_EQ(IsShapeEqual(in_3_shape->GetStorageShape(), shape2), true);

  auto out_0_shape = op_check_ctx->GetOutputShape(0);
  EXPECT_NE(out_0_shape, nullptr);
  EXPECT_EQ(IsShapeEqual(out_0_shape->GetOriginShape(), shape1), true);
  EXPECT_EQ(IsShapeEqual(out_0_shape->GetStorageShape(), shape1), true);
}

TEST_F(OpExeResTest, SK_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mm = std::make_shared<OpDesc>("mm", "WeightQuantBatchMatmulV2");
  // add descriptor
  ge::GeShape shape0({2,4,9,16});
  ge::GeShape ori_shape0({1,16});
  GeTensorDesc tensor_desc1(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(ori_shape0);

  ge::GeShape shape1({2,4,9,16});
  GeTensorDesc tensor_desc2(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_INT8);
  tensor_desc2.SetOriginShape(shape1);

  ge::GeShape shape2({1, 16});
  GeTensorDesc tensor_desc3(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc3.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc3.SetOriginShape(shape2);
  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc2);
  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc3);
  mm->AddOutputDesc(tensor_desc2);
  NodePtr mm_node = graph->AddNode(mm);
  std::string sk_scope = "_ascendc_super_kernel_scope";
  std::string scope_val = "mla";
  ge::AttrUtils::SetStr(mm, sk_scope, scope_val);
  ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpCheckContext(*mm_node);
  EXPECT_NE(res_ptr_holder, nullptr);
  auto op_exe_res_ctx = reinterpret_cast<ExeResGenerationContext *>(res_ptr_holder->context_);
  EXPECT_NE(op_exe_res_ctx, nullptr);
  ge::AscendString str_ret;
  op_exe_res_ctx->GetStrAttrVal(sk_scope.c_str(), str_ret);
  EXPECT_EQ(*(str_ret.GetString()), *(scope_val.c_str()));
  std::string scope_val2 = "mlp";
  op_exe_res_ctx->SetStrAttrVal(sk_scope.c_str(), scope_val2.c_str());
  std::string str_tmp;
  ge::AttrUtils::GetStr(mm, sk_scope, str_tmp);
  EXPECT_EQ(str_tmp, scope_val2);
  std::string sk_id = "sub_id";
  ge::AttrUtils::SetInt(mm, sk_id, 1);
  int64_t int_val = 0;
  op_exe_res_ctx->GetIntAttrVal(sk_id.c_str(), int_val);
  EXPECT_EQ(int_val, 1);
  op_exe_res_ctx->SetIntAttrVal(sk_id.c_str(), 3);
  int64_t int_ret = 0;
  ge::AttrUtils::GetInt(mm, sk_id, int_ret);
  EXPECT_EQ(int_ret, 3);
}
}  // namespace
