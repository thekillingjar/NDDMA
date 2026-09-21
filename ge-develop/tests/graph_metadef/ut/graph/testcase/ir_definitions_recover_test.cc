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

#include "graph/op_desc.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/ir_definitions_recover.h"
#include "graph/utils/recover_ir_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/operator_reg.h"
#include "dlog_pub.h"

using namespace ge;

namespace gert {
class IrDefinitionsRecoverUT : public testing::Test {
 public:
  void SetUp() override {
    dlog_setlevel(0, 1, 0);
  }
  void TearDown() override {
    dlog_setlevel(0, 3, 0);
  }
};
REG_OP(DataUt)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(DataUt)

REG_OP(MatMulUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .REQUIRED_ATTR(loss_attr, Bool)
    .OP_END_FACTORY_REG(MatMulUt)

class MatMulUtFuture : public op::MatMulUt {
public:
  MatMulUtFuture(const std::string &name) : MatMulUt(name.c_str()) {}
  using Operator::DynamicInputRegister;
  using Operator::InputRegister;
  using Operator::OptionalInputRegister;

  using Operator::DynamicOutputRegister;
  using Operator::OutputRegister;

  using Operator::AttrRegister;
  using Operator::RequiredAttrWithTypeRegister;
};

REG_OP(ConcatV2DUt)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2DUt)

REG_OP(BNInferenceDUt)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF_16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF_16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF_16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(momentum, Float, 0.9f)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(use_global_stats, Bool, true)
    .ATTR(mode, Int, 1)
    .OP_END_FACTORY_REG(BNInferenceDUt)

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_inputs_not_match_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_ = op_desc_origin->GetIrAttrNames();
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs = op_desc_origin->GetIrInputs();
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs.empty());
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.empty());
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs[0].first = "fake";
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_TRUE(op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs[0].first == "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_inputs_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs.emplace_back(std::pair<std::string, IrInputType>("fake", kIrInputRequired));
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs[0].first,  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_attr_name_not_match_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.emplace_back("fake");
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_[0],  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_attr_name_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.emplace_back("fake");
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.back(),  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_empty_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_partial_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->AppendIrAttrName(op_desc_origin->GetIrAttrNames().at(0));
  auto &pair = op_desc_origin->GetIrInputs().at(0);
  op_desc->AppendIrInput(pair.first, pair.second);

  // recover success
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_same_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  for (const auto &attr : op_desc_origin->GetIrAttrNames()) {
    op_desc->AppendIrAttrName(attr);
  }
  for (const auto &pair : op_desc_origin->GetIrInputs()) {
    op_desc->AppendIrInput(pair.first, pair.second);
  }
  // recover success
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_frameworkop_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "FrameworkOp");
  AttrUtils::SetStr(op_desc, "original_type", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());

}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_op_loss_not_has_default_value) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_FALSE(ge::AttrUtils::HasAttr(op_desc, "loss_attr"));
  EXPECT_TRUE(ge::AttrUtils::HasAttr(op_desc, "transpose_x1"));
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_outputs_not_match_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_ = op_desc_origin->GetIrAttrNames();
  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs = op_desc_origin->GetIrOutputs();
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs.empty());
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.empty());
  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs[0].first = "fake";
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_TRUE(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs[0].first == "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_outputs_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs.emplace_back(std::pair<std::string, IrOutputType>("fake", kIrOutputRequired));
  auto ret = RecoverIrUtils::RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs[0].first,  "fake");
}

// TODO if all depended is replace, this 2 function will be deleted
TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_wrapper_empty_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

// TODO if all depended is replace, this 2 function will be deleted
TEST_F(IrDefinitionsRecoverUT, RecoverOpDescIrDefinition_wrapper_empty_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverOpDescIrDefinition(op_desc);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, CheckIrSpe_ir_input_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto ret = CheckIrSpec(op_desc);
  EXPECT_EQ(ret, false);
}

TEST_F(IrDefinitionsRecoverUT, CheckIrSpe_ir_input_dynamic_skip_check) {
  auto op_desc = std::make_shared<ge::OpDesc>("concatv2d", "ConcatV2DUt");
  ASSERT_NE(op_desc, nullptr);
  auto ret = CheckIrSpec(op_desc);
  EXPECT_EQ(ret, false);
}

TEST_F(IrDefinitionsRecoverUT, CheckIrSpe_ir_input_optional_skip_check) {
  auto op_desc = std::make_shared<ge::OpDesc>("BNInferenceDUt", "BNInferenceDUt");
  ASSERT_NE(op_desc, nullptr);
  auto ret = CheckIrSpec(op_desc);
  EXPECT_EQ(ret, false);
}

TEST_F(IrDefinitionsRecoverUT, CheckIrSpec_ir_attr_not_match_failed) {
  dlog_setlevel(0, 0, 0);
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs = op_desc_origin->GetIrOutputs();
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs = op_desc_origin->GetIrInputs();
  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.emplace_back("fake");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);

  op_desc->AddOutputDesc(out_desc);
  op_desc->AddInputDesc(0, out_desc);
  op_desc->AddInputDesc(1, out_desc);
  op_desc->AddInputDesc(2, out_desc);
  (void)AttrUtils::SetBool(op_desc, "transpose_x1", true);
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs.empty());
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.empty());
  auto ret = CheckIrSpec(op_desc);
  EXPECT_EQ(ret, false);
  dlog_setlevel(0, 3, 0);
}

// 前向兼容测试用例：属性相关
TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_forward_compat_attr_optional_success) {
  auto op_now = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_now = ge::OpDescUtils::GetOpDescFromOperator(op_now);

  // 模拟前向兼容场景：直构图IR版本 > 兼容图IR版本
  // 节点包含兼容图的所有属性，再加上一个额外的可选属性
  gert::MatMulUtFuture op_future("matmul_future");
  // 模拟可选输入未配置
  op_future.AttrRegister("new_optional_attr", AttrValue());
  auto op_desc_future = ge::OpDescUtils::GetOpDescFromOperator(op_future);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_NE(compute_graph->AddNode(op_desc_future), nullptr);

  // 额外属性未配置值（可选属性且未配置，应该被删除）
  auto ret = RecoverIrUtils::RecoverIrDefinitions(compute_graph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  // 验证额外属性已被删除
  EXPECT_EQ(op_desc_future->GetIrAttrNames().size(), op_desc_now->GetIrAttrNames().size());
  EXPECT_FALSE(ge::AttrUtils::HasAttr(op_desc_future, "new_optional_attr"));
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_forward_compat_attr_required_failed) {
  // 模拟前向兼容场景：直构图IR版本 > 兼容图IR版本
  // 节点包含兼容图的所有属性，再加上一个额外的必需属性
  gert::MatMulUtFuture op_future("matmul_future");
  op_future.RequiredAttrWithTypeRegister("new_required_attr", "Bool");
  auto op_desc_future = ge::OpDescUtils::GetOpDescFromOperator(op_future);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_NE(compute_graph->AddNode(op_desc_future), nullptr);

  // 额外属性是必需属性（不在兼容图IR中），应该失败
  auto ret = RecoverIrUtils::RecoverIrDefinitions(compute_graph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_forward_compat_attr_configured_failed) {
  // 模拟前向兼容场景：直构图IR版本 > 兼容图IR版本
  // 节点包含兼容图的所有属性，再加上一个额外的可选属性，但已配置非默认值
  gert::MatMulUtFuture op_future("matmul_future");
  op_future.AttrRegister("new_optional_attr", true);  // 配置非默认值
  auto op_desc_future = ge::OpDescUtils::GetOpDescFromOperator(op_future);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_NE(compute_graph->AddNode(op_desc_future), nullptr);

  // 额外属性已配置非默认值，应该失败
  auto ret = RecoverIrUtils::RecoverIrDefinitions(compute_graph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

// 前向兼容测试用例：输入相关
TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_forward_compat_input_optional_unconnected_success) {
  auto op_now = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_now = ge::OpDescUtils::GetOpDescFromOperator(op_now);

  // 模拟前向兼容场景：直构图IR版本 > 兼容图IR版本
  // 节点包含兼容图的所有输入，再加上一个额外的可选输入
  // 直接构造子类实例
  gert::MatMulUtFuture op_future("matmul_future");
  op_future.OptionalInputRegister("new_optional_input");
  auto op_desc_future = ge::OpDescUtils::GetOpDescFromOperator(op_future);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_NE(compute_graph->AddNode(op_desc_future), nullptr);

  // 额外输入未连边（可选输入且未连边，应该被删除）
  auto ret = RecoverIrUtils::RecoverIrDefinitions(compute_graph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  // 验证额外输入已被删除
  EXPECT_EQ(op_desc_future->GetIrInputs().size(), op_desc_now->GetIrInputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_forward_compat_input_required_failed) {
  // 模拟前向兼容场景：直构图IR版本 > 兼容图IR版本
  // 节点包含兼容图的所有输入，再加上一个额外的必需输入
  // 直接构造子类实例
  gert::MatMulUtFuture op_future("matmul_future");
  op_future.InputRegister("new_required_input");
  auto op_desc_future = ge::OpDescUtils::GetOpDescFromOperator(op_future);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_NE(compute_graph->AddNode(op_desc_future), nullptr);

  // 额外输入是必需输入（不在兼容图IR中），应该失败
  auto ret = RecoverIrUtils::RecoverIrDefinitions(compute_graph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_forward_compat_input_connected_failed) {
  auto data = gert::op::DataUt("data0");
  // 模拟前向兼容场景：直构图IR版本 > 兼容图IR版本
  // 节点包含兼容图的所有输入，再加上一个额外的可选输入，但已连边
  // 直接构造子类实例
  gert::MatMulUtFuture op_future("matmul_future");
  op_future.OptionalInputRegister("new_optional_input");
  op_future.SetInput("new_optional_input", data);
  auto graph = std::make_shared<ge::Graph>("graph_name");
  ASSERT_NE(graph, nullptr);
  graph->SetInputs({data});

  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  auto node = compute_graph->FindNode("matmul_future");
  ASSERT_NE(node, nullptr);
  ASSERT_NE(node->GetInDataAnchor(3), nullptr);
  ASSERT_NE(node->GetInDataAnchor(3)->GetPeerOutAnchor(), nullptr);
  // 额外输入已连边，应该失败
  auto ret = RecoverIrUtils::RecoverIrDefinitions(compute_graph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_forward_compat) {
  // 测试分支1: attr_diff > 0 && input_diff > 0 -> kForward
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  // 添加更多属性和输入
  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrAttrName("attr2");
  op_desc->AppendIrAttrName("attr3");
  op_desc->AppendIrInput("input1", kIrInputRequired);
  op_desc->AppendIrInput("input2", kIrInputOptional);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1"};                   // 兼容图只有1个属性
  ir_def.inputs = {{"input1", kIrInputRequired}};  // 兼容图只有1个输入

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kForward);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_backward_compat) {
  // 测试分支2: attr_diff < 0 && input_diff < 0 -> kBackward
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  // 添加少量属性和输入
  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrInput("input1", kIrInputRequired);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1", "attr2", "attr3"};                               // 兼容图有3个属性
  ir_def.inputs = {{"input1", kIrInputRequired}, {"input2", kIrInputOptional}};  // 兼容图有2个输入

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kBackward);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_failed_attr_forward_input_backward) {
  // 测试分支3: attr_diff > 0 && input_diff < 0 -> kFailed
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  // 属性多，输入少
  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrAttrName("attr2");
  op_desc->AppendIrInput("input1", kIrInputRequired);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1"};                                                 // 兼容图只有1个属性
  ir_def.inputs = {{"input1", kIrInputRequired}, {"input2", kIrInputOptional}};  // 兼容图有2个输入

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kFailed);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_failed_attr_backward_input_forward) {
  // 测试分支4: attr_diff < 0 && input_diff > 0 -> kFailed
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  // 属性少，输入多
  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrInput("input1", kIrInputRequired);
  op_desc->AppendIrInput("input2", kIrInputOptional);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1", "attr2"};          // 兼容图有2个属性
  ir_def.inputs = {{"input1", kIrInputRequired}};  // 兼容图只有1个输入

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kFailed);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_none) {
  // 测试分支5: attr_diff == 0 && input_diff == 0 -> kNone
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrInput("input1", kIrInputRequired);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1"};
  ir_def.inputs = {{"input1", kIrInputRequired}};

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kNone);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_forward_attr_only) {
  // 测试分支6: attr_diff > 0 && input_diff == 0 -> kForward
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrAttrName("attr2");
  op_desc->AppendIrInput("input1", kIrInputRequired);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1"};                   // 兼容图只有1个属性
  ir_def.inputs = {{"input1", kIrInputRequired}};  // 输入数量相同

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kForward);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_forward_input_only) {
  // 测试分支7: attr_diff == 0 && input_diff > 0 -> kForward
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrInput("input1", kIrInputRequired);
  op_desc->AppendIrInput("input2", kIrInputOptional);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1"};                   // 属性数量相同
  ir_def.inputs = {{"input1", kIrInputRequired}};  // 兼容图只有1个输入

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kForward);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_backward_attr_only) {
  // 测试分支8: attr_diff < 0 && input_diff == 0 -> kBackward
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrInput("input1", kIrInputRequired);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1", "attr2"};          // 兼容图有2个属性
  ir_def.inputs = {{"input1", kIrInputRequired}};  // 输入数量相同

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kBackward);
}

TEST_F(IrDefinitionsRecoverUT, DeriveCompatibilityStrategy_backward_input_only) {
  // 测试分支9: attr_diff == 0 && input_diff < 0 -> kBackward
  auto op_desc = std::make_shared<ge::OpDesc>("test", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrAttrName("attr1");
  op_desc->AppendIrInput("input1", kIrInputRequired);

  ge::RecoverIrUtils::IrDefinition ir_def;
  ir_def.attr_names = {"attr1"};                                                 // 属性数量相同
  ir_def.inputs = {{"input1", kIrInputRequired}, {"input2", kIrInputOptional}};  // 兼容图有2个输入

  auto strategy = RecoverIrUtils::DeriveCompatibilityStrategy(op_desc, ir_def);
  EXPECT_EQ(strategy, ge::CompatibilityStrategy::kBackward);
}
} // namespace gert
