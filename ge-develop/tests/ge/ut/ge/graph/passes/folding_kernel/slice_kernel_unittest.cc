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

#include "macro_utils/dt_public_scope.h"
#include "host_kernels/selection_ops/slice_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "gen_node.h"
#include "graph/op_desc.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
using namespace testing;

class UtestFoldingSliceKernel : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() { pass_ = new ::ge::ConstantFoldingPass(); }

  void destory() {
    delete pass_;
    pass_ = NULL;
  }

 protected:
  ::ge::ConstantFoldingPass *pass_;

  NodePtr initNode_int32(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_temp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);
    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_temp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[4];
    value[0] = 1;
    value[1] = 2;
    value[2] = 3;
    value[3] = 4;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));
    int32_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)&value1, sizeof(int32_t));
    int32_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)&value2, sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);

    OpDescUtils::SetWeights(node_temp, op_weights);
    NodePtr node = graph->AddNode(node_temp);
    return node;
  }

  NodePtr initNode_float(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_FLOAT);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape);
    float value[4];
    value[0] = 1.0;
    value[1] = 2.0;
    value[2] = 3.0;
    value[3] = 4.0;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(float));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value1, sizeof(int32_t));
    int32_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value2, sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_errtype(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_UNDEFINED);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_UNDEFINED);
    int32_t value[4];
    value[0] = 1;
    value[1] = 2;
    value[2] = 3;
    value[3] = 4;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value1, sizeof(int32_t));
    int32_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value2, sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    vector<int64_t> dims(4, 2);
    ge::GeShape op_shape(dims);
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2][2][2][2] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 16 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[4] = {1, 1, 1, 1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 4 * sizeof(int32_t));
    int32_t value2[4] = {1, 1, 1, 1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 4 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_3muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    vector<int64_t> dims(3, 2);
    ge::GeShape op_shape(dims);
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2][2][2] = {1, 2, 3, 4, 5, 6, 7, 8};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 8 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[3] = {1, 1, 1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 3 * sizeof(int32_t));
    int32_t value2[3] = {1, 1, 1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 3 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_2muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    vector<int64_t> dims(2, 2);
    ge::GeShape op_shape(dims);
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2][2] = {1, 2, 3, 4};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[2] = {1, 1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 2 * sizeof(int32_t));
    int32_t value2[2] = {1, -1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 2 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_1muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({2});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2] = {
        1,
        2,
    };

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 2 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[1] = {1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 1 * sizeof(int32_t));
    int32_t value2[1] = {1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 1 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_size_not_equal_fail(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);
    OpDescPtr child_opdef = std::make_shared<OpDesc>("child_opdef", "test");
    child_opdef->SetIsInputConst({false});

    vector<bool> is_input_const(3, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({3});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[3] = {1, 2, 3};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 3 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value1, 1 * sizeof(int32_t));
    int32_t value2[2] = {0, 1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 2 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    NodePtr child_node = graph->AddNode(child_opdef);
    return node;
  }

  NodePtr initNode_offset_int64(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_temp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_temp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    GeTensorDesc op_desc2(op_shape, FORMAT_NCHW, DT_INT64);
    int32_t value[4];
    value[0] = 1;
    value[1] = 2;
    value[2] = 3;
    value[3] = 4;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));
    int64_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc2, (uint8_t *)&value1, sizeof(int64_t));
    int64_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc2, (uint8_t *)&value2, sizeof(int64_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    
    OpDescUtils::SetWeights(node_temp, op_weights);
    NodePtr node = graph->AddNode(node_temp);
    return node;
  }
};

// test func：SliceKernel::Compute
// case：optimize op of int
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerIntSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_int32(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

// test func：SliceKernel::Compute
// case：optimize op of int, offset/size is int64
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOffsetInt64Success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_offset_int64(graph);
  auto inputs = OpDescUtils::GetWeights(*node);

  SliceKernel slice_kerel;
  vector<GeTensorPtr> v_output;
  auto ret = slice_kerel.Compute(node->GetOpDesc(), inputs, v_output);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(v_output.size(), 1);
  auto ret_shape = v_output.at(0)->GetTensorDesc().GetShape();
  EXPECT_EQ(ret_shape.GetDim(0), 2);
}

// test func：SliceKernel::Compute
// case：optimize op of float
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerFloatSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_float(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

// test func：SliceKernel::Compute
// case：optimize op of initNode_errtype
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerErrtypeSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_errtype(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

// test func：SliceKernel::Compute
// case：optimize op of initNode_muldims
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerIntMulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

// test func：SliceKernel::Compute
// case：optimize op of initNode_3muldims
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerInt3MulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_3muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

// test func：SliceKernel::Compute
// case：optimize op of initNode_2muldims
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerInt2MulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_2muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}
// test func：SliceKernel::Compute
// case：optimize op of initNode_1muldims
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerInt1MulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_1muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

// test func：SliceKernel::Compute
// case：optimize op of initNode_size_not_equal_fail
// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerSizeNotEqual) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_size_not_equal_fail(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestFoldingSliceKernel, InputCheckFailed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_offset_int64(graph);
  auto inputs = OpDescUtils::GetWeights(*node);

  vector<int64_t> dims_vec_0 = {1};
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  vector<ConstGeTensorPtr> input = {tensor_0};

  SliceKernel slice_kerel;
  vector<GeTensorPtr> v_output;
  auto ret = slice_kerel.Compute(node->GetOpDesc(), inputs, v_output);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(v_output.size(), 1);
  auto ret_shape = v_output.at(0)->GetTensorDesc().GetShape();
  EXPECT_EQ(ret_shape.GetDim(0), 2);
  ret = slice_kerel.Compute(nullptr, inputs, v_output);
  EXPECT_EQ(NOT_CHANGED, ret);
  ret = slice_kerel.Compute(node->GetOpDesc(), input, v_output);
  EXPECT_EQ(NOT_CHANGED, ret);
}
