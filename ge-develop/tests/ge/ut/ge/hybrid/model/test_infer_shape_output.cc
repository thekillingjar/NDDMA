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
#include "graph/ge_tensor.h"
#include "hybrid/model/infer/node_shape_infer.h"
#include "hybrid/model/infer/shape_propagator.h"
#include "hybrid/model/infer/node_shape_propagator.h"
#include "hybrid/model/infer/graph_stage_cache.h"
#include "hybrid/model/infer/tensor_desc_observer.h"
#include "hybrid/model/infer/tensor_desc_holder.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "common/types.h"
#include "hybrid/model/node_item.h"
#include "graph/compute_graph.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
using  namespace hybrid;

class TestInferShapeOutput : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {

static NodePtr CreateNode() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Data", "Data");
  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);

  return graph->AddNode(op_desc);
}

struct FakeNodeItem : NodeItem {
  FakeNodeItem(const std::vector<int64_t> &shape, int stage = -1, NodePtr node = nullptr) : NodeItem(node) {
    op_desc_ = OP_CFG(DATA)
                  .InCnt(1)
                  .OutCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                  .Build("data1");
    op_desc = op_desc_.get();
    group = stage;
    is_input_shape_static_.push_back(false);
    is_dynamic = true;
  }

  GeShape OutputShape(){
    return op_desc->MutableOutputDesc(0)->GetShape();
  }

  GeShape InputShape(){
    return op_desc->MutableInputDesc(0)->GetShape();
  }

private:
  OpDescPtr op_desc_;
};

}

TEST_F(TestInferShapeOutput, tensor_desc_propagate_one_output) {
  GeTensorDesc output(GeShape({1, 2, 3, 4}), FORMAT_NCHW);
  GeTensorDesc input(GeShape({1, 2, 3}), FORMAT_ND);

  NodeShapePropagator node;
  node.CreatePropagator(TensorDescHolder(output), TensorDescObserver(TensorDescHolder(input)));
  ASSERT_EQ(SUCCESS, node.DoPropagate());

  ASSERT_EQ(input.GetShape().GetDims(), output.GetShape().GetDims());
  ASSERT_EQ(input.GetOriginShape().GetDims(), output.GetOriginShape().GetDims());
}

TEST_F(TestInferShapeOutput, tensor_desc_propagated_two_output) {
  GeTensorDesc output(GeShape({1, 2, 3, 4}), FORMAT_NCHW);
  GeTensorDesc input1(GeShape({1, 2, 3}), FORMAT_ND);
  GeTensorDesc input2(GeShape({1, 2, 3}), FORMAT_ND);

  NodeShapePropagator node;
  node.CreatePropagator(TensorDescHolder(output), TensorDescObserver(TensorDescHolder(input1)));
  node.CreatePropagator(TensorDescHolder(output), TensorDescObserver(TensorDescHolder(input2)));
  ASSERT_EQ(SUCCESS, node.DoPropagate());

  ASSERT_EQ(input1.GetShape().GetDims(), output.GetShape().GetDims());
  ASSERT_EQ(input2.GetShape().GetDims(), output.GetShape().GetDims());
}

TEST_F(TestInferShapeOutput, tensor_desc_propagated_two_output_and_input) {
  GeTensorDesc output1(GeShape({1, 2, 3, 4}), FORMAT_NCHW);
  GeTensorDesc output2(GeShape({1, 2}), FORMAT_NCHW);
  GeTensorDesc input1(GeShape({1, 2, 3}), FORMAT_ND);
  GeTensorDesc input2(GeShape({1, 2, 3}), FORMAT_ND);

  NodeShapePropagator node;
  node.CreatePropagator(TensorDescHolder(output1), TensorDescObserver(TensorDescHolder(input1)));
  node.CreatePropagator(TensorDescHolder(output2), TensorDescObserver(TensorDescHolder(input2)));
  ASSERT_EQ(SUCCESS, node.DoPropagate());

  ASSERT_EQ(input1.GetShape().GetDims(), output1.GetShape().GetDims());
  ASSERT_EQ(input2.GetShape().GetDims(), output2.GetShape().GetDims());
}

TEST_F(TestInferShapeOutput, test_stage_propagation_success) {

  FakeNodeItem output({1, 2, 3, 4}, 0, CreateNode());
  FakeNodeItem input({1, 2, 3}, 1, CreateNode());

  GraphStageCache stage_cache;
  stage_cache.CreatePropagator(output, 0, input, 0);
  ASSERT_EQ(SUCCESS, output.DoPropagate());
  ASSERT_EQ(SUCCESS, stage_cache.DoPropagate(1));

  ASSERT_EQ(output.OutputShape().GetDims(), input.InputShape().GetDims());
}

TEST_F(TestInferShapeOutput, test_stage_propagation_success_when_tensor_desc_update) {

  FakeNodeItem output({1, 2, 3, 4}, 0, CreateNode());
  FakeNodeItem input({1, 2, 3}, 1, CreateNode());

  GraphStageCache stage_cache;
  stage_cache.CreatePropagator(output, 0, input, 0);

  input.op_desc->UpdateInputDesc(0, GeTensorDesc());

  ASSERT_EQ(SUCCESS, output.DoPropagate());
  ASSERT_EQ(SUCCESS, stage_cache.DoPropagate(1));

  ASSERT_EQ(output.OutputShape().GetDims(), input.InputShape().GetDims());
}

TEST_F(TestInferShapeOutput, test_multi_shape_propagation_success) {

  FakeNodeItem output({1, 2, 3, 4}, 0, CreateNode());
  FakeNodeItem input1({1, 2, 3}, 1, CreateNode());
  FakeNodeItem input2({1, 2, 3}, 1, CreateNode());

  GraphStageCache stage_cache;
  stage_cache.CreatePropagator(output, 0, input1, 0);
  stage_cache.CreatePropagator(output, 0, input2, 0);

  ASSERT_EQ(SUCCESS, output.DoPropagate());
  ASSERT_EQ(SUCCESS, stage_cache.DoPropagate(1));

  ASSERT_EQ(output.OutputShape().GetDims(), input1.InputShape().GetDims());
  ASSERT_EQ(output.OutputShape().GetDims(), input2.InputShape().GetDims());
}

TEST_F(TestInferShapeOutput, test_multi_stage_propagation_success) {

  FakeNodeItem output({1, 2, 3, 4}, 0, CreateNode());
  FakeNodeItem input1({1, 2, 3}, 1, CreateNode());

  GraphStageCache stage_cache;
  stage_cache.CreatePropagator(output, 0, input1, 0);

  ASSERT_EQ(SUCCESS, output.DoPropagate());

  ASSERT_EQ(output.OutputShape().GetDims(), input1.InputShape().GetDims());
}

TEST_F(TestInferShapeOutput, test_stage_propagation_do_nothing_in_knowshape) {

  FakeNodeItem output({1, 2, 3, 4}, 0, CreateNode());
  FakeNodeItem input({1, 2, 3}, 1, CreateNode());

  GraphStageCache stage_cache;
  stage_cache.CreatePropagator(output, 0, input, 0);
  ASSERT_EQ(SUCCESS, stage_cache.DoPropagate(1));
  ASSERT_NE(output.OutputShape().GetDims(), input.InputShape().GetDims());
}

}
