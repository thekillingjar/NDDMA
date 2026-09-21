/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <common/share_graph.h>
#include <faker/fake_value.h>
#include <ge_running_env/fake_op.h>
#include <ge_running_env/ge_running_env_faker.h>
#include <gtest/gtest.h>
#include <stub/gert_runtime_stub.h>
#include <utils/mock_ops_kernel_builder.h>
#include <utils/synchronizer.h>

#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/graph.h"
#include "graph/tuning_utils.h"
#include "api/gelib/gelib.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_running_env/tensor_utils.h"
#include "common/ge_types.h"
#include "graph/load/model_manager/task_info/aicpu/kernel_ex_task_info.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace ge;
using namespace std;
namespace {
INFER_VALUE_RANGE_DEFAULT_REG(Add);
REG_OP(Shape)
    .OP_END_FACTORY_REG(Shape)
IMPL_INFER_VALUE_RANGE_FUNC(Shape, ShapeValueRangeFunc) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range;
  op_desc->MutableInputDesc(0)->GetShapeRange(in_shape_range);
  if (!in_shape_range.empty()) {
    output_tensor_desc->SetValueRange(in_shape_range);
  }
  return GRAPH_SUCCESS;
};

const auto ShapeInfer = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto td = op_desc->MutableOutputDesc(0);
  auto input_dims = op_desc->MutableInputDesc(0)->MutableShape().GetDims();
  if (input_dims == UNKNOWN_RANK) {
    td->SetShape(ge::GeShape(UNKNOWN_SHAPE));
    td->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
    td->SetShapeRange(std::vector<std::pair<int64_t, int64_t>>{{1, 8}});
  } else {
    int64_t size = static_cast<int64_t>(input_dims.size());
    std::vector<int64_t> size_v{size};
    td->SetShape(ge::GeShape(size_v));
    td->SetOriginShape(ge::GeShape(size_v));
  }
  uint32_t out_type = DT_INT32;
  (void)op.GetAttr("dtype", out_type);
  td->SetDataType((DataType)out_type);

  std::vector<std::pair<int64_t, int64_t>> inRange;
  op_desc->MutableInputDesc(0)->GetShapeRange(inRange);
  if (!inRange.empty()) {
    std::vector<int64_t> pre_op_range;
    pre_op_range.resize(2 * inRange.size());
    if (pre_op_range.size() >= INT_MAX) {
      return GRAPH_FAILED;
    }
    for (size_t i = 0; i < pre_op_range.size(); i = i + 2) {
      pre_op_range[i] = inRange[i / 2].first;
      pre_op_range[i + 1] = inRange[i / 2].second;
    }
    ge::AttrUtils::SetListInt(*td, "_pre_op_in_range", pre_op_range);
  }
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(Shape, ShapeInfer);

// Relu infer
REG_OP(Relu).OP_END_FACTORY_REG(Relu)

const auto ReluInfer = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto td = op_desc->MutableOutputDesc(0);
  td->SetShape(ge::GeShape({512}));
  td->SetOriginShape(ge::GeShape({512}));
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(Relu, ReluInfer);

// ConcatV2 infer
REG_OP(ConcatV2)
  .DYNAMIC_INPUT(x, TensorType::BasicType())
  .INPUT(concat_dim, TensorType::IndexNumberType())
  .OUTPUT(y, TensorType::BasicType())
  .ATTR(N, Int, 1)
  .OP_END_FACTORY_REG(ConcatV2)


const auto ConcatV2Infer = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  for (const auto &input_desc : op_desc->GetAllInputsDesc()) {
    auto input_dims = input_desc.GetShape().GetDims();
    if (input_dims.empty()) {
      return FAILED;
    }
  }
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(ConcatV2, ConcatV2Infer);

// Assign infer
REG_OP(Assign)
    .INPUT(resource, TensorType::ALL())
    .INPUT(value, TensorType::ALL())
    .OUTPUT(resource, TensorType::ALL())
    .OP_END_FACTORY_REG(Assign)

// OneInOneOutDynamicInfer function
const auto AssignInfer = [](Operator &op) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(1);
  auto &input_shape = input_desc->MutableShape();
  DataType input_dtype = input_desc->GetDataType();

  vector<int64_t> output_idx_list = {0};
  for (const int64_t& output_idx : output_idx_list) {
    auto output_desc = op_info->MutableOutputDesc(output_idx);
    output_desc->SetShape(input_shape);
    output_desc->SetDataType(input_dtype);
  }
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(Assign, AssignInfer);

/*
*              (unknow)data2  
*                      |  
*                   shape   
*                    | |   data3
* data1(unknow)      add    |
*         \         /   switchn  const
*           reshape         |    /
*              |       expanddims
*              \           /
*                netoutput
*
*/
Graph BuildValueRangeInferGraph() {
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 4, -1})
                     .InCnt(1)
                     .OutCnt(1)
                     .Build("data1");
    auto data2 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 4, -1})
                     .InCnt(1)
                     .OutCnt(1)
                     .Build("data2");
    auto add = OP_CFG(ADD)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {4})
                     .InCnt(1)
                     .OutCnt(1)
                     .Build("add");
    auto data3 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 2)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 4, 16, 16})
                     .InCnt(1)
                     .OutCnt(1)
                     .Build("data3");
    auto switchn = OP_CFG("SwitchN")
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 4, 16, 16})
                     .InCnt(1)
                     .OutCnt(1)
                     .Build("switchn");
    auto shape = OP_CFG(SHAPE)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("_force_infershape_when_running", true)
        .Build("shape");
    auto reshape = OP_CFG(RESHAPE)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("_force_infershape_when_running", true)
        .Build("reshape");
    std::vector<int64_t> const_shape = {1};
    auto data_tensor = GenerateTensor(const_shape);
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {1})
        .Weight(data_tensor)
        .InCnt(1)
        .OutCnt(1)
        .Build("const1");
    auto expanddim = OP_CFG(EXPANDDIMS)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("expanddim");
    CHAIN(NODE(data1)->EDGE(0,0)
      ->NODE(reshape)->EDGE(0,0)
      ->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE(data2)->EDGE(0,0)
      ->NODE(shape)->EDGE(0,0)
      ->NODE(add)->EDGE(0,1)
      ->NODE(reshape));
    CHAIN(NODE(shape)->EDGE(0,1)
      ->NODE(add));
    CHAIN(NODE(data3)->EDGE(0,0)
      ->NODE(switchn)->EDGE(0,0)
      ->NODE(expanddim)->EDGE(0,1)
      ->NODE("netoutput"));
    CHAIN(NODE(const1)->EDGE(0,1)
      ->NODE(expanddim));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data1_node = compute_graph->FindNode("data1");
  auto out_desc = data1_node->GetOpDesc()->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 100), make_pair(1, 240), make_pair(4, 4),
                                                          make_pair(192, 192)};
  out_desc->SetShapeRange(shape_range);
  auto data2_node = compute_graph->FindNode("data2");
  auto out_desc2 = data2_node->GetOpDesc()->MutableOutputDesc(0);
  out_desc2->SetShapeRange(shape_range);

  auto add_node = compute_graph->FindNode("add");
  auto out_desc_add = add_node->GetOpDesc()->MutableOutputDesc(0);
  out_desc_add->SetShapeRange(shape_range);

  auto shape_node = compute_graph->FindNode("shape");
  shape_node->GetOpDesc()->AddInferFunc(ShapeInfer);

  return graph;
}

/*
*              (unknow)data2
*                      |
*                   shape
*                    | |   data3
* data1(unknow)      add    |
*         \         /   switchn  const
*           reshape         |    /
*              |       expanddims
*              \           /
*                netoutput
*
*/
Graph BuildValueRangeInferGraph2() {
  DEF_GRAPH(g1) {
                  auto data1 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 4, -1})
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("data1");
                  auto data2 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, 4, -1})
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("data2");
                  auto add = OP_CFG(ADD)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {256})
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("add");
                  auto data3 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 2)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 4, 16, 16})
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("data3");
                  auto switchn = OP_CFG("SwitchN")
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 4, 16, 16})
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("switchn");
                  auto shape = OP_CFG(SHAPE)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                      .Attr("_force_infershape_when_running", true)
                      .Build("shape");
                  auto reshape = OP_CFG(RESHAPE)
                      .InCnt(2)
                      .OutCnt(1)
                      .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                      .Attr("_force_infershape_when_running", true)
                      .Build("reshape");
                  std::vector<int64_t> const_shape = {1};
                  auto data_tensor = GenerateTensor(const_shape);
                  auto const1 = OP_CFG(CONSTANT)
                      .TensorDesc(FORMAT_ND, DT_INT32, {1})
                      .Weight(data_tensor)
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("const1");
                  auto expanddim = OP_CFG(EXPANDDIMS)
                      .InCnt(2)
                      .OutCnt(1)
                      .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                      .Build("expanddim");
                  CHAIN(NODE(data1)->EDGE(0,0)
                            ->NODE(reshape)->EDGE(0,0)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE(data2)->EDGE(0,0)
                            ->NODE(shape)->EDGE(0,0)
                            ->NODE(add)->EDGE(0,1)
                            ->NODE(reshape));
                  CHAIN(NODE(shape)->EDGE(0,1)
                            ->NODE(add));
                  CHAIN(NODE(data3)->EDGE(0,0)
                            ->NODE(switchn)->EDGE(0,0)
                            ->NODE(expanddim)->EDGE(0,1)
                            ->NODE("netoutput"));
                  CHAIN(NODE(const1)->EDGE(0,1)
                            ->NODE(expanddim));
                };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data1_node = compute_graph->FindNode("data1");
  auto out_desc = data1_node->GetOpDesc()->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 100), make_pair(1, 240), make_pair(4, 4),
                                                          make_pair(192, 192)};
  out_desc->SetShapeRange(shape_range);
  auto data2_node = compute_graph->FindNode("data2");
  auto out_desc2 = data2_node->GetOpDesc()->MutableOutputDesc(0);
  out_desc2->SetShapeRange(shape_range);

  auto add_node = compute_graph->FindNode("add");
  auto out_desc_add = add_node->GetOpDesc()->MutableOutputDesc(0);
  out_desc_add->SetShapeRange(shape_range);

  auto shape_node = compute_graph->FindNode("shape");
  shape_node->GetOpDesc()->AddInferFunc(ShapeInfer);

  return graph;
}

/**
 *    sub_data1            Data1
 *        |                  |
 *     700Relu   ==>>  PartitionedCall
 *        |                  |
 *   sub_output1         NetOutput
 */
Graph BuildPartitionedCallGraph() {
  DEF_GRAPH(sub) {
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("sub_output1");
    auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).Build("hcom1");
    CHAIN(NODE("sub_relu", RELU)->NODE(hcom1)->NODE(netoutput));
  };

  DEF_GRAPH(root_graph) {
    CHAIN(NODE("data1", DATA)->NODE("partitionedcall", PARTITIONEDCALL, sub)->NODE("output1", NETOUTPUT));
  };
  sub.Layout();
  auto graph = ToGeGraph(root_graph);
  auto sub_graph = ToGeGraph(sub);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto sub_compute_graph = GraphUtilsEx::GetComputeGraph(sub_graph);
  auto partitionedcall = compute_graph->FindNode("partitionedcall");
  auto sub_relu = sub_compute_graph->FindNode("sub_relu");
  auto hcom1 = sub_compute_graph->FindNode("hcom1");
  auto sub_output1 = sub_compute_graph->FindNode("sub_output1");
  auto partition_output = partitionedcall->GetOpDesc()->MutableOutputDesc(0);
  partition_output->SetShape(GeShape({1,0,2}));
  auto sub_relu_output = sub_relu->GetOpDesc()->MutableOutputDesc(0);
  sub_relu_output->SetShape(GeShape({1,0,2}));
  auto hcom1_output = hcom1->GetOpDesc()->MutableOutputDesc(0);
  hcom1_output->SetShape(GeShape({1,0,2}));
  auto sub_output1_input = sub_output1->GetOpDesc()->MutableInputDesc(0);
  sub_output1_input->SetShape(GeShape({1,0,2}));
  auto sub_output1_output = sub_output1->GetOpDesc()->MutableOutputDesc(0);
  sub_output1_output->SetShape(GeShape({1,0,2}));
  AttrUtils::SetInt(sub_output1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  return graph;
}

/**
 *    sub_data1               Data1
 *       |                      |
 *     hcom1(empty)  ==>>  PartitionedCall
 *       |                      |
 *     hcom3                hcom2(empty)
 *       |                       |
 *     sub_output1          NetOutput
 */
Graph BuildPartitionedCall2EmptyOpGraph() {
  DEF_GRAPH(sub) {
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("sub_output1");
    auto hcom1 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("hcom1");
    auto hcom3 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("hcom3");
    CHAIN(NODE("sub_relu", RELU)->NODE(hcom1)->NODE(hcom3)->NODE(netoutput));
  };

  DEF_GRAPH(root_graph) {
    auto hcom2 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("hcom2");
    CHAIN(NODE("data1", DATA)->NODE("partitionedcall", PARTITIONEDCALL, sub)->NODE(hcom2)->NODE("output1", NETOUTPUT));
  };
  sub.Layout();
  auto graph = ToGeGraph(root_graph);
  auto sub_graph = ToGeGraph(sub);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto sub_compute_graph = GraphUtilsEx::GetComputeGraph(sub_graph);
  auto hcom2 = compute_graph->FindNode("hcom2");
  auto hcom1 = sub_compute_graph->FindNode("hcom1");

  auto hcom1_output = hcom1->GetOpDesc()->MutableOutputDesc(0);
  hcom1_output->SetShape(GeShape({1,0,2}));
  auto hcom2_output = hcom2->GetOpDesc()->MutableOutputDesc(0);
  hcom2_output->SetShape(GeShape({1,0,2}));
  return graph;
}
/**
 *     Data1
 *       |
 *     Relu1       Const1
 *         \      /
 *          Switch
 *          |     \
 *          |    Relu2   Const2
 *          |     /        |
 *          Merge        Relu3
 *         /     \      /
 *      Relu4     Concat
 *         \       /
 *         Netoutput
 */
Graph BuildSwitchDeadEliminationGraph() {
  int32_t weight[1] = {0};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));

  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_INT32, {1}).Weight(tensor).InCnt(1).OutCnt(1);

    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->EDGE(0, 0)->NODE("switch1", SWITCH)->EDGE(0, 0)->
          NODE("merge1", MERGE)->EDGE(0, 0)->NODE("concat", CONCATV2)->NODE("output", NETOUTPUT));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("switch1")->EDGE(1, 0)->NODE("relu2", RELU)->EDGE(0, 1)->
          NODE("merge1")->EDGE(1, 0)->NODE("relu4", RELU)->NODE("output"));
    CHAIN(NODE("const2", const1)->NODE("relu3", RELU)->EDGE(0, 1)->NODE("concat"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto relu1 = compute_graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFunc(ReluInfer);
  auto relu2 = compute_graph->FindNode("relu2");
  relu2->GetOpDesc()->AddInferFunc(ReluInfer);
  auto relu3 = compute_graph->FindNode("relu3");
  relu3->GetOpDesc()->AddInferFunc(ReluInfer);
  auto relu4 = compute_graph->FindNode("relu4");
  relu4->GetOpDesc()->AddInferFunc(ReluInfer);
  auto concat = compute_graph->FindNode("concat");
  concat->GetOpDesc()->AddInferFunc(ConcatV2Infer);

  return graph;
}

/**
 *                       Data1
 *                         |
 *     Variable          Relu1
 *    {ND, [512]}     {NHWC, [512]}
 *              \      /
 *         {ND, []}  {ND, [512]}
 *               Assign
 *              {ND, []}
 *                 |
 *             Netoutput
 */
Graph BuildAssignGraph() {
  DEF_GRAPH(g1) {
    auto relu1 = OP_CFG(RELU).TensorDesc(FORMAT_NHWC, DT_INT32, {512}).InCnt(1).OutCnt(1).Build("relu1");
    auto variable1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_ND, DT_INT32, {512}).InCnt(1).OutCnt(1).Build("var1");
    CHAIN(NODE(variable1)->EDGE(0, 0)->NODE("assign", ASSIGN)->NODE("output", NETOUTPUT));
    CHAIN(NODE("data1", DATA)->NODE(relu1)->EDGE(0, 1)->NODE("assign"));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto assign = compute_graph->FindNode("assign");
  assign->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({512}));
  assign->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(GeShape({512}));
  assign->GetOpDesc()->AddInferFunc(AssignInfer);
  auto relu1 = compute_graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFunc(ReluInfer);

  return graph;
}

Graph BuildForSplitshapeNGraph() {
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA).InCnt(0).OutCnt(1).Build("data1");
    auto data2 = OP_CFG(DATA).InCnt(0).OutCnt(1).Build("data2");
    auto data3 = OP_CFG(DATA).InCnt(0).OutCnt(1).Build("data3");
    auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {8, 4, 16, 16}).Build("relu1");
    auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu2");
    auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu3");
    auto shapeN = OP_CFG(SHAPEN).InCnt(3).OutCnt(3).TensorDesc(FORMAT_NCHW, DT_INT32, {}).Build("shapeN");
    auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu4");
    auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu5");
    auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu6");
    CHAIN(NODE("data1", DATA)
      ->NODE(relu1)->EDGE(0,0)
      ->NODE(shapeN)->EDGE(0,0)
      ->NODE(relu4)->EDGE(0,0)
      ->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0,0)
      ->NODE(relu2)->EDGE(0,1)
      ->NODE(shapeN)->EDGE(1,0)
      ->NODE(relu5)->EDGE(0,0)
      ->NODE("netoutput2", NETOUTPUT));
    CHAIN(NODE("data3", DATA)->EDGE(0,0)
      ->NODE(relu3)->EDGE(0,2)
      ->NODE(shapeN)->EDGE(2,0)
      ->NODE(relu6)->EDGE(0,0)
      ->NODE("netoutput3", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data2_node = compute_graph->FindNode("data1");
  auto input_desc_data1 = data2_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_data1->SetShape(GeShape({1, 1, 2, 3}));

  auto data3_node = compute_graph->FindNode("relu1");
  auto input_desc_relu1 = data3_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc_relu1 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_relu1->SetShape(GeShape({1, 1, 2, 3}));
  output_desc_relu1->SetShape(GeShape({1, 1, 2, 3}));

  auto data4_node = compute_graph->FindNode("relu4");
  auto input_desc_relu4 = data4_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc_relu4 = data4_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_relu4->SetShape(GeShape({1, 1, 2, 3}));
  output_desc_relu4->SetShape(GeShape({1, 1, 2, 3}));

  auto data5_node = compute_graph->FindNode("netoutput1");
  auto input_desc_netoutput1 = data5_node->GetOpDesc()->MutableInputDesc(0);
  input_desc_netoutput1->SetShape(GeShape({1, 1, 2, 3}));

  auto data6_node = compute_graph->FindNode("data2");
  auto input_desc_data2 = data6_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_data2->SetShape(GeShape({-1, -1, 4, -1}));

  auto data7_node = compute_graph->FindNode("relu2");
  auto input_desc_relu2 = data7_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc_relu2 = data7_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_relu2->SetShape(GeShape({-1, -1, 4, -1}));
  output_desc_relu2->SetShape(GeShape({-1, -1, 4, -1}));

  auto data8_node = compute_graph->FindNode("relu5");
  auto input_desc_relu5 = data8_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc_relu5 = data8_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_relu5->SetShape(GeShape({-1, -1, 4, -1}));
  output_desc_relu5->SetShape(GeShape({-1, -1, 4, -1}));

  auto data9_node = compute_graph->FindNode("netoutput2");
  auto input_desc_netoutput2 = data9_node->GetOpDesc()->MutableInputDesc(0);
  input_desc_netoutput2->SetShape(GeShape({-1, -1, 4, -1}));

  auto data10_node = compute_graph->FindNode("data3");
  auto output_desc_data3 = data10_node->GetOpDesc()->MutableOutputDesc(0);
  output_desc_data3->SetShape(GeShape({-1, -1, 4, -1}));

  auto data11_node = compute_graph->FindNode("relu3");
  auto input_desc_relu3 = data11_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc_relu3 = data11_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_relu3->SetShape(GeShape({-1, -1, 4, -1}));
  output_desc_relu3->SetShape(GeShape({-1, -1, 4, -1}));

  auto data12_node = compute_graph->FindNode("relu6");
  auto input_desc_relu6 = data12_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc_relu6 = data12_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc_relu6->SetShape(GeShape({-1, -1, 4, -1}));
  output_desc_relu6->SetShape(GeShape({-1, -1, 4, -1}));

  auto data13_node = compute_graph->FindNode("netoutput3");
  auto input_desc_netoutput3 = data13_node->GetOpDesc()->MutableInputDesc(0);
  input_desc_netoutput3->SetShape(GeShape({-1, -1, 4, -1}));

  auto data1_node = compute_graph->FindNode("shapeN");
  auto input_desc0 = data1_node->GetOpDesc()->MutableInputDesc(0);
  auto output_desc0 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  input_desc0->SetShape(GeShape({8, 4, 16, 16}));
  output_desc0->SetShape(GeShape({8, 4, 16, 16}));

  auto input_desc1 = data1_node->GetOpDesc()->MutableInputDesc(1);
  auto output_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(1);
  input_desc1->SetShape(GeShape({-1, -1, 4, -1}));
  output_desc1->SetShape(GeShape({-1, -1, 4, -1}));

  auto input_desc2 = data1_node->GetOpDesc()->MutableInputDesc(2);
  auto output_desc2 = data1_node->GetOpDesc()->MutableOutputDesc(2);
  input_desc2->SetShape(GeShape({-1, -1, 4, -1}));
  output_desc2->SetShape(GeShape({-1, -1, 4, -1}));

  return graph;
}
}  // namespace

class InferAndFoldingTest : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {}
};

/*
* 该用例通过shape、add算子的value range推导，最终得出shape的输出value range为较精准的范围
*/
TEST_F(InferAndFoldingTest, test_infer_value_range) {
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueRangeFunc);
  INFER_VALUE_RANGE_DEFAULT_REG(ADD);
  
  // build graph
  Graph graph = BuildValueRangeInferGraph();
  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  // new session & add graph
  map<AscendString, AscendString> options;
  options[AscendString(MEMORY_OPTIMIZATION_POLICY.c_str())] = AscendString(kMemoryPriority.c_str());
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  session.BuildGraph(graph_id, inputs); // we only care about compile stage

  CHECK_GRAPH(OptimizeStage1_2) {
    // check shape node output desc has value range
    auto shape_node = graph->FindNode("shape");
    auto shape_output_desc = shape_node->GetOpDesc()->GetOutputDescPtr(0);
    std::vector<std::pair<int64_t, int64_t>> expect_value_range = {make_pair(1, 100), make_pair(1, 240), make_pair(4, 4),
                                                          make_pair(192, 192)};
    std::vector<std::pair<int64_t, int64_t>> value_range;
    shape_output_desc->GetValueRange(value_range);

    for (size_t i =0; i < value_range.size(); ++i) {
      ASSERT_EQ(value_range[i].first, expect_value_range[i].first);
      ASSERT_EQ(value_range[i].second, expect_value_range[i].second);
    }

    // check expandims has been removed
    auto expanddim = graph->FindNode("expanddim");
    ASSERT_EQ(expanddim, nullptr);

    // add node value range infer
    auto add_node = graph->FindNode("add");
    auto add_output_desc = add_node->GetOpDesc()->GetOutputDescPtr(0);
    std::vector<std::pair<int64_t, int64_t>> add_value_range;
    add_output_desc->GetValueRange(add_value_range);
    EXPECT_FALSE(add_value_range.empty());
  };
}

/*
* 该用例通过shape、add算子的value range推导，最终得出shape的输出value range为较精准的范围
*/
TEST_F(InferAndFoldingTest, test_skip_infer_value_range_when_output_shape_size_too_large) {
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueRangeFunc);
  INFER_VALUE_RANGE_DEFAULT_REG(ADD);

  // build graph
  Graph graph = BuildValueRangeInferGraph2();
  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  // new session & add graph
  map<AscendString, AscendString> options;
  options[AscendString(MEMORY_OPTIMIZATION_POLICY.c_str())] = AscendString(kMemoryPriority.c_str());
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  session.BuildGraph(graph_id, inputs); // we only care about compile stage

  CHECK_GRAPH(OptimizeStage1_2) {
    // check shape node output desc has value range
    auto shape_node = graph->FindNode("shape");
    auto shape_output_desc = shape_node->GetOpDesc()->GetOutputDescPtr(0);
    std::vector<std::pair<int64_t, int64_t>> expect_value_range = {make_pair(1, 100), make_pair(1, 240), make_pair(4, 4),
                                                                   make_pair(192, 192)};
    std::vector<std::pair<int64_t, int64_t>> value_range;
    shape_output_desc->GetValueRange(value_range);

    for (size_t i =0; i < value_range.size(); ++i) {
      ASSERT_EQ(value_range[i].first, expect_value_range[i].first);
      ASSERT_EQ(value_range[i].second, expect_value_range[i].second);
    }
    // add node skip value range infer
    auto add_node = graph->FindNode("add");
    auto add_output_desc = add_node->GetOpDesc()->GetOutputDescPtr(0);
    std::vector<std::pair<int64_t, int64_t>> add_value_range;
    add_output_desc->GetValueRange(add_value_range);
    EXPECT_TRUE(add_value_range.empty());
  };
}

// 若在potential const生效过程中，节点的父节点已经被删除，则跳过优化
TEST_F(InferAndFoldingTest, test_replace_empty_tensor_with_partitioncall) {
  // build graph
  Graph graph = BuildPartitionedCallGraph();
  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  session.BuildGraph(graph_id, inputs); // we only care about compile stage

  CHECK_GRAPH(OptimizeStage1_2) {
    // check partitionedcall has been removed
    auto partitionedcall = graph->FindNode("partitionedcall");
    ASSERT_EQ(partitionedcall, nullptr);
  };
}

// 若在potential const生效后，节点的父节点已经被删除，则跳过repass
// 如构图所示，hcom1在potential const生效时被加入repass
// 而partitioncall在hcom2常量折叠的时候被删除，并加入repass
// 那么在找下轮repass node的时候，通过hcom1找父节点时，应返回null并跳过repass
TEST_F(InferAndFoldingTest, test_replace_empty_tensor_both_in_subgraph_and_rootgraph) {
  // build graph
  Graph graph = BuildPartitionedCall2EmptyOpGraph();
  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  session.BuildGraph(graph_id, inputs); // we only care about compile stage

  CHECK_GRAPH(OptimizeStage1_2) {
    // check partitionedcall has been removed
    auto partitionedcall = graph->FindNode("partitionedcall");
    ASSERT_EQ(partitionedcall, nullptr);
  };
}

// after SwitchDeadBranchElimination and MergePass, switch's input node should do infershape again before its output
// nodes, so that these can get input shape in infershape process.
TEST_F(InferAndFoldingTest, test_switch_merge_remove_and_infershape) {
  // build graph
  Graph graph = BuildSwitchDeadEliminationGraph();
  DUMP_GRAPH_WHEN("AfterInfershape");
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(graph_id, inputs); // we only care about compile stage
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterInfershape) {
    auto switch1 = graph->FindNode("switch1");
    EXPECT_EQ(switch1, nullptr);
    auto merge1 = graph->FindNode("merge1");
    EXPECT_EQ(merge1, nullptr);

    auto concat = graph->FindNode("concat");
    EXPECT_NE(concat, nullptr);
    auto input_shape = concat->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();
    EXPECT_EQ(input_shape.size(), 1);
  };
}

TEST_F(InferAndFoldingTest, test_split_shape_n) {
    // build graph
  Graph graph = BuildForSplitshapeNGraph();
  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  session.BuildGraph(graph_id, inputs); // we only care about compile stage
  CHECK_GRAPH(OptimizeStage1_2) {
    std::vector<int32_t> known_opdesc_index;
    std::vector<int32_t> unknown_opdesc_index;
    std::vector<GeTensorDesc> known_input_opdesc;
    std::vector<GeTensorDesc> known_output_opdesc;
    std::vector<GeTensorDesc> unknown_input_opdesc;
    std::vector<GeTensorDesc> unknown_output_opdesc;
    int32_t n = 0;
    for (auto &node : graph->GetAllNodes()) {
      auto op_desc = node->GetOpDesc();
      if(op_desc->GetType() == SHAPEN) {
        n++;
        auto name = op_desc->GetName();
        for (size_t i =0; i < op_desc->GetAllOutputsDescSize(); i++) {
          GeShape out_shape = op_desc->GetOutputDesc(i).GetShape();
          if (out_shape.IsUnknownShape()) {
            unknown_opdesc_index.emplace_back(static_cast<int32_t> (i));
            unknown_input_opdesc.emplace_back(op_desc->GetOutputDesc(i));
            unknown_output_opdesc.emplace_back(op_desc->GetInputDesc(i));
          } else {
            known_opdesc_index.emplace_back(static_cast<int32_t> (i));
            known_input_opdesc.emplace_back(op_desc->GetOutputDesc(i));
            known_output_opdesc.emplace_back(op_desc->GetInputDesc(i));
          }
        }
      } else {
        continue;
      }
    }
    ASSERT_EQ(n, 1);
    ASSERT_EQ(unknown_output_opdesc.size(), 2);
    ASSERT_EQ(known_output_opdesc.size(), 0);
  };
}

TEST_F(InferAndFoldingTest, test_assign_infershape) {
  // build graph
  Graph graph = BuildAssignGraph();
  DUMP_GRAPH_WHEN("AfterInfershape");
  // new session & add graph
  map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(graph_id, inputs); // we only care about compile stage
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterInfershape) {
    auto assign = graph->FindNode("assign");
    EXPECT_NE(assign, nullptr);
    auto input0 = assign->GetOpDesc()->MutableInputDesc(0);
    auto input1 = assign->GetOpDesc()->MutableInputDesc(1);
    EXPECT_EQ(input0->GetFormat(), input1->GetFormat());
    EXPECT_EQ(input0->GetShape(), input1->GetShape());
  };
}

ComputeGraphPtr MakeFunctionGraph2(const std::string &func_node_name, const std::string &func_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE(func_node_name, func_node_type)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE(func_node_name));
    CHAIN(NODE("_arg_2", DATA)->NODE(func_node_name));
  };
  return ToComputeGraph(g1);
};

ComputeGraphPtr MakeSubGraph(const std::string &prefix) {
  DEF_GRAPH(g2, prefix.c_str()) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
    auto netoutput = OP_CFG(NETOUTPUT)
                         .InCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16})
                         .OutCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16})
                         .Build("netoutput");
    auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU))
                              .InCnt(1)
                              .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16})
                              .OutCnt(1)
                              .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16});
    CHAIN(NODE(prefix  + "_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Relu", relu_0))
        ->NODE(netoutput);
    CHAIN(NODE(prefix + "_arg_1", data_1)->EDGE(0, 1)->NODE(prefix + "Conv2D", conv_0));
  };
  return ToComputeGraph(g2);
}

ComputeGraphPtr MakeSubGraph1(const std::string &prefix) {
  DEF_GRAPH(g2, prefix.c_str()) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
    auto netoutput = OP_CFG(NETOUTPUT)
                         .InCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_INT32, {-1, 3})
                         .OutCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_INT32, {-1, 3})
                         .Build("netoutput");
    auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU))
                              .InCnt(1)
                              .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3})
                              .OutCnt(1)
                              .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3});
    CHAIN(NODE(prefix  + "_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Relu", relu_0))
        ->NODE(netoutput);
    CHAIN(NODE(prefix + "_arg_1", data_1)->EDGE(0, 1)->NODE(prefix + "Conv2D", conv_0));
  };
  return ToComputeGraph(g2);
}


void AddPartitionedCall(const ComputeGraphPtr &graph, const std::string &call_name, const ComputeGraphPtr &subgraph) {
  const auto &call_node = graph->FindNode(call_name);
  if (call_node == nullptr) {
    return;
  }
  call_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);

  size_t index = call_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  call_node->GetOpDesc()->AddSubgraphName(subgraph->GetName());
  call_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph->GetName());

  subgraph->SetParentNode(call_node);
  subgraph->SetParentGraph(graph);
  GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph);
}

// 该用例验证子图中的ref_output_tensor在unknown_rank场景是否更新正确
TEST_F(InferAndFoldingTest, test_update_output_from_subgraphs_get_unknown_rank) {
  // build graph
  std::string func_node_name = "Case_0";
  const auto &root_graph = MakeFunctionGraph2(func_node_name, CASE);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_1 = MakeSubGraph1("sub_graph_1/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph_1);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  DUMP_GRAPH_WHEN("AfterInfershape");

  // new session & add graph
  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterInfershape) {
    auto sub = graph->GetSubgraph("sub_graph_1/");
    EXPECT_NE(sub, nullptr);
    auto case_node = sub->FindNode("Case_0_sub_graph_1//sub_graph_1/Relu");
    EXPECT_NE(case_node, nullptr);
    auto case_tensor = case_node->GetOpDesc()->MutableInputDesc(0);
    std::vector<std::pair<int64_t, int64_t>> ge_tensor_desc1_ptr_shape_range;
    EXPECT_EQ(case_tensor->GetShapeRange(ge_tensor_desc1_ptr_shape_range), GRAPH_SUCCESS);
    EXPECT_EQ(ge_tensor_desc1_ptr_shape_range.empty(), true);
  };
}
/**
 * 这是一个编译+执行的端到端用例
 * 测试背景： 动态静态的hybrid model，若静态子图输出边界的squeeze被优化删除掉（这是一类gelocal算子，只变shape不该内存，在静态模型中通常采取删除掉的方式替代下沉执行），
 *          静态子图中shape产生不连续，静态子图的输出shape应该从子图中netoutput输入上获取。
 *          而不是获取netoutput直连的上游节点的输出shape。否则获取到的shape是不正确的
 *
   测试目标： 构造如上背景中的图，从编译到rt2执行，测试执行中infershape获取的shape是否正确。
   测试步骤： 1. 打桩（算子infershape桩、引擎桩、gentask桩、runtime stub）。todo,这里桩机制要继续优化，当前使用过于复杂
            2. 构图
                root_graph

                ┌───────┐  (0,1)   ┌────────┐  (0,0)   ┌───────────┐  (0,0)   ┌───────────┐
                │ input │ ───────> │   if   │ ───────> │ relu_main │ ───────> │ NetOutput │
                └───────┘          └────────┘          └───────────┘          └───────────┘
                [2,3,1,3]            ∧   [-2]          [-2]
                                     │ (0,0)
                                     │
                                   ┌────────┐
                                   │  pred  │
                                   └────────┘

                then_sub_graph

                ┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────┐  (0,0)   ┌─────────┐  (0,0)   ┌───────────┐
                │ data │ ───────> │ relu │ ───────> │ relu1 │ ───────> │ squeeze │ ───────> │ NetOutput │
                └──────┘          └──────┘          └───────┘          └─────────┘          └───────────┘
                [2,3,1,3]         [2,3,1,3]         [2,3,1,3]          [2,3,3]              [2,3,3]

                else_sub_graph

                ┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────┐
                │ data │ ───────> │ relu │ ───────> │ NetOutput │
                └──────┘          └──────┘          └───────────┘
                [2,3,1,3]         [2,3,1,3]         [2,3,1,3]
          3. 校验
             （1）relu_main算子的infershape被2次调用，编译1次、执行1次
             （2）在relu_main算子的infershape函数中设置校验点。预期输入shape是[2,3,3]
   遗留问题：RunGraphWithStreamAsync 因桩不完备，执行后期有报错。但不影响校验点，暂时不处理，也不对接口返回做校验
 */
TEST_F(InferAndFoldingTest, test_If_InferShape_change_rank_in_branch) {
  // engine & infershape stub
  size_t relu_main_call_infer_count = 0;
  const auto SingleIOForwardInfer = [&relu_main_call_infer_count](Operator &op) {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    auto in_td = op_desc->GetInputDescPtr(0);
    if (op_desc->GetName() == "relu_main") {
      relu_main_call_infer_count++;
      // runtime call infer func, input shape should be [2,3,3]
      if (relu_main_call_infer_count == 2) {
        EXPECT_EQ(in_td->GetShape().GetDimNum(), 3);
        EXPECT_EQ(in_td->GetShape().GetDim(0), 2);
        EXPECT_EQ(in_td->GetShape().GetDim(1), 3);
        EXPECT_EQ(in_td->GetShape().GetDim(2), 3);
      }
    }

    auto td = op_desc->MutableOutputDesc(0);
    td->SetShape(in_td->GetShape());
    td->SetOriginShape(in_td->GetOriginShape());
    td->SetDataType(in_td->GetDataType());
    td->SetOriginDataType(in_td->GetOriginDataType());
    return GRAPH_SUCCESS;
  };
  const auto SqueezeInfer = [](Operator &op) {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    auto in_td = op_desc->GetInputDescPtr(0);
    auto td = op_desc->MutableOutputDesc(0);
    std::vector<int64_t> output_shape;
    for(const auto dim : in_td->GetShape().GetDims()) {
      if (dim != 1) {
        output_shape.emplace_back(dim);
      }
    }
    td->SetShape(GeShape(output_shape));
    td->SetOriginShape(GeShape(output_shape));
    td->SetDataType(in_td->GetDataType());
    td->SetOriginDataType(in_td->GetOriginDataType());
    return GRAPH_SUCCESS;
  };
  GeRunningEnvFaker().Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
      .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(SQUEEZE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(SqueezeInfer))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(SingleIOForwardInfer));

   // gen task stub
   auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
     auto op_desc = node.GetOpDesc();
     op_desc->SetOpKernelLibName("AIcoreEngine");
     ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
     ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
     const char kernel_bin[] = "kernel_bin";
     vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
     ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
     op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
     size_t arg_size = 100;
     std::vector<uint8_t> args(arg_size, 0);
     domi::TaskDef task_def;
     task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
     auto kernel_info = task_def.mutable_kernel();
     kernel_info->set_args(args.data(), args.size());
     kernel_info->set_args_size(arg_size);
     kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
     kernel_info->set_kernel_name(node.GetName());
     kernel_info->set_block_dim(1);
     uint16_t args_offset[2] = {0};
     kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
     kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

     tasks.emplace_back(task_def);
     return SUCCESS;
   };
  MockForGenerateTask("AIcoreEngine", aicore_func);
  // build graph
  auto compute_graph = gert::ShareGraph::IfGraphRankChangedOneBranch();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // build input tensor
  TensorDesc input_tensor_desc(Shape({2,3,1,3}), FORMAT_ND, DT_FLOAT);
  input_tensor_desc.SetPlacement(kPlacementDevice);
  Tensor input(input_tensor_desc);
  std::vector<int64_t> scaler_shape= {};
  Tensor pred{TensorDesc(Shape(scaler_shape), FORMAT_ND, DT_INT32)};
  Tensor output(input_tensor_desc);
  std::vector<Tensor> inputs = {pred, input};
  std::vector<Tensor> outputs = {output};

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    // new session & add graph
    map<AscendString, AscendString> options;
    options[OPTION_STATIC_MODEL_OPS_LOWER_LIMIT] = "1";
    Session session(options);
    uint32_t graph_id = 1;
    auto ret = session.AddGraph(graph_id, graph, options);
    EXPECT_EQ(ret, SUCCESS);

    (void)session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs);
    EXPECT_EQ(relu_main_call_infer_count, 2);
  }
  // teardown
  TearDownForGenerateTask("AIcoreEngine");
  runtime_stub.Clear();
  GeRunningEnvFaker().InstallDefault();
}
