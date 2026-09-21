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
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "runtime/mem.h"
#include "graph/passes/memory_optimize/concat_notask_pass.h"

using namespace std;
using namespace ge;

class UtestConcatNotaskPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static Graph BuildStroageFormatTrainGraph1(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape) {
  DEF_GRAPH(g1) {
    auto data1 = OP_DATA(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_1");
    auto data2 = OP_DATA(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_2");
  auto HcomAllGather1 = OP_CFG(HCOMALLGATHER)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_1");
    HcomAllGather1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather1->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather1->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto HcomAllGather2 = OP_CFG(HCOMALLGATHER)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_2");
    HcomAllGather2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather2->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather2->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto concat = OP_CFG("ConcatD")
        .InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    CHAIN(NODE(data1)->NODE(HcomAllGather1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data2)->NODE(HcomAllGather2)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };
  return ToGeGraph(g1);                                                        
}

static Graph BuildStroageFormatTrainGraph2(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape) {
  DEF_GRAPH(g1) {
    auto data1 = OP_DATA(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_1");
    auto data2 = OP_DATA(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_2");
    auto HcomAllGather1 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_1");
    HcomAllGather1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather1->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather1->MutableOutputDesc(0)->SetShape(GeShape(shape));
    auto HcomAllGather2 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_2");
    HcomAllGather2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather2->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather2->MutableOutputDesc(0)->SetShape(GeShape(shape));
    auto concat = OP_CFG("ConcatD")
        .InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    CHAIN(NODE(data1)->NODE(HcomAllGather1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data2)->NODE(HcomAllGather2)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };
  return ToGeGraph(g1);                                                        
}

static Graph BuildStroageFormatTrainGraphReshape(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape) {
  DEF_GRAPH(g1) {
    auto data1 = OP_DATA(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_1");
    auto data2 = OP_DATA(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_2");
    auto HcomAllGather1 = OP_CFG(HCOMALLGATHER)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_1");
    HcomAllGather1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather1->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather1->MutableOutputDesc(0)->SetShape(GeShape(shape));
    auto HcomAllGather2 = OP_CFG(HCOMALLGATHER)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_2");
    HcomAllGather2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather2->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather2->MutableOutputDesc(0)->SetShape(GeShape(shape));
    auto concat = OP_CFG("ConcatD")
        .InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    auto reshape = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).Build("reshape");
    CHAIN(NODE(data1)->NODE(HcomAllGather1)->NODE(reshape)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data2)->NODE(HcomAllGather2)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };
  return ToGeGraph(g1);                                                        
}
TEST_F(UtestConcatNotaskPass, allgather_connect_to_concat) {
  auto train_graph =
      BuildStroageFormatTrainGraph1(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_EQ(can_reuse, false);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_EQ(can_reuse, false);
}

TEST_F(UtestConcatNotaskPass, allgather_connect_to_concat_reshape) {
  auto train_graph =
      BuildStroageFormatTrainGraphReshape(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);

  bool can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_EQ(can_reuse, false);
}

static Graph BuildRefDataWithStroageFormatTrainGraph1(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape,
                                                             const ge::DataType data_type) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d1->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d1->MutableOutputDesc(0)->SetDataType(data_type);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d2->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d2->MutableOutputDesc(0)->SetDataType(data_type);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(0)->SetDataType(data_type);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetDataType(data_type);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(storage_format);

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(conv2d2));
    CHAIN(NODE(refdata2)->EDGE(0, 1)->NODE(conv2d2)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildTrainGraphPreNodeConnectOutput(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape,
                                                             const ge::DataType data_type) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d1->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d1->MutableOutputDesc(0)->SetDataType(data_type);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d2->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d2->MutableOutputDesc(0)->SetDataType(data_type);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(0)->SetDataType(data_type);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetDataType(data_type);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(storage_format);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(2).OutCnt(1).Build("output");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(conv2d2)->NODE(netoutput));
    CHAIN(NODE(refdata2)->EDGE(0, 1)->NODE(conv2d2)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildTrainGraphPreNodeConnectRelu(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape,
                                                             const ge::DataType data_type) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d1->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d1->MutableOutputDesc(0)->SetDataType(data_type);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d2->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d2->MutableOutputDesc(0)->SetDataType(data_type);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(0)->SetDataType(data_type);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetDataType(data_type);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(storage_format);
    auto relu1 = OP_CFG(RELU)
      .InCnt(1).OutCnt(1).Build("relu_1");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(conv2d2)->NODE(relu1));
    CHAIN(NODE(refdata2)->EDGE(0, 1)->NODE(conv2d2)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildRefDataWithStroageFormatTrainGraphSameAnchor(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::string &expand_dims_rule) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->SetOpEngineName("AIcoreEngine");
    auto reshape = OP_CFG(RESHAPE).InCnt(1).OutCnt(1).Build("reshape");
    reshape->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    reshape->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    reshape->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    reshape->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(reshape)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(conv2d1)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildStroageFormatTrainGraphFusionOp(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape) {
  DEF_GRAPH(g1) {
    auto data1 = OP_DATA(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_1");
    auto data2 = OP_DATA(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_2");
  auto HcomAllGather1 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_1");
    HcomAllGather1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather1->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather1->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto HcomAllGather2 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_2");
    HcomAllGather2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather2->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather2->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto concat = OP_CFG("ConcatD")
        .InCnt(2).OutCnt(1).Build("concat_lxslice");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    CHAIN(NODE(data1)->NODE(HcomAllGather1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data1)->NODE(HcomAllGather2)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };
  return ToGeGraph(g1);                                                        
}

TEST_F(UtestConcatNotaskPass, concat_notask_lxfusion_op) {
  auto train_graph =
      BuildStroageFormatTrainGraphFusionOp(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat_lxslice");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(UtestConcatNotaskPass, input_mem_type_invalid) {
  auto graph =
      BuildStroageFormatTrainGraph2(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");
  std::vector<int64_t> in_mem_type {RT_MEMORY_L1, RT_MEMORY_L1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST,
                                in_mem_type);

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(UtestConcatNotaskPass, output_mem_type_invalid) {
  auto graph =
      BuildStroageFormatTrainGraph2(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");
  std::vector<int64_t> in_mem_type {RT_MEMORY_L1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                in_mem_type);

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

static Graph BuildTrainGraphOutputInvalid(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape) {
  DEF_GRAPH(g1) {
    auto data1 = OP_DATA(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_1");
    auto data2 = OP_DATA(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_2");
  auto HcomAllGather1 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_1");
    HcomAllGather1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather1->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather1->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto HcomAllGather2 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_2");
    HcomAllGather2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather2->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather2->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto concat = OP_CFG("ConcatD")
        .InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
  auto relu1 = OP_CFG(RELU)
      .InCnt(1).OutCnt(1).Build("relu_1");
    auto reshape = OP_CFG(RESHAPE).InCnt(1).OutCnt(1).Build("reshape");
    CHAIN(NODE(data1)->NODE(HcomAllGather1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data1)->NODE(HcomAllGather2)->EDGE(0, 1)->NODE(concat)->NODE(reshape)->NODE(relu1));
    ADD_OUTPUT(relu1, 0);
  };
  return ToGeGraph(g1);                                                        
}

TEST_F(UtestConcatNotaskPass, concat_notask_output_invalid) {
  auto graph =
      BuildTrainGraphOutputInvalid(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();

  auto relu1_node = compute_graph->FindNode("relu_1");
  auto relu1_desc = relu1_node->GetOpDesc();
  (void)ge::AttrUtils::SetBool(relu1_desc, ge::ATTR_NAME_NOTASK, true);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(UtestConcatNotaskPass, concat_notask_pre_node_attr_invalid) {
  auto graph =
      BuildStroageFormatTrainGraph2(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");
  (void)ge::AttrUtils::SetBool(allgather_desc2, ge::ATTR_NAME_CONTINUOUS_INPUT, true);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(UtestConcatNotaskPass, concat_notask_pre_node_attr_notask) {
  auto graph =
      BuildStroageFormatTrainGraph2(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");
  (void)ge::AttrUtils::SetBool(allgather_desc2, ge::ATTR_NAME_NOTASK, true);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(UtestConcatNotaskPass, concat_notask_pre_node_attr_atomic) {
  auto graph =
      BuildStroageFormatTrainGraph2(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");
  (void)ge::AttrUtils::SetListInt(allgather_desc2, ge::ATOMIC_ATTR_OUTPUT_INDEX, {0});

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(UtestConcatNotaskPass, concat_notask_input_mem_type_not_same) {
  auto graph =
      BuildStroageFormatTrainGraph2(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");
  std::vector<int64_t> in_mem_type {RT_MEMORY_L1};
  (void)ge::AttrUtils::SetListInt(allgather_desc2, ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                in_mem_type);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

static Graph BuildTrainGraphDataConcat(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape) {
  DEF_GRAPH(g1) {
    auto data1 = OP_DATA(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_1");
    auto data2 = OP_DATA(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288}).Build("data_2");
  auto HcomAllGather1 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_1");
    HcomAllGather1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather1->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather1->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto HcomAllGather2 = OP_CFG(RELU)
        .InCnt(1).OutCnt(1).Build("HcomAllGather_2");
    HcomAllGather2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    HcomAllGather2->MutableOutputDesc(0)->SetFormat(storage_format);
    HcomAllGather2->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    HcomAllGather2->MutableOutputDesc(0)->SetShape(GeShape(shape));
  auto concat = OP_CFG("ConcatD")
        .InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(storage_format);
    concat->MutableInputDesc(1)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(1)->SetShape(GeShape(shape));
    CHAIN(NODE(data1)->NODE(concat));
    CHAIN(NODE(data2)->NODE(HcomAllGather2)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };
  return ToGeGraph(g1);                                                        
}

TEST_F(UtestConcatNotaskPass, data_connect_to_concat) {
  auto graph =
      BuildTrainGraphDataConcat(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  const auto in_anchor = std::make_shared<InDataAnchor>(concat_node, 0);
  EXPECT_EQ(pass.IsPreNodeTypeValid(in_anchor), false);
  EXPECT_EQ(pass.IsPreNodeWithSubgraph(in_anchor), false);
}

TEST_F(UtestConcatNotaskPass, allgather_connect_to_concat_unknownop) {
  auto graph =
      BuildTrainGraphOutputInvalid(FORMAT_ND, FORMAT_ND, {2048, 12288}, {2048, 12288});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);

  auto allgather_node1 = compute_graph->FindNode("relu_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();

  ConcatNotaskPass pass;
  op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({4096, -1}));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4096, -1}));
  pass.Run(compute_graph);
  op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({4096, 12288}));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4096, 12288}));
  op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({4096, -1}));
  op_desc->MutableOutputDesc(0)->SetShape(GeShape({4096, -1}));
  pass.Run(compute_graph);
  op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({4096, 12288}));
  op_desc->MutableOutputDesc(0)->SetShape(GeShape({4096, 12288}));
  compute_graph->SetGraphUnknownFlag(true);
  pass.Run(compute_graph);
  
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

TEST_F(UtestConcatNotaskPass, pre_node_connect_output) {
  auto train_graph =
      BuildTrainGraphPreNodeConnectOutput(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, pre_node_connect_relu) {
  auto train_graph =
      BuildTrainGraphPreNodeConnectRelu(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);

  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_pre_outanchor_can_reuse) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  auto conv2d1_node2 = compute_graph->FindNode("conv2d1");
  auto op_desc2 = conv2d1_node2->GetOpDesc();
  auto output_tensor_desc = op_desc2->MutableOutputDesc(0);
  (void)ge::AttrUtils::SetBool(output_tensor_desc, "can_reused_for_concat_optimize", false);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_has_same_anchor) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraphSameAnchor(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_c1_not_align) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 31, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_tensor_not_align) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1,2,15,15,15}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_dim_minus3) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", -3);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_dim_minus10) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", -10);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_memory_discontinuous) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {1, 2, 16, 16, 16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  AttrUtils::SetBool(compute_graph, ATTR_NAME_MEMORY_DISCONTIGUOUS_ALLOCATION, true);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", -3);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_first_axis_invalid) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {2,2,16,16,16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_shape_invalid) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {2}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(UtestConcatNotaskPass, conv_5hd_connect_to_concat_ori_shape_invalid) {
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {32}, {1,2,16,16,16}, DT_FLOAT16);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  ConcatNotaskPass pass;
  auto ret = pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  auto concat = OP_CFG("ConcatD")
    .InCnt(2).OutCnt(1).Build("concat2");
  concat->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2}));
  concat->MutableInputDesc(0)->SetShape(GeShape({1, 2}));
  concat->MutableInputDesc(1)->SetOriginShape(GeShape({1, 2}));
  concat->MutableInputDesc(1)->SetShape(GeShape({1, 2}));
  gert::Shape aligned_shape;
  aligned_shape.SetDimNum(2);
  aligned_shape[0] = 0;
  aligned_shape[1] = 1;
  GeShape ori_shape({1, 2});
  EXPECT_EQ(pass.CheckConcatDimAlignment(concat, aligned_shape, 0, ori_shape), false);
  concat->MutableInputDesc(0)->SetOriginShape(GeShape({1,2}));
  aligned_shape[0] = 1;
  aligned_shape[0] = 3;
  EXPECT_EQ(pass.CheckConcatDimAlignment(concat, aligned_shape, 0, ori_shape), false);
}

static Graph BuildTrainGraphConcatOneInput(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::vector<int64_t> &shape,
                                                             const ge::DataType data_type) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(storage_format);
    conv2d1->MutableOutputDesc(0)->SetOriginShape(GeShape(origin_shape));
    conv2d1->MutableOutputDesc(0)->SetShape(GeShape(shape));
    conv2d1->MutableOutputDesc(0)->SetDataType(data_type);

    auto concat = OP_CFG("ConcatD").InCnt(1).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(storage_format);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    concat->MutableInputDesc(0)->SetShape(GeShape(shape));
    concat->MutableInputDesc(0)->SetDataType(data_type);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(storage_format);

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}
TEST_F(UtestConcatNotaskPass, nullptr_check) {
  ConcatNotaskPass pass;
  auto train_graph2 =
      BuildTrainGraphConcatOneInput(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {2}, DT_FLOAT16);
  auto compute_graph2 = GraphUtilsEx::GetComputeGraph(train_graph2);
  auto concat_node2 = compute_graph2->FindNode("concat");
  EXPECT_EQ(pass.CheckTensorAlign(concat_node2, 0), true);
  auto train_graph3 =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, {2}, DT_UNDEFINED);
  auto compute_graph3 = GraphUtilsEx::GetComputeGraph(train_graph3);
  auto concat_node3 = compute_graph3->FindNode("concat");
  EXPECT_EQ(pass.CheckTensorAlign(concat_node3, 0), false);
  EXPECT_EQ(pass.IsOwnerGraphUnknown(concat_node3), false);
  compute_graph3->SetGraphUnknownFlag(true);
  EXPECT_EQ(pass.IsOwnerGraphUnknown(concat_node3), true);
  (void)AttrUtils::SetBool(compute_graph2, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  EXPECT_EQ(pass.IsOwnerGraphUnknown(concat_node2), true);
}

TEST_F(UtestConcatNotaskPass, CheckRealConcatDimTest) {
  ConcatNotaskPass pass;
  gert::Shape aligned_shape;
  aligned_shape.SetDimNum(4);
  aligned_shape[0] = 1;
  aligned_shape[1] = 16;
  aligned_shape[2] = 1;
  aligned_shape[3] = 1;
  gert::Shape src_shape;
  src_shape.SetDimNum(4);
  src_shape[0] = 1;
  src_shape[1] = 16;
  src_shape[2] = 5;
  src_shape[3] = 6;
  transformer::AxisIndexMapping axis_index_mapping;
  axis_index_mapping.src_to_dst_transfer_dims = {{1, 2}, {0, 3}, {0}, {0}};
  axis_index_mapping.dst_to_src_transfer_dims = {{1,2,3},{0},{0},{1}};
  int64_t concat_dim = 2;
  ge::GeTensorDesc input_tensor(GeShape({30,1,16,16}), ge::FORMAT_FRACTAL_Z, DT_FLOAT16);
  EXPECT_EQ(pass.CheckRealConcatDim(aligned_shape, src_shape, axis_index_mapping, concat_dim,input_tensor), true);
  concat_dim = 3;
  EXPECT_EQ(pass.CheckRealConcatDim(aligned_shape, src_shape, axis_index_mapping, concat_dim,input_tensor), false);
  src_shape[2] = 1;
  EXPECT_EQ(pass.CheckRealConcatDim(aligned_shape, src_shape, axis_index_mapping, concat_dim,input_tensor), true);
  concat_dim = 0;
  ge::GeTensorDesc input_tensor3(GeShape({6,1,16,16}), ge::FORMAT_FRACTAL_Z, DT_FLOAT16);
  EXPECT_EQ(pass.CheckRealConcatDim(aligned_shape, src_shape, axis_index_mapping, concat_dim,input_tensor3), false);
  src_shape[3] = 1;
  ge::GeTensorDesc input_tensor2(GeShape({1,1,16,16}), ge::FORMAT_FRACTAL_Z, DT_FLOAT16);
  EXPECT_EQ(pass.CheckRealConcatDim(aligned_shape, src_shape, axis_index_mapping, concat_dim,input_tensor2), true);
  src_shape[1] = 32;
  concat_dim = 3;
  ge::GeTensorDesc input_tensor4(GeShape({2,1,16,16}), ge::FORMAT_FRACTAL_Z, DT_FLOAT16);
  EXPECT_EQ(pass.CheckRealConcatDim(aligned_shape, src_shape, axis_index_mapping, concat_dim,input_tensor4), false);
}

TEST_F(UtestConcatNotaskPass, scalar_input_check) {
    auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {6});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto relu1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {6});
  auto relu2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("relu1", relu1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data2)->NODE("relu2", relu2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  std::vector<int64_t> dims;
  op_desc->MutableInputDesc(0)->SetOriginShape(GeShape(dims));
  dims.push_back(100);
  op_desc->MutableInputDesc(1)->SetOriginShape(GeShape(dims));
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  ConcatNotaskPass pass;
  EXPECT_EQ(pass.IsScalarInput(concat_node, 0), true);
  EXPECT_EQ(pass.IsScalarInput(concat_node, 1), false);
  EXPECT_EQ(pass.InputCheck(concat_node), false);
}
