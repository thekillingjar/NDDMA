/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/node_faker.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "register/node_converter_registry.h"
#include <gtest/gtest.h>
#include "kernel/common_kernel_impl/build_tensor.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "engine/aicpu/converter/sequence/sequence_ops.h"
#include "engine/gelocal/inputs_converter.h"
#include "common/const_data_helper.h"

namespace gert {
using namespace bg;
class SequenceLengthUT : public BgTestAutoCreate3StageFrame {};
TEST_F(SequenceLengthUT, success) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("SequenceLengthNode", "SequenceLength")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("SequenceLengthNode", "SequenceLength"));
  };

  auto graph = ge::ToComputeGraph(g1);
  auto split_to_sequence_op_desc = graph->FindNode("SequenceLengthNode")->GetOpDesc();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1 = graph->FindNode("data1");
  ge::AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto data2 = graph->FindNode("data2");
  ge::AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  auto data1_ret = LoweringDataNode(data1, data_input);
  auto data2_ret = LoweringDataNode(data2, data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto sequence_ret = LoweringSequenceLength(graph->FindNode("SequenceLengthNode"), input);
  ASSERT_TRUE(sequence_ret.result.IsSuccess());
  ASSERT_EQ(sequence_ret.out_shapes.size(), 1);
  ASSERT_EQ(sequence_ret.out_addrs.size(), 1);
}

TEST_F(SequenceLengthUT, failed) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("SequenceLengthNode", "SequenceLength")->NODE("NetOutput", "NetOutput"));
  };

  auto graph = ge::ToComputeGraph(g1);
  auto split_to_sequence_op_desc = graph->FindNode("SequenceLengthNode")->GetOpDesc();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput input = {{nullptr},{nullptr},&global_data};
  auto sequence_ret = LoweringSequenceLength(graph->FindNode("SequenceLengthNode"), input);
  ASSERT_FALSE(sequence_ret.result.IsSuccess());
}
}  // namespace gert
