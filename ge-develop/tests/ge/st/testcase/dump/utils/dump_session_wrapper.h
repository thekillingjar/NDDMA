/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTCASE_DUMP_UTILS_DUMP_SESSION_WRAPPER_H
#define TESTCASE_DUMP_UTILS_DUMP_SESSION_WRAPPER_H

#include <map>
#include <vector>

// Headers for setting up test framework.
#include <gtest/gtest.h>
#include "faker/fake_value.h"                   // FakeTensors
#include "ge_graph_dsl/graph_dsl.h"             // DEF_GRPAH

#include "common/types.h"                       // Operator names.
#include "graph/utils/graph_utils_ex.h"         // CreateGraphFromComputeGraph

// Public interfaces.
#include "ge/ge_api.h"                          // Session::*

constexpr int64_t kTensorDim = 16;

namespace ge {
struct GraphAndIoNum {
  auto operator|(void (*modifier)(const ComputeGraphPtr &)) && {
    modifier(graph);
    return std::move(*this);
  }
  ComputeGraphPtr graph;
  size_t in_cnt;
  size_t out_cnt;
};

class SessionWrapper {
public:
  SessionWrapper(GraphAndIoNum graph_io,
                 std::map<AscendString, AscendString> options = {}) : session_(options) {
    auto [graph, in_cnt, out_cnt] = std::move(graph_io);
    inputs_ = gert::FakeTensors({kTensorDim}, in_cnt).Steal();
    outputs_ = gert::FakeTensors({kTensorDim}, out_cnt).Steal();

    EXPECT_EQ(session_.AddGraph(0, GraphUtilsEx::CreateGraphFromComputeGraph(graph)), SUCCESS);
    EXPECT_EQ(session_.CompileGraph(0), SUCCESS);
    EXPECT_EQ(session_.LoadGraph(0, options, nullptr), SUCCESS);
  }
  void Run() {
    EXPECT_EQ(session_.ExecuteGraphWithStreamAsync(0, nullptr, inputs_, outputs_), SUCCESS);
  }
  void Finalize() {
    EXPECT_EQ(session_.RemoveGraph(0), SUCCESS);
  }
private:
  Session session_;
  std::vector<gert::Tensor> inputs_;
  std::vector<gert::Tensor> outputs_;
};

// +---------+  (0,0)   +---------+  (0,0)   +-----------+
// | _data_0 | -------> | _add_0  | -------> | _output_0 |
// +---------+          +---------+          +-----------+
//                        ^
//                        | (0,1)
//                        |
//                      +---------+
//                      | _data_1 |
//                      +---------+
inline GraphAndIoNum BuildGraph_OneAdd(bool dynamic) {
  auto dim = dynamic ? -1 : kTensorDim;
  auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_data_0");
  auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_data_1");
  auto add_0 = OP_CFG(ADD).InCnt(2).OutCnt(1)
                          .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_0_fake_id")
                          .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_add_0");
  auto output_0 = OP_CFG(NETOUTPUT)
                      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_output_0_fake_id")
                      .Build("_output_0");
  DEF_GRAPH(one_add) {
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(add_0));
    CHAIN(NODE(data_1)->EDGE(0, 1)->NODE(add_0));
    CHAIN(NODE(add_0)->NODE(output_0));
  };

  auto graph = ToComputeGraph(one_add);
  graph->SetGraphUnknownFlag(dynamic);
  return { graph, 2, 1 };
}

//             (0,0)
//   +-----------------------------------------+
//   |                                         v
// +---------+  (0,0)   +---------+  (0,1)   +--------+  (0,0)   +-----------+
// | _data_0 | -------> | _add_0  | -------> | _add_1 | -------> | _output_0 |
// +---------+          +---------+          +--------+          +-----------+
//                        ^
//                        | (0,1)
//                        |
//                      +---------+
//                      | _data_1 |
//                      +---------+
inline GraphAndIoNum BuildGraph_TwoAdd(bool dynamic) {
  auto dim = dynamic ? -1 : kTensorDim;
  auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_data_0");
  auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_data_1");
  auto add_0 = OP_CFG(ADD).InCnt(2).OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_add_0");
  auto add_1 = OP_CFG(ADD).InCnt(2).OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_FLOAT, {dim}).Build("_add_1");
  auto output_0 = OP_CFG(NETOUTPUT).Build("_output_0");
  DEF_GRAPH(two_add) {
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(add_0));
    CHAIN(NODE(data_1)->EDGE(0, 1)->NODE(add_0));
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(add_1));
    CHAIN(NODE(add_0)->EDGE(0, 1)->NODE(add_1));
    CHAIN(NODE(add_1)->NODE(output_0));
  };

  auto graph = ToComputeGraph(two_add);
  graph->SetGraphUnknownFlag(dynamic);
  return { graph, 2, 1 };
}
} // namespace ge

#endif
