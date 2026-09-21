/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_unsupported_op/print_op_pass.h"

#include <gtest/gtest.h>

#include "omg/omg_inner_types.h"
#include "utils/op_desc_utils.h"

using domi::GetContext;
namespace ge {
class UtestGraphPassesPrintOpPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void make_graph(ComputeGraphPtr graph, bool match = true, int flag = 0) {
    auto data = std::make_shared<OpDesc>("Data", DATA);
    GeTensorDesc tensor_desc_data(GeShape({1, 1, 1, 1}));
    data->AddInputDesc(tensor_desc_data);
    data->AddOutputDesc(tensor_desc_data);
    auto data_node = graph->AddNode(data);

    auto data1 = std::make_shared<OpDesc>("Data", DATA);
    data1->AddInputDesc(tensor_desc_data);
    data1->AddOutputDesc(tensor_desc_data);
    auto data_node1 = graph->AddNode(data1);

    auto print_desc = std::make_shared<OpDesc>("Print", "Print");
    print_desc->AddInputDesc(tensor_desc_data);
    print_desc->AddInputDesc(tensor_desc_data);
    print_desc->AddOutputDesc(tensor_desc_data);
    auto print_node = graph->AddNode(print_desc);

    auto ret_val_desc = std::make_shared<OpDesc>("RetVal", "RetVal");
    ret_val_desc->AddInputDesc(tensor_desc_data);
    ret_val_desc->AddOutputDesc(tensor_desc_data);
    auto ret_val_node = graph->AddNode(ret_val_desc);

    (void)GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), print_node->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), print_node->GetInDataAnchor(1));
    (void)GraphUtils::AddEdge(print_node->GetOutDataAnchor(0), ret_val_node->GetInDataAnchor(0));
  }
};

TEST_F(UtestGraphPassesPrintOpPass, apply_success) {
  GetContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph(graph);
  ge::PrintOpPass apply_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("PrintOpPass", &apply_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesPrintOpPass, param_invalid) {
  ge::NodePtr node = nullptr;
  ge::PrintOpPass apply_pass;
  Status status = apply_pass.Run(node);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}
}  // namespace ge
