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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/feature/parallel_concat_start_op_pass.h"
#include "graph/compute_graph.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestParallel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
  OpDescPtr CreateOpDesc(const string &name, const string &type, const GeTensorDesc &in_tensor, uint32_t input_num,
                         const GeTensorDesc &out_tensor, uint32_t output_num) {
    OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
    if (op_desc == nullptr) {
      return nullptr;
    }
    for (uint32_t i = 0; i < input_num; i++) {
      op_desc->AddInputDesc(in_tensor);
    }
    for (uint32_t i = 0; i < output_num; i++) {
      op_desc->AddOutputDesc(out_tensor);
    }
    return op_desc;
  }
};
TEST_F(UtestParallel, parallelRun_dataNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("node", DATA, GeTensorDesc(), 1, GeTensorDesc(), 1));
  ParallelConcatStartOpPass pass;
  EXPECT_EQ(pass.Run(data_node), SUCCESS);
}

TEST_F(UtestParallel, parallelRun_succ1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr node = graph->AddNode(CreateOpDesc("node", PARALLELCONCATSTART, GeTensorDesc(), 1, GeTensorDesc(), 1));
  ParallelConcatStartOpPass pass;
  ASSERT_EQ(pass.Run(node), PARAM_INVALID);
}

TEST_F(UtestParallel, parallelRun_succ2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr node = graph->AddNode(CreateOpDesc("node", PARALLELCONCATSTART, GeTensorDesc(), 1, GeTensorDesc(), 1));
  AttrUtils::SetDataType(node->GetOpDesc(), "dtype", ge::DT_INT64);
  ParallelConcatStartOpPass pass;
  ASSERT_EQ(pass.Run(node), PARAM_INVALID);
}

TEST_F(UtestParallel, parallelRun_succ3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr node = graph->AddNode(CreateOpDesc("node", PARALLELCONCATSTART, GeTensorDesc(), 1, GeTensorDesc(), 1));
  AttrUtils::SetDataType(node->GetOpDesc(), "dtype", ge::DT_INT64);
  AttrUtils::SetListInt(node->GetOpDesc(), "shape", std::vector<int32_t>({1, 2, 3}));
  ParallelConcatStartOpPass pass;
  EXPECT_EQ(pass.Run(node), SUCCESS);
}
}
