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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "framework/memory/memory_assigner.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "graph/build/memory/graph_mem_assigner.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/max_block_mem_assigner.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"
#include "../passes/graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestMemoryAssigner : public testing::Test {
protected:
  void SetUp() {
  }
  void TearDown() {
  }
public:
  NodePtr UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt);
};

NodePtr UtestMemoryAssigner::UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt){
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    std::vector<int64_t> shape = {1, 1, 224, 224};
    tensor_desc->SetShape(GeShape(shape));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);
    tensor_desc->SetOriginFormat(FORMAT_NCHW);
    tensor_desc->SetOriginShape(GeShape(shape));
    tensor_desc->SetOriginDataType(DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < in_cnt; ++i) {
        op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int i = 0; i < out_cnt; ++i) {
        op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    return graph->AddNode(op_desc);
}

TEST_F(UtestMemoryAssigner, Success) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
    std::map<uint64_t, size_t> mem_offset;
    size_t zero_copy_mem_size = 10;
    GraphMemoryAssigner graph_mem_assigner(graph);
    EXPECT_EQ(graph_mem_assigner.AssignMemory(), SUCCESS);
    EXPECT_EQ(graph_mem_assigner.ReAssignMemory(mem_offset), SUCCESS);
    EXPECT_EQ(graph_mem_assigner.AssignZeroCopyMemory(mem_offset, zero_copy_mem_size), SUCCESS);
    EXPECT_EQ(graph_mem_assigner.AssignMemory2HasRefAttrNode(), SUCCESS);
    EXPECT_EQ(graph_mem_assigner.AssignReferenceMemory(), SUCCESS);
    EXPECT_EQ(graph_mem_assigner.AssignVarAttr2Nodes(), SUCCESS);
    EXPECT_EQ(graph_mem_assigner.SetInputOffset(), SUCCESS);
    MemoryAssigner mem_assigner(graph);
    EXPECT_EQ(mem_assigner.AssignMemory(mem_offset, zero_copy_mem_size), SUCCESS);
}


} // namespace ge
