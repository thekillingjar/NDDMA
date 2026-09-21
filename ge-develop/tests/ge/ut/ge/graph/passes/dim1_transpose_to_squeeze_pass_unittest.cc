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

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/format_optimize/dim1_transpose_to_squeeze_pass.h"

#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "graph/passes/pass_manager.h"
#include "graph_builder_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"
#include "attribute_group/attr_group_symbolic_desc.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestDim1TransposeToSqueezePass : public Test {
 protected:
  void SetUp() {}
  void TearDown() {}

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  ComputeGraphPtr BuildGraph(std::vector<int64_t> &data_vec, std::vector<inner_dim_type> &dim_value_vec,
                             std::vector<int64_t> &result_vec, const string &transpose_type, const bool isSymbol) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

    // add data node
    OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    std::vector<inner_data_type> data_value_vec(dims_size, inner_data_type(1));
    GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_ND, data_type);
    GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    if (isSymbol) {
      gert::SymbolShape symbol_shape({
          Symbol(1),
          Symbol(1),
          Symbol("s0"),
          Symbol("s1"),
      });
      const auto attr = data_tensor_desc.GetOrCreateAttrsGroup<SymbolicDescAttr>();
      attr->symbolic_tensor.SetSymbolShape(symbol_shape);
    }
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
    std::vector<int64_t> dim_vec;
    dim_vec.push_back(dim_value_vec.size());
    GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_ND, dim_type);
    GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                        dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add perm node
    OpDescPtr perm_op_desc = std::make_shared<OpDesc>("Transpose", transpose_type);
    if (transpose_type == TRANSPOSED) {
      perm_op_desc->SetAttr(PERMUTE_ATTR_PERM, GeAttrValue::CreateFrom<std::vector<inner_dim_type>>(dim_value_vec));
    }
    GeTensorDesc perm_tensor_desc(ge::GeShape(result_vec), FORMAT_ND, data_type);
    GeTensorPtr perm_tensor = std::make_shared<GeTensor>(perm_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(perm_op_desc, perm_tensor);
    perm_op_desc->AddInputDesc(data_tensor_desc);
    perm_op_desc->AddInputDesc(dim_tensor_desc);
    perm_op_desc->AddOutputDesc(perm_tensor_desc);
    NodePtr op_node = graph->AddNode(perm_op_desc);
    op_node->Init();

    // add output node
    OpDescPtr netoutput_op_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    netoutput_op_desc->AddInputDesc(perm_tensor_desc);
    NodePtr netoutput_node = graph->AddNode(netoutput_op_desc);
    netoutput_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(op_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    return graph;
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  ComputeGraphPtr BuildGraphForDifferentFormat(std::vector<int64_t> &data_vec, std::vector<inner_dim_type> &dim_value_vec,
                             std::vector<int64_t> &result_vec, const string &transpose_type, const bool isSymbol) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

    // add data node
    OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    std::vector<inner_data_type> data_value_vec(dims_size, inner_data_type(1));
    GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, data_type);
    GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    if (isSymbol) {
      gert::SymbolShape symbol_shape({
          Symbol(1),
          Symbol(1),
          Symbol("s0"),
          Symbol("s1"),
      });
      const auto attr = data_tensor_desc.GetOrCreateAttrsGroup<SymbolicDescAttr>();
      attr->symbolic_tensor.SetSymbolShape(symbol_shape);
    }
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
    std::vector<int64_t> dim_vec;
    dim_vec.push_back(dim_value_vec.size());
    GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_ND, dim_type);
    GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                        dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add perm node
    OpDescPtr perm_op_desc = std::make_shared<OpDesc>("Transpose", transpose_type);
    if (transpose_type == TRANSPOSED) {
      perm_op_desc->SetAttr(PERMUTE_ATTR_PERM, GeAttrValue::CreateFrom<std::vector<inner_dim_type>>(dim_value_vec));
    }
    GeTensorDesc perm_tensor_desc(ge::GeShape(result_vec), FORMAT_NCHW, data_type);
    GeTensorPtr perm_tensor = std::make_shared<GeTensor>(perm_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(perm_op_desc, perm_tensor);
    perm_op_desc->AddInputDesc(data_tensor_desc);
    perm_op_desc->AddInputDesc(dim_tensor_desc);
    perm_op_desc->AddOutputDesc(perm_tensor_desc);
    NodePtr op_node = graph->AddNode(perm_op_desc);
    op_node->Init();

    // add output node
    OpDescPtr netoutput_op_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    netoutput_op_desc->AddInputDesc(perm_tensor_desc);
    NodePtr netoutput_node = graph->AddNode(netoutput_op_desc);
    netoutput_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(op_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    return graph;
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeReplaceFailed(std::vector<int64_t> &data_vec, std::vector<inner_dim_type> &dim_value_vec,
                                          std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraph<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSE, false);
    GE_DUMP(graph, "Before RemoveConstInput");
    auto dim_node = graph->FindNode("dim");
    graph->RemoveNode(dim_node);
    auto op_node = graph->FindNode("Transpose");
    GE_DUMP(graph, "After RemoveConstInput");
    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(3, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(3, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
    auto transpose = graph->FindNode("Transpose");
    auto netOutput = graph->FindNode("NetOutput");
    EXPECT_NE(transpose, nullptr);
    EXPECT_NE(netOutput, nullptr);
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeReplaceSuccess(std::vector<int64_t> &data_vec, std::vector<inner_dim_type> &dim_value_vec,
                                          std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraph<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSE, false);
    auto op_node = graph->FindNode("Transpose");
    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(4, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(4, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
    auto squeeze = graph->FindNode("Transpose_replaced_Squeeze");
    auto unsqueeze = graph->FindNode("Transpose_replaced_Unsqueeze");
    auto netOutput = graph->FindNode("NetOutput");
    EXPECT_NE(squeeze, nullptr);
    EXPECT_NE(unsqueeze, nullptr);
    EXPECT_NE(netOutput, nullptr);
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeReplaceNotTrigger(std::vector<int64_t> &data_vec, std::vector<inner_dim_type> &dim_value_vec,
                                             std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraph<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSE, false);
    auto op_node = graph->FindNode("Transpose");

    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(4, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(4, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeDReplaceNotTrigger(std::vector<int64_t> &data_vec,
                                              std::vector<inner_dim_type> &dim_value_vec,
                                              std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraph<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSED, false);
    auto op_node = graph->FindNode("Transpose");

    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(4, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(4, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeSymbolReplaceNotTrigger(std::vector<int64_t> &data_vec,
                                                   std::vector<inner_dim_type> &dim_value_vec,
                                                   std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraph<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSE, true);
    auto op_node = graph->FindNode("Transpose");

    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(4, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(4, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeSymbolReplaceTrigger(std::vector<int64_t> &data_vec,
                                                std::vector<inner_dim_type> &dim_value_vec,
                                                std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraph<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSE, true);
    auto op_node = graph->FindNode("Transpose");

    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(4, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(4, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
    auto squeeze_node = graph->FindNode("Transpose_replaced_Squeeze");
    auto unsqueeze_node = graph->FindNode("Transpose_replaced_Unsqueeze");
    EXPECT_NE(squeeze_node, nullptr);
    EXPECT_NE(unsqueeze_node, nullptr);
    auto squeeze_out_symbolic = squeeze_node->GetOpDesc()
                                    ->GetOutputDesc(0)
                                    .GetAttrsGroup<SymbolicDescAttr>()
                                    ->symbolic_tensor.GetOriginSymbolShape();
    auto squeeze_out = squeeze_node->GetOpDesc()->GetOutputDesc(0).GetShape();
    EXPECT_EQ(squeeze_out_symbolic, gert::SymbolShape({Symbol("s0"), Symbol("s1")}));
    EXPECT_EQ(squeeze_out.GetDim(0), -1);
    EXPECT_EQ(squeeze_out.GetDim(1), -1);
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void EXPECT_TestTransposeInsertedByFeNotTrigger(std::vector<int64_t> &data_vec,
                                                std::vector<inner_dim_type> &dim_value_vec,
                                                std::vector<int64_t> &result_vec) {
    ComputeGraphPtr graph = BuildGraphForDifferentFormat<inner_data_type, data_type, inner_dim_type, dim_type>(
        data_vec, dim_value_vec, result_vec, TRANSPOSE, true);
    auto op_node = graph->FindNode("Transpose");
    auto op_desc = op_node->GetOpDesc();
    op_desc->AppendIrAttrName("_inserted_by_fe");
    AttrUtils::SetInt(op_desc, "_inserted_by_fe", 1);
    std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
    EXPECT_EQ(4, graph->GetDirectNodesSize());
    Status ret = pass->Run(op_node);
    EXPECT_EQ(SUCCESS, ret);
    EXPECT_EQ(4, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
  }
};

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_failed) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int32_t> dim_value_vec = {1, 0, 2, 3};
  std::vector<int64_t> result_vec = {16, 1, 32, 64};
  EXPECT_TestTransposeReplaceFailed<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_succeed1) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int32_t> dim_value_vec = {1, 0, 2, 3};
  std::vector<int64_t> result_vec = {16, 1, 32, 64};
  EXPECT_TestTransposeReplaceSuccess<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_succeed2) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int64_t> dim_value_vec = {1, 2, 0, 3};
  std::vector<int64_t> result_vec = {16, 1, 32, 64};
  EXPECT_TestTransposeReplaceSuccess<uint64_t, DT_UINT64, int64_t, DT_INT64>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_failed1) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int16_t> dim_value_vec = {1, 0, 2, 3};
  std::vector<int64_t> result_vec = {16, 1, 32, 64};
  EXPECT_TestTransposeReplaceNotTrigger<uint64_t, DT_UINT64, int16_t, DT_INT16>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_not_trigger1) {
  std::vector<int64_t> data_vec = {16, 16, 16};
  std::vector<int32_t> dim_value_vec = {2, 1, 0};
  std::vector<int64_t> result_vec = {16, 16, 16};
  EXPECT_TestTransposeReplaceNotTrigger<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_not_trigger2) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int32_t> dim_value_vec = {2, 1, 0, 3};
  std::vector<int64_t> result_vec = {32, 16, 1, 64};
  EXPECT_TestTransposeReplaceNotTrigger<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpseD_not_trigger1) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int64_t> dim_value_vec = {2, 1, 0, 3};
  std::vector<int64_t> result_vec = {32, 16, 1, 64};
  EXPECT_TestTransposeDReplaceNotTrigger<uint64_t, DT_UINT64, int64_t, DT_INT64>(data_vec, dim_value_vec, result_vec);
}

TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_symbol_not_trigger1) {
  std::vector<int64_t> data_vec = {-1, -1, -1, -1};
  std::vector<int64_t> dim_value_vec = {3, 1, 0, 2};
  std::vector<int64_t> result_vec = {32, 16, 1, 64};
  EXPECT_TestTransposeSymbolReplaceNotTrigger<uint64_t, DT_UINT64, int64_t, DT_INT64>(data_vec, dim_value_vec,
                                                                                      result_vec);
}
TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_symbol_trigger1) {
  std::vector<int64_t> data_vec = {-1, -1, -1, -1};
  std::vector<int64_t> dim_value_vec = {1, 0, 2, 3};
  std::vector<int64_t> result_vec = {32, 16, 1, 64};
  EXPECT_TestTransposeSymbolReplaceTrigger<uint64_t, DT_UINT64, int64_t, DT_INT64>(data_vec, dim_value_vec, result_vec);
}
TEST_F(UtestDim1TransposeToSqueezePass, replace_transpse_inserted_by_fe_not_trigger1) {
  std::vector<int64_t> data_vec = {1, 16, 32, 64};
  std::vector<int32_t> dim_value_vec = {2, 1, 0, 3};
  std::vector<int64_t> result_vec = {32, 16, 1, 64};
  EXPECT_TestTransposeInsertedByFeNotTrigger<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result_vec);
}
