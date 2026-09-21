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
#include <nlohmann/json.hpp>
#include <mutex>
#include "graph/op_kernel_bin.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/scope_allocator.h"
#include "graph_optimizer/weight_prefetch/weight_prefetch_utils.h"
#include "graph_optimizer/ub_fusion/fusion_graph_merge/ub_fusion_graph_merge.h"

using namespace ge;
namespace fe {
class WeightPrefetchST : public testing::Test {
 protected:
  void SetUp() {
    GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(AI_CORE_NAME);
    graph_comm_ptr->Initialize();
    ub_fusion_graph_merge_ptr_ = std::make_shared<UBFusionGraphMerge>(SCOPE_ID_ATTR, graph_comm_ptr);
  }
  std::shared_ptr<FusionGraphMerge> ub_fusion_graph_merge_ptr_;

 protected:
  void AddInputConst(const ge::NodePtr &node, const int64_t offset, const uint32_t weight_size) {
    vector<int32_t> data(24, 5);
    vector<int64_t> dims = {1, 2, 3, 4};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_INT32);
    ge::TensorUtils::SetDataOffset(tensor_desc, offset);
    ge::TensorUtils::SetWeightSize(tensor_desc, weight_size);
    ge::GeTensorPtr filter = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data.data(), 24 * sizeof(int32_t));
    OpDescPtr  const_opdesc = OpDescUtils::CreateConstOp(filter);
    ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
    ge::NodePtr const_node = owner_graph->AddNode(const_opdesc);
    node->AddLinkFrom(const_node);
  }

  ge::ComputeGraphPtr CreateGraphWithGraph(const uint32_t case_type) {
    vector<int64_t> dims = {3,4,5,6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

    OpDescPtr quant_op = std::make_shared<OpDesc>("quant", "AscendQuant");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr dequant_op = std::make_shared<OpDesc>("dequant", "AscendDequant");
    OpDescPtr transpose_op = std::make_shared<OpDesc>("transpose", "Transpose");

    quant_op->AddInputDesc(tensor_desc);
    quant_op->AddOutputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddOutputDesc(tensor_desc);
    relu_op->AddInputDesc(tensor_desc);
    relu_op->AddOutputDesc(tensor_desc);
    dequant_op->AddInputDesc(tensor_desc);
    dequant_op->AddOutputDesc(tensor_desc);
    transpose_op->AddInputDesc(tensor_desc);
    transpose_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetInt(quant_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(dequant_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(transpose_op, "_fe_imply_type", 6);

    const char tbe_bin[] = "tbe_bin";
    std::vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>("name", std::move(buffer));
    quant_op->SetExtAttr("tbeKernel", tbe_kernel_ptr);
    conv_op->SetExtAttr("tbeKernel", tbe_kernel_ptr);
    relu_op->SetExtAttr("tbeKernel", tbe_kernel_ptr);
    dequant_op->SetExtAttr("tbeKernel", tbe_kernel_ptr);
    transpose_op->SetExtAttr("tbeKernel", tbe_kernel_ptr);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr quant_node = graph->AddNode(quant_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr dequant_node = graph->AddNode(dequant_op);
    NodePtr transpose_node = graph->AddNode(transpose_op);
    GraphUtils::AddEdge(quant_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), dequant_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));

    AddInputConst(conv_node, 100, 16);
    AddInputConst(conv_node, 200, 0);
    AddInputConst(dequant_node, 300, 25);
    AddInputConst(transpose_node, 400, 94);

    int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
    switch (case_type) {
      case 1:
        AttrUtils::SetBool(graph, "_is_graph_prefetch", true);
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchType, {"1", "1", "1", "1"});
        AttrUtils::SetListInt(quant_op, kAttrWeightPrefetchDstOffset, {10,20,30,40});
        break;
      case 2:
        AttrUtils::SetBool(graph, "_is_graph_prefetch", true);
        ScopeAllocator::SetScopeAttr(quant_op, scope_id);
        ScopeAllocator::SetScopeAttr(conv_op, scope_id);
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchType, {"1", "1", "1", "1"});
        AttrUtils::SetListInt(quant_op, kAttrWeightPrefetchDstOffset, {10,20,30,40});
        break;
      case 3:
        // quant -> conv, relu->dequant, dequant->transpose
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchType, {"1", "1"});
        AttrUtils::SetListInt(quant_op, kAttrWeightPrefetchDstOffset, {10,20});
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchNodeName, {conv_op->GetName()});

        AttrUtils::SetListStr(relu_op, kAttrWeightPrefetchType, {"1"});
        AttrUtils::SetListInt(relu_op, kAttrWeightPrefetchDstOffset, {30});
        AttrUtils::SetListStr(relu_op, kAttrWeightPrefetchNodeName, {dequant_node->GetName()});

        AttrUtils::SetListStr(dequant_op, kAttrWeightPrefetchType, {"1"});
        AttrUtils::SetListInt(dequant_op, kAttrWeightPrefetchDstOffset, {40});
        AttrUtils::SetListStr(dequant_op, kAttrWeightPrefetchNodeName, {transpose_op->GetName()});
        break;
      case 4:
        // quant -> (conv+relu+dequant), (conv+relu+dequant) -> transpose
        ScopeAllocator::SetScopeAttr(conv_op, scope_id);
        ScopeAllocator::SetScopeAttr(relu_op, scope_id);
        ScopeAllocator::SetScopeAttr(dequant_op, scope_id);
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchType, {"1", "1", "1"});
        AttrUtils::SetListInt(quant_op, kAttrWeightPrefetchDstOffset, {10,20,30});
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchNodeName, {conv_op->GetName(), dequant_op->GetName()});

        AttrUtils::SetListStr(conv_op, kAttrWeightPrefetchType, {"1"});
        AttrUtils::SetListInt(conv_op, kAttrWeightPrefetchDstOffset, {40});
        AttrUtils::SetListStr(conv_op, kAttrWeightPrefetchNodeName, {transpose_op->GetName()});
        break;
      case 5:
        // quant -> (conv+relu+dequant), (conv+relu+dequant) -> transpose
        ScopeAllocator::SetScopeAttr(conv_op, scope_id);
        ScopeAllocator::SetScopeAttr(relu_op, scope_id);
        ScopeAllocator::SetScopeAttr(dequant_op, scope_id);
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchType, {"1", "1", "1"});
        AttrUtils::SetListInt(quant_op, kAttrWeightPrefetchDstOffset, {10,20,30});
        AttrUtils::SetListStr(quant_op, kAttrWeightPrefetchNodeName, {conv_op->GetName(), dequant_op->GetName()});

        AttrUtils::SetListStr(conv_op, kAttrWeightPrefetchType, {"1"});
        AttrUtils::SetListInt(conv_op, kAttrWeightPrefetchDstOffset, {40});
        AttrUtils::SetListStr(conv_op, kAttrWeightPrefetchNodeName, {transpose_op->GetName()});
        AttrUtils::SetListStr(relu_op, kAttrWeightPrefetchType, {"1"});
        AttrUtils::SetListInt(relu_op, kAttrWeightPrefetchDstOffset, {40});
        AttrUtils::SetListStr(relu_op, kAttrWeightPrefetchNodeName, {transpose_op->GetName()});
        AttrUtils::SetListStr(dequant_op, kAttrWeightPrefetchType, {"1"});
        AttrUtils::SetListInt(dequant_op, kAttrWeightPrefetchDstOffset, {40});
        AttrUtils::SetListStr(dequant_op, kAttrWeightPrefetchNodeName, {transpose_op->GetName()});
      default:
        break;
    }
    return graph;
  }
};

TEST_F(WeightPrefetchST, weight_prefetch_case1) {
  ComputeGraphPtr graph = CreateGraphWithGraph(1);
  Status ret = WeightPrefetchUtils::HandleWeightPrefetch(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);

  for (const NodePtr &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "AscendQuant") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {100, 200, 300, 400};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {16, 0, 28, 96};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1", "1", "1", "1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {10,20,30,40};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
  }
}

TEST_F(WeightPrefetchST, weight_prefetch_case2) {
  ComputeGraphPtr graph = CreateGraphWithGraph(2);
  Status ret = WeightPrefetchUtils::HandleWeightPrefetch(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);

  for (const NodePtr &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "AscendQuant") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {100, 200, 300, 400};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {16, 0, 28, 96};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1", "1", "1", "1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {10,20,30,40};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
  }
}

TEST_F(WeightPrefetchST, weight_prefetch_case3) {
  ComputeGraphPtr graph = CreateGraphWithGraph(3);
  Status ret = WeightPrefetchUtils::HandleWeightPrefetch(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);

  for (const NodePtr &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "AscendQuant") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {100, 200};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {16, 0};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1", "1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {10,20};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
    if (op_desc->GetType() == "Relu") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {300};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {28};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {30};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
    if (op_desc->GetType() == "AscendDequant") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {400};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {96};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {40};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
  }
}

TEST_F(WeightPrefetchST, weight_prefetch_case4) {
  ComputeGraphPtr graph = CreateGraphWithGraph(4);
  Status ret = WeightPrefetchUtils::HandleWeightPrefetch(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);

  for (const NodePtr &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "AscendQuant") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {100, 200, 300};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {16, 0, 28};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1", "1", "1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {10,20,30};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
    if (op_desc->GetType() == "Conv2D") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {400};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {96};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {40};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
  }
}

TEST_F(WeightPrefetchST, weight_prefetch_case5) {
  ComputeGraphPtr graph = CreateGraphWithGraph(5);
  Status ret = WeightPrefetchUtils::HandleWeightPrefetch(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);

  for (const NodePtr &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "AscendQuant") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {100, 200, 300};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {16, 0, 28};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1", "1", "1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {10,20,30};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
    if (op_desc->GetType() == "Conv2D") {
      vector<int64_t> src_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, src_offset);
      vector<int64_t> exp_src_offset = {400};
      EXPECT_EQ(src_offset, exp_src_offset);

      vector<int64_t> data_size;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, data_size);
      vector<int64_t> exp_data_size = {96};
      EXPECT_EQ(data_size, exp_data_size);

      vector<string> prefetch_type;
      AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, prefetch_type);
      vector<string> exp_prefetch_type = {"1"};
      EXPECT_EQ(prefetch_type, exp_prefetch_type);

      vector<int64_t> dst_offset;
      AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, dst_offset);
      vector<int64_t> exp_dst_offset = {40};
      EXPECT_EQ(dst_offset, exp_dst_offset);
    }
  }
}
}
