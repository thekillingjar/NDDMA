/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder_utils.h"

#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace ut {
NodePtr GraphBuilder::AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt, Format format,
                              DataType data_type, std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(std::move(shape)));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  // to make ut pass. set default infer func.
  op_desc->AddInferFunc([](Operator &op) {return GRAPH_SUCCESS;});
  return graph_->AddNode(op_desc);
}

NodePtr GraphBuilder::AddNodeByIr(const std::string &op_name, const std::string &op_type) const {
  auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
  if (op.IsEmpty()) {
    return nullptr;
  }
  OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  return graph_->AddNode(op_desc);
}

void GraphBuilder::AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
}
void GraphBuilder::AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
  GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
}
void GraphBuilder::AddPartitionedCall(const ComputeGraphPtr &graph, const std::string &call_name, const ComputeGraphPtr &subgraph) {
  const auto &call_node = graph->FindNode(call_name);
  if (call_node ==nullptr) {
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
}  // namespace ut
}  // namespace ge
