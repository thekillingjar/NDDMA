/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_with_broadcast_eliminate_pass.h"
#include "common/checker.h"
#include "debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_reg.h"
#include "graph/ge_tensor.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace {
constexpr const char *const kTranspose = "Transpose";
constexpr const char *const kTransposeD = "TransposeD";
constexpr const char *const kZerosLike = "ZerosLike";
constexpr const char *const kFill = "Fill";
constexpr const char *const kBroadcastTo = "BroadcastTo";
constexpr const char *const kConstant = "Constant";
constexpr const char *const kValueAttrName = "value";

// 创建 Constant 节点
NodePtr CreateConstantNode(const ComputeGraphPtr &graph, const std::string &name,
                           const GeShape &shape, DataType dtype,
                           const uint8_t *data, size_t data_size) {
  auto const_op = ge::OperatorFactory::CreateOperator(name.c_str(), kConstant);
  const_op.BreakConnect();
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(const_op);
  GE_ASSERT_NOTNULL(op_desc);

  auto tensor_desc = op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(tensor_desc);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetDataType(dtype);
  tensor_desc->SetOriginDataType(dtype);
  tensor_desc->SetFormat(FORMAT_ND);
  tensor_desc->SetOriginFormat(FORMAT_ND);

  GeTensorPtr tensor = ge::ComGraphMakeShared<GeTensor>(*tensor_desc, data, data_size);
  GE_ASSERT_NOTNULL(tensor);
  GE_ASSERT_TRUE(AttrUtils::SetTensor(op_desc, kValueAttrName, tensor));

  return graph->AddNode(op_desc);
}

// 创建 shape 常量节点（使用 INT64 以支持大模型场景）
NodePtr CreateShapeConstantNode(const ComputeGraphPtr &graph, const std::string &name,
                                const GeShape &shape) {
  const auto &dims = shape.GetDims();
  std::vector<int64_t> shape_int64(dims.begin(), dims.end());
  return CreateConstantNode(graph, name + "_const_shape",
                            GeShape({static_cast<int64_t>(shape_int64.size())}),
                            DT_INT64,
                            reinterpret_cast<const uint8_t *>(shape_int64.data()),
                            shape_int64.size() * sizeof(int64_t));
}

// 创建 BroadcastTo 节点
NodePtr CreateBroadcastToNode(const ComputeGraphPtr &graph, const std::string &name,
                              const NodePtr &value_node, const NodePtr &shape_node,
                              const GeTensorDesc &output_desc) {
  auto op_desc = std::make_shared<ge::OpDesc>(name, kBroadcastTo);
  GE_ASSERT_NOTNULL(op_desc);

  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(0U, value_node->GetOpDesc()->GetOutputDesc(0U)));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(1U, shape_node->GetOpDesc()->GetOutputDesc(0U)));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc("y", output_desc));

  auto broadcast_to_node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(broadcast_to_node);
  GE_ASSERT_GRAPH_SUCCESS(
    GraphUtils::AddEdge(value_node->GetOutDataAnchor(0U), broadcast_to_node->GetInDataAnchor(0U)));
  GE_ASSERT_GRAPH_SUCCESS(
    GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0U), broadcast_to_node->GetInDataAnchor(1U)));

  return broadcast_to_node;
}

// 检查 value 是否为标量（空 shape 或单元素且值为 1）
bool IsScalarValue(const GeShape &value_shape) {
  const auto &dims = value_shape.GetDims();
  return dims.empty() || (dims.size() == 1UL && dims[0] == 1);
}

// 获取或创建 value 节点
// 对于 ZerosLike，创建值为 0 的常量；对于 Fill，直接返回其 value 输入
graphStatus GetOrCreateValueNode(const ComputeGraphPtr &graph,
                                 const NodePtr &broadcast_op_node,
                                 const std::string &op_type,
                                 NodePtr &value_node) {
  if (op_type == kZerosLike) {
    const auto &dtype = broadcast_op_node->GetOpDesc()->GetOutputDesc(0U).GetDataType();
    const std::string &base_name = broadcast_op_node->GetName();

    auto dtype_size = GetSizeByDataType(dtype);
    if (dtype_size <= 0) {
      GELOGW("Invalid dtype %d for ZerosLike node %s", dtype, base_name.c_str());
      return GRAPH_FAILED;
    }

    std::vector<uint8_t> zero_data(dtype_size, 0);
    value_node = CreateConstantNode(graph, base_name + "_const_value",
                                    GeShape(), dtype,
                                    zero_data.data(), zero_data.size());
    GE_ASSERT_NOTNULL(value_node);
    return GRAPH_SUCCESS;
  }

  if (op_type == kFill) {
    const auto &inputs = broadcast_op_node->GetInDataNodes();
    if (inputs.size() < 2UL) {
      return GRAPH_FAILED;
    }
    value_node = inputs.at(1U);
    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}

// 通用的替换函数：将 BroadcastOp + Transpose 替换为 value + BroadcastTo
graphStatus ReplaceWithBroadcastTo(const ComputeGraphPtr &graph,
                                   const NodePtr &transpose_node,
                                   const NodePtr &broadcast_op_node,
                                   const std::string &op_type,
                                   bool &changed) {
  // 1. 获取或创建 value 节点
  NodePtr value_node = nullptr;
  auto status = GetOrCreateValueNode(graph, broadcast_op_node, op_type, value_node);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }

  // 2. 对于 Fill，检查 value 是否为标量
  if (op_type == kFill && !IsScalarValue(value_node->GetOpDesc()->GetOutputDesc(0U).GetShape())) {
    return GRAPH_SUCCESS;
  }

  // 3. 缓存 transpose_node 的输出描述
  const auto &transpose_output_desc = transpose_node->GetOpDesc()->GetOutputDesc(0U);
  const std::string &base_name = broadcast_op_node->GetName();

  // 4. 创建 shape 常量节点和 BroadcastTo 节点
  auto shape_node = CreateShapeConstantNode(graph, base_name, transpose_output_desc.GetShape());
  GE_ASSERT_NOTNULL(shape_node);

  auto broadcast_to_node = CreateBroadcastToNode(graph, base_name + "_broadcast_to",
                                                 value_node, shape_node, transpose_output_desc);
  GE_ASSERT_NOTNULL(broadcast_to_node);

  // 5. 替换节点并删除旧节点
  GE_ASSERT_SUCCESS(GraphUtils::ReplaceNodeAnchors(broadcast_to_node, transpose_node, {}, {0U}),
                    "Failed to replace anchors for transpose node %s", transpose_node->GetNamePtr());
  GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(transpose_node, {0U}),
                    "Failed to isolate transpose node %s", transpose_node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, transpose_node),
                          "Failed to remove transpose node %s", transpose_node->GetNamePtr());
  GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(broadcast_op_node, {0U}),
                    "Failed to isolate broadcast op node %s", broadcast_op_node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, broadcast_op_node),
                          "Failed to remove broadcast op node %s", broadcast_op_node->GetNamePtr());

  GELOGD("TransposeWithBroadcastEliminatePass: replaced %s + Transpose[%s] with BroadcastTo",
         op_type.c_str(), transpose_node->GetNamePtr());

  changed = true;
  return GRAPH_SUCCESS;
}

bool CanOptimize(const NodePtr &node) {
  return node->GetOutDataNodesSize() == 1UL;
}
} // namespace

graphStatus TransposeWithBroadcastEliminatePass::Run(const ComputeGraphPtr &graph, bool &changed) {
  GE_ASSERT_NOTNULL(graph);

  // 收集并处理所有 Transpose 节点
  for (const auto &node: graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    const auto &type = node->GetType();

    if (type != kTranspose && type != kTransposeD) {
      continue;
    }

    auto input_nodes = node->GetInDataNodes();
    if (input_nodes.empty()) {
      continue;
    }

    const auto &input_node = input_nodes.at(0U);
    GE_ASSERT_NOTNULL(input_node);
    if (!CanOptimize(input_node)) {
      continue;
    }

    const auto &input_type = input_node->GetType();
    if (input_type != kZerosLike && input_type != kFill) {
      continue; // 只支持 ZerosLike 和 Fill
    }

    GE_ASSERT_GRAPH_SUCCESS(ReplaceWithBroadcastTo(graph, node, input_node, input_type, changed),
                            "Failed to process transpose node %s with input type %s",
                            node->GetNamePtr(), input_type.c_str());
  }

  return GRAPH_SUCCESS;
}
} // namespace ge
