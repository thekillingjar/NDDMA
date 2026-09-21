/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trust_out_tensor.h"

#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/graph_dump_utils.h"

#include "core/builder/node_types.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "kernel/outputs/model_outputs.h"
#include "utils/pruner.h"
#include "lowering/pass_changed_kernels_info.h"
#include "exe_graph/lowering/exe_graph_attrs.h"

namespace gert {
namespace bg {
namespace {
bool IsSameOutput(const ge::EdgeSrcEndpoint &endpoint1, const ge::EdgeSrcEndpoint &endpoint2) {
  return endpoint1 == endpoint2;
}
bool IsCopyNode(const ge::FastNode *node) {
  return node->GetType() == kernel::kEnsureTensorAtOutMemory;
}
std::vector<ge::FastNode *> FindAllCopyNodes(const ge::FastNode *output_data_node,
                                             std::map<ge::FastNode *, int32_t> &copy_nodes_to_model_out_index) {
  std::vector<ge::FastNode *> copy_nodes;
  copy_nodes.reserve(output_data_node->GetDataOutNum());
  // 获取 copy_nodes 和 copy_nodes 的输入数据边的 src_index (即 OutputData 的对应输出 index)
  for (const auto &out_data_edges : output_data_node->GetAllOutDataEdgesRef()) {
    for (const auto out_data_edge : out_data_edges) {
      if (out_data_edge != nullptr) {
        const auto out_node = out_data_edge->dst;
        if (IsCopyNode(out_node)) {
          // 记录 copy_node，同时记录 copy_node 当前输入数据边的对端的输出 index
          copy_nodes.emplace_back(out_node);
          copy_nodes_to_model_out_index[out_node] = out_data_edge->src_output;
          continue;
        }
      }
    }
  }

  return copy_nodes;
}
ge::FastNode *FindInferShapeNode(const ge::FastNode *copy_node) {
  ge::FastNode *in_node = nullptr;
  if (copy_node->GetType() == kernel::kEnsureTensorAtOutMemory) {
    in_node =
        ge::FastNodeUtils::GetInDataNodeByIndex(copy_node, static_cast<int32_t>(kernel::BuildTensorInputs::kShape));
  }

  if (in_node == nullptr) {
    return nullptr;
  }

  if (!IsInferShapeNode(in_node->GetTypePtr())) {
    return nullptr;
  }

  return in_node;
}
bool CanBeDeleted(const ge::FastNode *infer_shape_node,
                  const std::map<ge::FastNode *, int32_t> &copy_nodes_to_model_out_index,
                  std::vector<int32_t> &infer_node_out_indexes_to_model_out_index) {
  // 逐个遍历 infer_shape_node 的输出，判断第 output_idx 个输出下是否包含 copy node
  for (int32_t output_idx = 0; static_cast<size_t>(output_idx) < infer_shape_node->GetDataOutNum(); ++output_idx) {
    // 如果某个输出没被使用，那么有没有输出 shape 都不影响结果，这种输出即使没有对应的 copy node 也没关系
    const size_t output_size = infer_shape_node->GetOutEdgesSizeByIndex(output_idx);
    if (output_size == 0) {
      infer_node_out_indexes_to_model_out_index[output_idx] = -1;
      continue;
    }

    bool has_copy_node = false;
    for (const auto out_data_edge : infer_shape_node->GetOutEdgesRefByIndex(output_idx)) {
      if (out_data_edge == nullptr) {
        continue;
      }
      auto dst_node = out_data_edge->dst;
      GE_ASSERT_NOTNULL(dst_node);
      if (!IsCopyNode(dst_node)) {
        continue;
      }
      auto iter = copy_nodes_to_model_out_index.find(dst_node);
      if (iter == copy_nodes_to_model_out_index.cend()) {
        GELOGW("Find Copy node %s, but which does not the model out copy node", dst_node->GetNamePtr());
        continue;
      }
      // 记录 infer_shape_node 连接
      GE_ASSERT_TRUE(infer_node_out_indexes_to_model_out_index.size() > static_cast<size_t>(output_idx));
      infer_node_out_indexes_to_model_out_index[output_idx] = iter->second;
      has_copy_node = true;
    }
    if (!has_copy_node) {
      return false;
    }
  }
  return true;
}

/* 该接口为删除InferShape节点作准备，会将OutputData连到InferShape节点的每个输出上，然后剪枝。
 *                           NetOutput                                            NetOutput
 *                            /c      \                                            /c      \
 *             EnsureAtUserMemory      \                            EnsureAtUserMemory      \
 *           /      |       |    \      \                         /      |       |    \      \
 * InferShape tensor-data attrs stream  OutputData     =>        | tensor-data attrs stream   |
 *     |                                                          \                           |
 * SomeNodes                                                       \                          |
 *                                                                  +-------------------- OutputData
 */
ge::graphStatus PrepareDeleteInferShapeNodes(ge::FastNode *output_data_node, ge::FastNode *copy_node,
                                             const std::map<ge::FastNode *, int32_t> &copy_nodes_to_model_out_index,
                                             std::vector<ge::FastNode *> &delete_nodes) {
  auto infer_shape_node = FindInferShapeNode(copy_node);
  if (infer_shape_node == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  std::vector<int32_t> node_out_indexes_to_model_out_index(infer_shape_node->GetDataOutNum(), -1);
  if (!CanBeDeleted(infer_shape_node, copy_nodes_to_model_out_index, node_out_indexes_to_model_out_index)) {
    return ge::GRAPH_SUCCESS;
  }
  PassChangedKernels pass_changed_kernels{};
  for (int32_t node_index = 0; static_cast<size_t>(node_index) < node_out_indexes_to_model_out_index.size();
       ++node_index) {
    auto model_index = node_out_indexes_to_model_out_index.at(node_index);
    if (model_index < 0) {
      continue;
    }

    // 将 output_data_node 第 model_index 个输出连接到 infer_shape_node 第 node_idx 个输出的每个 OutData 节点上
    auto owner_graph = output_data_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    for (const auto out_data_edge : infer_shape_node->GetOutEdgesRefByIndex(node_index)) {
      if (out_data_edge != nullptr) {
        auto dst_node = out_data_edge->dst;
        auto dst_index = out_data_edge->dst_input;
        GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(out_data_edge));
        GE_ASSERT_NOTNULL(owner_graph->AddEdge(output_data_node, model_index, dst_node, dst_index),
                          "Link Node[%s]:Idx[%d] to Node[%s]:Idx[%d] failed.", output_data_node->GetNamePtr(),
                          model_index, dst_node->GetNamePtr(), dst_index);
        pass_changed_kernels.pass_changed_kernels.emplace_back(std::pair<KernelNameAndIdx, KernelNameAndIdx>(
            {infer_shape_node->GetName(), node_index}, {output_data_node->GetName(), model_index}));
      }
    }
    pass_changed_kernels.pass_changed_kernels.emplace_back(std::pair<KernelNameAndIdx, KernelNameAndIdx>(
        {infer_shape_node->GetName(), node_index}, {output_data_node->GetName(), model_index}));
  }
  const auto op_desc = infer_shape_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  (void)op_desc->SetExtAttr(kPassChangedInfo, pass_changed_kernels);
  delete_nodes.emplace_back(infer_shape_node);

  return ge::GRAPH_SUCCESS;
}
/*
 * 本接口找到冗余的size计算和shape更新Node，为剪枝做准备
 *
 *                              NetOutput
 *                               /c    ^
 *      + ------->  EnsureAtUserMemory |
 *      |             |   |     |    \ |
 *      |             |  attrs stream \|                                 SomeNodes
 *      |             |                +                                     |     c
 *      |             |      SomeNodes |       =>        SplitTensorForOutputData ----> NetOutput
 *      |             |          |     |                     c|                \         /
 *      |    SplitTensorForOutputData  |                   allocator            OutputData
 *      |      |     |      c|         |
 *      |      | CalcSize allocator    |
 *      |      |     |                 |
 *      +------+-----+-- OutputData ---+
 *
 * size计算Node删除条件，如果同一个输出同时：
 * 1. 连给了SplitTensorForOutputData的Tensor输入
 * 2. 通过CalcSize连给了同一个SplitTensorForOutputData的Size输入
 * 那么说明CalcSize的计算是冗余的，将其删除
 *
 * size更新Node删除条件（EnsureAtUserMemory），如果EnsureAtUserMemory前面接了SplitTensorForOutputData，
 * 并且InferShape被换成OutputData，那么EnsureAtUserMemory的两个作用就都不存在了（分别是兜底申请输出内存、更新输出shape），
 * 因此EnsureAtUserMemory可以被删除
 */
ge::graphStatus PrepareDeleteSizeNodes(ge::FastNode *output_data_node, ge::FastNode *copy_node,
                                       std::vector<ge::FastNode *> &delete_nodes) {
  // 首先确认Ensure操作的是输出数据，理论上必然是这样的
  auto copy_in_edge_from_output =
      copy_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::EnsureTensorAtOutMemoryInputs::kOutputData));
  GE_ASSERT_NOTNULL(copy_in_edge_from_output);
  auto copy_in_edge_src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(copy_in_edge_from_output);
  if (copy_in_edge_src_endpoint.node != output_data_node) {
    GE_ASSERT_NOTNULL(copy_in_edge_src_endpoint.node);
    GELOGW("The copy node %s input %d does not connect to OutputData, %s instead", copy_node->GetNamePtr(),
           static_cast<int32_t>(kernel::EnsureTensorAtOutMemoryInputs::kOutputData),
           copy_in_edge_src_endpoint.node->GetNamePtr());
    return ge::GRAPH_SUCCESS;
  }

  // 找到 SplitTensorForOutputData，并确认该节点操作的与 Ensure 节点操作的是同一个输出（即上图 OutputData）
  // 必须同时校验 node 和 index，唯一确定同一个节点的同一个输出/输入
  auto split_node =
      ge::FastNodeUtils::GetInDataNodeByIndex(copy_node, static_cast<int32_t>(kernel::BuildTensorInputs::kTensorData));
  GE_ASSERT_NOTNULL(split_node);
  if (split_node->GetType() != kernel::kSplitTensorForOutputData) {
    return ge::GRAPH_SUCCESS;
  }
  auto split_in_edge_of_tensor = split_node->GetInDataEdgeByIndex(0);
  GE_ASSERT_NOTNULL(split_in_edge_of_tensor);
  auto split_in_edge_src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(split_in_edge_of_tensor);
  if (!IsSameOutput(split_in_edge_src_endpoint, copy_in_edge_src_endpoint)) {
    return ge::GRAPH_SUCCESS;
  }

  // 判断是否可以删除Ensure，ensure_node 和 copy_node 使用的同一个输出则可删
  auto copy_in_edge_of_shape = copy_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::BuildTensorInputs::kShape));
  GE_ASSERT_NOTNULL(copy_in_edge_of_shape);
  auto shape_edge_src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(copy_in_edge_of_shape);
  if (IsSameOutput(shape_edge_src_endpoint, copy_in_edge_src_endpoint)) {
    delete_nodes.emplace_back(copy_node);
  }

  // 判断是否可以删除calc_size
  if (split_node->GetAllInDataEdgesSize() <= static_cast<size_t>(kernel::SplitTensorInputs::kSize)) {
    // 此处为确保可重入，copy_node可能重复，此时 split_node 和 calc_node 之间的连边
    // 会在第一次调用本函数时被删除，同时 split_node 的输入数量相应减少
    return ge::GRAPH_SUCCESS;
  }
  auto split_in_edge_of_size = split_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::SplitTensorInputs::kSize));
  if (split_in_edge_of_size == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto calc_size_node = split_in_edge_of_size->src;
  auto calc_size_src_index = split_in_edge_of_size->src_output;
  if (calc_size_node == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (!IsCalcSizeNode(calc_size_node->GetTypePtr())) {
    return ge::GRAPH_SUCCESS;
  }
  if (calc_size_node->GetOutEdgesSizeByIndex(calc_size_src_index) > 1U) {
    return ge::GRAPH_SUCCESS;
  }
  auto calc_in_edge = calc_size_node->GetInDataEdgeByIndex(1);
  GE_ASSERT_NOTNULL(calc_in_edge);
  auto calc_in_edge_src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(calc_in_edge);
  if (IsSameOutput(calc_in_edge_src_endpoint, copy_in_edge_src_endpoint)) {
    // 删除 calc_size 的所有输出数据边，连接 calc_size 到 split_node 的控制边，等待被剪枝
    auto owner_graph = calc_size_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    for (auto out_edge : calc_size_node->GetOutEdgesByIndex(calc_size_src_index)) {
      if (out_edge != nullptr) {
        GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(out_edge));
      }
    }
    GE_ASSERT_NOTNULL(owner_graph->AddEdge(calc_size_node, ge::kControlEdgeIndex, split_node, ge::kControlEdgeIndex));
    GE_ASSERT_GRAPH_SUCCESS(
        ge::FastNodeUtils::RemoveInputEdgeInfo(split_node, static_cast<uint32_t>(kernel::SplitTensorInputs::kSize)));
    delete_nodes.emplace_back(calc_size_node);
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace
ge::graphStatus TrustOutTensor::Run(ge::ExecuteGraph *const graph, bool &changed) {
  if (!GetLoweringOption().trust_shape_on_out_tensor) {
    return ge::GRAPH_SUCCESS;
  }
  // todo 遗留事项，实现返回首个符合 Type 的节点，避免遍历全图节点
  auto output_data_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph, "OutputData");
  GE_ASSERT_TRUE(!output_data_nodes.empty());
  auto output_data_node = output_data_nodes.at(0U);
  GE_ASSERT_NOTNULL(output_data_node);

  std::vector<ge::FastNode *> prune_nodes;
  std::map<ge::FastNode *, int32_t> copy_nodes_to_model_out_index;
  auto copy_nodes = FindAllCopyNodes(output_data_node, copy_nodes_to_model_out_index);
  for (const auto copy_node : copy_nodes) {
    GE_ASSERT_SUCCESS(
        PrepareDeleteInferShapeNodes(output_data_node, copy_node, copy_nodes_to_model_out_index, prune_nodes));
    GE_ASSERT_SUCCESS(PrepareDeleteSizeNodes(output_data_node, copy_node, prune_nodes));
  }
  auto ret = Pruner().StartNodesMustBeDeleted().PruneFromNodes(prune_nodes, changed);
  if (changed) {
    ge::DumpGraph(graph, "AfterTrustOutTensor");
  }
  return ret;
}
}  // namespace bg
}  // namespace gert
