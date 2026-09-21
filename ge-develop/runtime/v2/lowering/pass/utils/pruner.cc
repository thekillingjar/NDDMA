/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pruner.h"

#include <queue>
#include <set>

#include "common/checker.h"
#include "resource_guarder.h"
#include "deduplicate_queue.h"
#include "core/builder/node_types.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/mem_utils.h"

namespace gert {
namespace bg {
namespace {
// todo 白名单判断比较粗暴，未来考虑更通用的做法，例如可以将kernel分为几类：
//       * 标准无状态kernel：写入输出，除此之外，不做任何其他影响到本Kernel之外的事情
//       * 写输入kernel：写入输入，除此之外，不做任何其他影响到本Kernel之外的事情
//       * 变更外部资源的kernel：例如写入了一个全局变量、操作了Device（流同步、launch等）
//       * 结构必备kernel：Init、Main、DeInit，控制算子子图中的那堆node，Data、OutputData
//       * 带子图的Node，因为不确定子图里面做了多少事情，简单处理，不做剪枝
//       上述几种kernel为黑名单kernel，剩余的kernel都可以被删除
std::set<std::string> g_prune_nodes_white_list = {
    "SelectL1Allocator",
    "CalcTensorSizeFromShape",
    "CalcTensorSizeFromStorage",
    "CalcUnalignedTensorSizeFromStorage",
    "SplitTensor",
    "Const",
    "InferShape",
    "CompatibleInferShape",
    "EnsureTensorAtOutMemory",
    "InnerData",
};
std::set<std::string> g_resource_alloc_list = {"AllocBatchHbm", "AllocMemHbm", "AllocMemHost", "AllocMemory"};
std::set<std::string> g_init_list = {"FindInferShapeFunc", "FindInferShapeRangeFunc", "FindCompatibleInferShapeFunc"};

bool IsResourceAllocNode(const ge::FastNode *const node) {
  return g_resource_alloc_list.count(node->GetType()) > 0U;
}

bool CanResourceAllocBeDeleted(const ge::FastNode *const alloc_node) {
  for (const auto out_node : alloc_node->GetOutDataNodes()) {
    if (!IsGuarderOf(alloc_node, out_node)) {
      return false;
    }
  }
  return true;
}

bool CanNodeBeDeleted(bool is_resource_alloc, const ge::FastNode *const node) {
  if (is_resource_alloc) {
    return CanResourceAllocBeDeleted(node);
  }
  if (g_init_list.count(node->GetType()) > 0U && node->GetAllOutDataEdgesSize() == 1U) {
    return true;
  }
  if (node->GetAllOutDataEdgesSize() > 0U) {
    return false;
  }
  if (g_prune_nodes_white_list.count(node->GetType()) > 0U) {
    return true;
  }
  return false;
}

ge::graphStatus DeleteResourceAllocAndFreeNodes(ge::FastNode *const alloc_node) {
  // alloc_node is created by ValueHolder::CreateNode, ExtendInfo is guaranteed to be non-null
  auto owner_graph = alloc_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  const auto out_data_nodes = alloc_node->GetOutDataNodes();
  for (auto guard_node : out_data_nodes) {
    GELOGI("Delete successor guarder node %s of alloc node %s by pruner", guard_node->GetNamePtr(),
           alloc_node->GetNamePtr());
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(guard_node, {}));
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(owner_graph, guard_node));
  }
  GELOGI("Delete alloc node %s by pruner", alloc_node->GetNamePtr());
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(alloc_node, {}));
  return ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(owner_graph, alloc_node);
}

ge::graphStatus DeleteNode(bool is_resource_alloc, ge::FastNode *const node) {
  if (is_resource_alloc) {
    return DeleteResourceAllocAndFreeNodes(node);
  } else {
    GELOGI("Delete node %s by pruner", node->GetNamePtr());
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(node, {}));
    auto owner_graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
    return ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(owner_graph, node);
  }
}

bool IsComputableOp(const ge::FastNode *const node) {
  if (IsInnerDataType(node->GetTypePtr()) || (strcmp(node->GetTypePtr(), ge::DATA) == 0) ||
      (strcmp(node->GetTypePtr(), ge::NETOUTPUT) == 0)) {
    return false;
  }
  return node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty();
}

ge::EdgeSrcEndpoint GetParentInputSrcEndpoint(const ge::FastNode *const node) {
  uint32_t parent_index = 0U;
  if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_INDEX, parent_index)) {
    return {nullptr, ge::kInvalidEdgeIndex};
  }
  // get the parent node of subgraph Data Node, check for constant input
  const auto extend_info = node->GetExtendInfo();
  // owner_graph can never be null
  const auto owner_graph = extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  const auto parent_node = owner_graph->GetParentNodeBarePtr();
  GE_WARN_ASSERT(parent_node != nullptr, "parent node of Graph[%s] is null", owner_graph->GetName().c_str());
  const auto in_edge = parent_node->GetInDataEdgeByIndex(parent_index);
  GE_WARN_ASSERT(in_edge != nullptr, "the %u-th InDataEdge of Node[%s] is nullptr", parent_index,
                 parent_node->GetNamePtr());
  const ge::EdgeSrcEndpoint src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(in_edge);
  GELOGD("GetParentInputSrcEndpoint success, node name %s[%s]", src_endpoint.node->GetNamePtr(),
         src_endpoint.node->GetTypePtr());
  return src_endpoint;
}

ge::graphStatus GetNodeInputFromInit(ge::FastNode *const node, uint32_t index, ge::FastNode *&peer_node) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE((IsInnerDataType(node->GetTypePtr())) && ((index <= node->GetDataInNum())));
  peer_node = node;
  GE_ASSERT_NOTNULL(peer_node);
  int32_t peer_out_src_index = -1;

  while (!IsComputableOp(peer_node)) {
    if (IsInnerDataType(peer_node->GetTypePtr())) {
      const auto parent_node_src_endpoint = GetParentInputSrcEndpoint(peer_node);
      // parent_node_src_endpoint.index must be the index of one data edge
      if ((parent_node_src_endpoint.node == nullptr) && (parent_node_src_endpoint.index < 0)) {
        GELOGW("Returned peer_out_node is nullptr because no valid attr[%s] on DATA[%s] node!",
               ge::ATTR_NAME_INDEX.c_str(), peer_node->GetNamePtr());
        peer_node = nullptr;
        return ge::GRAPH_SUCCESS;
      }
      peer_node = parent_node_src_endpoint.node;
      peer_out_src_index = parent_node_src_endpoint.index;
      continue;
    }

    if (!IsInitNode(peer_node->GetTypePtr())) {
      if (peer_node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
        GELOGI("Node [%s] type [%s], real peer in node [%s] type[%s].", node->GetNamePtr(), node->GetTypePtr(),
               peer_node->GetNamePtr(), peer_node->GetTypePtr());
        return ge::GRAPH_SUCCESS;
      }
      // other subgraph(if,while,case) currently not support, return node and warn
      GELOGW("Node [%s] type [%s], real peer in node [%s] type[%s] has subgraph. Current not support.",
             node->GetNamePtr(), node->GetTypePtr(), peer_node->GetNamePtr(), peer_node->GetTypePtr());

      return ge::GRAPH_SUCCESS;
    }

    // if peer node is PartionedCall, return owner graph's correspond node
    const auto sub_graph = ge::FastNodeUtils::GetSubgraphFromNode(peer_node, 0U);
    GE_ASSERT_NOTNULL(sub_graph);
    const auto sub_graph_netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_graph, "InnerNetOutput");
    GE_ASSERT_NOTNULL(sub_graph_netoutput);

    peer_node = ge::FastNodeUtils::GetInDataNodeByIndex(sub_graph_netoutput, peer_out_src_index);
    return ge::GRAPH_SUCCESS;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReplaceNodeWithNoOp(ge::FastNode *const old_node) {
  // old_node is guaranteed to be non-null
  std::string no_op_name = old_node->GetName() + "_Replaced_NoOp";
  auto dst_op_desc = ge::MakeShared<ge::OpDesc>(no_op_name, ge::NOOP);
  GE_ASSERT_NOTNULL(dst_op_desc);
  dst_op_desc->AddOutputDesc(ge::GeTensorDesc());
  auto exe_graph = old_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(exe_graph);
  auto no_op_node = exe_graph->AddNode(dst_op_desc);
  GE_ASSERT_NOTNULL(no_op_node);
  // 创建 NOOP 节点作为替代，将 old_node 所有的输入边全部断连, 输出边执行替换
  GE_ASSERT_GRAPH_SUCCESS(ge::ExecuteGraphUtils::ReplaceNodeEdges(no_op_node, old_node, {}, {0}));
  ge::FastNodeUtils::UnlinkAll(old_node);
  GE_ASSERT_GRAPH_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(exe_graph, old_node));
  GE_ASSERT_GRAPH_SUCCESS(no_op_node->GetExtendInfo()->SetOwnerGraph(exe_graph, no_op_node));
  GELOGD("Replace node %s[%s] with node %s[%s] success!", old_node->GetNamePtr(), old_node->GetTypePtr(),
         no_op_node->GetNamePtr(), no_op_node->GetTypePtr());
  return ge::GRAPH_SUCCESS;
}

bool IsSingleOutAndRef(const ge::FastNode *const node) {
  const auto data_out_num = node->GetDataOutNum();
  GELOGD("data out num of Node[%s] is [%zu].", node->GetNamePtr(), data_out_num);
  // check if the node has a single-output and reference of the output is less than or equal to 1
  if (data_out_num == 1U) {
    const auto out_data_edge_size = node->GetOutEdgesSizeByIndex(0U);
    GELOGD("out data edge size of Node[%s] is [%zu].", node->GetNamePtr(), out_data_edge_size);
    return out_data_edge_size <= 1U;
  }
  return false;
}

ge::graphStatus DealWithInnerData(ge::FastNode *node) {
  const auto src_endpoint = GetParentInputSrcEndpoint(node);
  GE_ASSERT_TRUE((src_endpoint.node != nullptr) && (src_endpoint.index >= 0));
  // 处理根图场景，根图的当前节点的输出属于“单输出单引用”才会入队列
  if (src_endpoint.node->GetOutEdgesSizeByIndex(src_endpoint.index) == 1U) {
    ge::FastNode *node_in_init = nullptr;
    GE_ASSERT_GRAPH_SUCCESS(GetNodeInputFromInit(node, 0, node_in_init));
    if (IsSingleOutAndRef(node_in_init)) {
      GELOGD("Prepare delete node %s[%s] in init", node_in_init->GetNamePtr(), node_in_init->GetTypePtr());
      // 此时并不删除init图中的节点，先将该节点使用NoOp进行替换，断开与输入的连边，输出连边复制老节点的连边
      return ReplaceNodeWithNoOp(node_in_init);
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus Pruner::PruneFromNodes(const vector<ge::FastNode *> &start_nodes, bool &changed) {
  // 标记node是否必须被删除，如果second为true，那么当这个node无法被被删除时，认为是失败场景，返回剪枝失败
  DeduplicateQueue<std::pair<ge::FastNode *, bool>> nodes;
  for (const auto start_node : start_nodes) {
    nodes.push({start_node, start_nodes_must_be_deleted_});
  }

  while (!nodes.empty()) {
    auto node = nodes.pop();
    bool is_resource_alloc = IsResourceAllocNode(node.first);
    if (!CanNodeBeDeleted(is_resource_alloc, node.first)) {
      if (node.second) {
        GELOGE(ge::GRAPH_FAILED, "Failed to delete node %s", node.first->GetName().c_str());
        return ge::GRAPH_FAILED;
      }
      continue;
    }
    changed = true;
    // 处理子图场景，子图内部，一个InnerData节点被多个引用的场景不再往Init图里面找
    if ((IsInnerDataType(node.first->GetTypePtr())) && (IsSingleOutAndRef(node.first))) {
      GE_ASSERT_GRAPH_SUCCESS(DealWithInnerData(node.first));
    } else {
      for (const auto in_node : node.first->GetInDataNodes()) {
        nodes.push({in_node, false});
      }
    }
    GE_ASSERT_SUCCESS(DeleteNode(is_resource_alloc, node.first));
  }
  return ge::GRAPH_SUCCESS;
}

Pruner &Pruner::StartNodesMustBeDeleted() {
  start_nodes_must_be_deleted_ = true;
  return *this;
}
}  // namespace bg
}  // namespace gert
