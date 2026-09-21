/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stack>
#include "graph/build/memory/mem_reuse_strategy.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "mem_assigner.h"

namespace ge {
int32_t MemReuseUtils::GetThreadScopeId(const ge::OpDesc *const desc) {
  int32_t thread_scope_id = ge::kInvalidThreadScopeId;
  if ((desc != nullptr) && ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_THREAD_SCOPE_ID, thread_scope_id)) {
    return thread_scope_id;
  }
  return ge::kInvalidThreadScopeId;
}

int64_t MemReuseUtils::GetStreamId(const ge::OpDesc *const desc) {
  if (desc != nullptr) {
    int64_t sub_stream_id = ge::kInvalidStreamId;
    if (ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_SUB_STREAM_ID, sub_stream_id)) {
      return sub_stream_id;
    } else {
      if ((GetThreadScopeId(desc) != ge::kInvalidThreadScopeId)) {
        return ge::kInvalidStreamId;
      }
    }
    return desc->GetStreamId();
  }
  return ge::kInvalidStreamId;
}

void MemReuseUtils::SetStreamId(ge::OpDesc *const desc, int64_t stream_id) {
  if (desc != nullptr) {
    (void) ge::AttrUtils::SetInt(desc, ge::ATTR_NAME_SUB_STREAM_ID, stream_id);
  }
}

bool MemReuseUtils::IsMergeNode(const NodePtr &node) {
  return IsMergeNode(node.get());
}

bool MemReuseUtils::IsMergeNode(const Node *node) {
  return (node->GetType() == STREAMMERGE) || (node->GetType() == MERGE);
}

bool MemReuseUtils::IsNoPaddingContinuousInput(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  bool is_nopading_input_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_nopading_input_continuous);
  return is_nopading_input_continuous && (node->GetAllInDataAnchorsSize() > 1U);
}

Status MemReuseUtils::GetOutputNoAlignSize(const ge::OpDesc &desc, uint32_t index, size_t &size) {
  const auto tensor_desc = desc.GetOutputDesc(index);
  GE_ASSERT_SUCCESS(GetNoAlignSize(tensor_desc, size), "node: %s, output index: %u", desc.GetNamePtr(), index);
  return SUCCESS;
}

Status MemReuseUtils::GetNoAlignSize(const GeTensorDesc &tensor, size_t &size) {
  int64_t tensor_size = 0;
  const GeShape &shape = tensor.GetShape();
  Format format = tensor.GetFormat();
  DataType data_type = tensor.GetDataType();
  bool is_no_tiling = false;
  (void)AttrUtils::GetBool(tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
  graphStatus graph_status;
  if (is_no_tiling) {
    graph_status = TensorUtils::CalcTensorMemSizeForNoTiling(tensor, format, data_type, tensor_size);
  } else {
    graph_status = TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  }
  GE_ASSERT_SUCCESS(graph_status, "[Calculate][TensorSize]shape:%s, format:%s, data_type:%s",
                    shape.ToString().c_str(),
                    TypeUtils::FormatToSerialString(format).c_str(),
                    TypeUtils::DataTypeToSerialString(data_type).c_str());
  size = static_cast<size_t>(tensor_size);
  return SUCCESS;
}

bool MemReuseUtils::IsAllOutRefAllInput(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    int32_t reuse_input_index = 0;
    if ((!GraphUtils::IsRefFromInput(out_data_anchor, reuse_input_index)) ||
        (reuse_input_index != out_data_anchor->GetIdx())) {
      return false;
    }
  }
  return true;
}

Status MemReuseUtils::GetTensorSize(const GeTensorDesc &tensor_desc, int64_t &size,
                                    const bool need_split_size) {
  int64_t ffts_size = -1;
  if (need_split_size && AttrUtils::GetInt(&tensor_desc, ATTR_NAME_FFTS_SUB_TASK_TENSOR_SIZE, ffts_size)
      && (ffts_size != -1)) {
    size = ffts_size;
    GELOGI("Sub task tensor size is:%ld", size);
    return SUCCESS;
  }

  if (ge::TensorUtils::GetSize(tensor_desc, size) != SUCCESS) {
    GELOGI("Tensor has no size");
  }
  bool is_tensor_desc_mem = false;
  (void)AttrUtils::GetBool(&tensor_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_tensor_desc_mem);
  if (size == 0 && is_tensor_desc_mem) {
    return ge::TensorUtils::CalcTensorMemSizeForNoTiling(tensor_desc, tensor_desc.GetFormat(),
                                                         tensor_desc.GetDataType(), size);
  }
  return SUCCESS;
}

bool MemReuseUtils::IsNeedSplitSize(const ge::NodePtr &node, const uint32_t index) {
  return IsNeedSplitSize(node.get(), index);
}

bool MemReuseUtils::IsNeedSplitSize(const ge::Node *const node, const uint32_t index) {
  if (node == nullptr) {
    return true;
  }

  if ((node->GetOpDescBarePtr() != nullptr) && ge::OpTypeUtils::IsDataNode(node->GetOpDescBarePtr()->GetType())) {
    GELOGD("Node:%s type is data.", node->GetOpDescBarePtr()->GetNamePtr());
    return false;
  }

  auto out_data_anchor = node->GetOutDataAnchor(index);
  if (out_data_anchor == nullptr) {
    return true;
  }
  for (const auto in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    auto owner_node = in_anchor->GetOwnerNodeBarePtr();
    if ((owner_node != nullptr) && (owner_node->GetOpDescBarePtr() != nullptr)) {
      if (owner_node->GetOpDescBarePtr()->GetType() == ge::NETOUTPUT) {
        GELOGD("Node:%s output:%u type is netoutput.", node->GetOpDescBarePtr()->GetNamePtr(), index);
        return false;
      }
    }
  }
  return true;
}

void MemReuseUtils::AlignMemOffset(size_t &mem_align_size) {
  if (mem_align_size <= 0UL) {
    return;
  }
  mem_align_size = (mem_align_size + MEM_ALIGN_SIZE - 1UL) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
}

Status MemReuseUtils::GetSrcNodeThroughRefNode(const Node *const node, const int32_t input_index, Node *&src_node,
                                               int32_t &out_index) {
  GE_ASSERT_NOTNULL(node);
  std::stack<InDataAnchor *> in_data_anchor_stack;
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(input_index), "node: %s, input_index: %d", node->GetNamePtr(), input_index);
  in_data_anchor_stack.push(node->GetInDataAnchor(input_index).get());
  while (!in_data_anchor_stack.empty()) {
    const auto in_data_anchor = in_data_anchor_stack.top();
    in_data_anchor_stack.pop();
    GE_ASSERT_NOTNULL(in_data_anchor);
    GE_ASSERT_NOTNULL(in_data_anchor->GetOwnerNodeBarePtr());
    const auto peer_out = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out, "peer out data anchor is null, dst_node: %s, index: %d",
                      in_data_anchor->GetOwnerNodeBarePtr()->GetNamePtr(), in_data_anchor->GetIdx());
    src_node = peer_out->GetOwnerNodeBarePtr();
    out_index = peer_out->GetIdx();
    GE_ASSERT_NOTNULL(src_node);
    int32_t reuse_in_index;
    if (GraphUtils::IsRefFromInput(peer_out, reuse_in_index)) {
      auto new_peer_in_anchor = src_node->GetInDataAnchor(reuse_in_index).get();
      GE_ASSERT_NOTNULL(new_peer_in_anchor);
      if (new_peer_in_anchor->GetPeerOutAnchor() != nullptr) {
        in_data_anchor_stack.push(new_peer_in_anchor);
      }
    }
  }
  return SUCCESS;
}

/*
 *    a
 *    /\
 *   b  ref_node
 *        /\
 *       c  d
 *
 */
Status MemReuseUtils::GetDstNodeThroughRefNode(const Node *const node, const int32_t out_index,
                                               std::vector<Node *> &dst_nodes, std::vector<int32_t> &in_indexes) {
  GE_ASSERT_NOTNULL(node);
  std::stack<const OutDataAnchor *> out_anchor_stack;
  GE_ASSERT_NOTNULL(node->GetOutDataAnchor(out_index), "node: %s, out_index: %d", node->GetNamePtr(), out_index);
  out_anchor_stack.push(node->GetOutDataAnchor(out_index).get());
  while (!out_anchor_stack.empty()) {
    const auto out_anchor = out_anchor_stack.top();
    out_anchor_stack.pop();
    GE_ASSERT_NOTNULL(out_anchor);
    GE_ASSERT_NOTNULL(out_anchor->GetOwnerNodeBarePtr());
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      const auto dst_node = in_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(dst_node, "src node: %s, index: %d",
                        out_anchor->GetOwnerNodeBarePtr()->GetNamePtr(), out_anchor->GetIdx());
      bool is_ref_node = false;
      for (const auto &dst_node_out_anchor : dst_node->GetAllOutDataAnchors()) {
        int32_t reuse_in_index;
        if (GraphUtils::IsRefFromInput(dst_node_out_anchor, reuse_in_index) &&
            (reuse_in_index == in_anchor->GetIdx())) {
          out_anchor_stack.push(dst_node_out_anchor.get());
          is_ref_node = true;
          // 这里认为只有一个输出引用一个输入，不会出现多个输出引用一个输入
          break;
        }
      }
      if (!is_ref_node) {
        dst_nodes.emplace_back(dst_node);
        in_indexes.emplace_back(in_anchor->GetIdx());
      }
    }
  }
  return SUCCESS;
}


bool GetDiffStreamNodes(const Node *const node, const int64_t stream_id, std::vector<const Node *> &diff_stream_nodes,
                        bool &back_to_same_stream, int64_t &tail_node_id) {
  std::vector<const Node *> nodes_stack;
  back_to_same_stream = false;
  nodes_stack.emplace_back(node);
  while (!nodes_stack.empty()) {
    const auto t_node = nodes_stack.back();
    nodes_stack.pop_back();
    if ((t_node->GetOutDataNodesSize() == 1U) && ((t_node == node) || (t_node->GetInDataNodesSize() == 1U))) {
      const auto out_data_node = t_node->GetOutDataNodes().at(0);
      GE_ASSERT_NOTNULL(out_data_node);
      const auto out_node_desc = out_data_node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(out_node_desc);
      const bool no_need_process = (stream_id == kInvalidStreamId) || MemReuseUtils::IsMergeNode(out_data_node)
          || (MemReuseUtils::GetThreadScopeId(out_node_desc) != ge::kInvalidThreadScopeId);
      if (no_need_process) {
        return false;
      }

      const int64_t out_stream_id = MemReuseUtils::GetStreamId(out_node_desc);
      if (out_stream_id == kInvalidStreamId) {
        return false;
      }
      if (out_stream_id != stream_id) {
        nodes_stack.emplace_back(out_data_node.get());
        diff_stream_nodes.emplace_back(out_data_node.get());
      } else {
        back_to_same_stream = true;
        tail_node_id = out_node_desc->GetId();
        break;
      }
    }
  }
  return true;
}

///        a stream:0
///              |
///        b stream:1
///              |
///        c stream:2
///              |
///        d stream:0
/// 其他stream上节点大于1个时，比如 b，c，此场景c有机会复用a的内存，设置a优先跨stream复用
bool MemReuseStrategy::GetDiffStreamPrior(const Node *const node) {
  std::vector<const Node *> diff_stream_nodes;
  bool back_to_same_stream = false;
  int64_t tail_node_id = 0;
  const bool success = GetDiffStreamNodes(node, MemReuseUtils::GetStreamId(node->GetOpDescBarePtr()), diff_stream_nodes,
                                    back_to_same_stream, tail_node_id);
  if (success && back_to_same_stream && (diff_stream_nodes.size() > 1U)) {
    // 跨流时，大于1个节点才有机会复用
    GELOGI("Node:%s is diff stream prior, diff stream nodes size:%zu", node->GetName().c_str(),
           diff_stream_nodes.size());
    return true;
  }
  return false;
}

/// 跨stream，topo id是连续的，实际还是串行，b，c可以当作stream 0进行复用
/// a (stream:0 id:1) -> b (stream:1 id:2) -> c (stream:2 id:3) -> d (stream:0 id:4)
void MemReuseStrategy::OptimizeDiffStream(const Node *const node) {
  std::vector<const Node *> diff_stream_nodes;
  bool back_to_same_stream = false;
  int64_t tail_node_id = 0;
  const auto node_desc = node->GetOpDescBarePtr();
  if (node_desc == nullptr) {
    return;
  }
  const int64_t stream_id = MemReuseUtils::GetStreamId(node_desc);
  const bool success = GetDiffStreamNodes(node, stream_id, diff_stream_nodes, back_to_same_stream, tail_node_id);
  if (success && back_to_same_stream && (!diff_stream_nodes.empty())) {
    if (static_cast<size_t>(tail_node_id - node_desc->GetId() - 1) == diff_stream_nodes.size()) {
      for (const auto diff_stream_node : diff_stream_nodes) {
        MemReuseUtils::SetStreamId(diff_stream_node->GetOpDescBarePtr(), stream_id);
        GELOGI("Set node:%s stream id to:%ld", diff_stream_node->GetName().c_str(), stream_id);
      }
    }
  }
}

std::string MemReuseUtils::GetGraphNameId(const ge::ComputeGraph *const graph) {
  GE_ASSERT_NOTNULL(graph);
  std::string graph_name = graph->GetName();
  graph_name.append("_");
  graph_name.append(std::to_string(graph->GetGraphID()));
  return graph_name;
}
}  // namespace ge
