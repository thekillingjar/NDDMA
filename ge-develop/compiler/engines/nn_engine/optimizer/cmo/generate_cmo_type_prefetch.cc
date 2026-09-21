/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generate_cmo_type_prefetch.h"
#include "common/fe_log.h"
#include "common/fe_op_info_common.h"
#include "common/configuration.h"
#include "common/graph/fe_graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"

namespace fe {
static const size_t kMinTaskSize = 2;

GenerateCMOTypePrefetch::GenerateCMOTypePrefetch()
    : GenerateCMOTypeBase() {
}

ge::NodePtr GenerateCMOTypePrefetch::GetLastPreNode(const ge::NodePtr &node,
    std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) const {
  ge::NodePtr nearest_in_node = nullptr;
  uint32_t stream_id = node->GetOpDesc()->GetStreamId();
  int64_t stream_index = -1;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, stream_index);
  auto iter = stream_node_map.find(stream_id);
  if (iter == stream_node_map.end()) {
    if (stream_index != -1) {
      std::map<int64_t, ge::NodePtr> tmp_map;
      tmp_map.insert(std::make_pair(stream_index, node));
      stream_node_map.insert(std::make_pair(stream_id, tmp_map));
    }
    return nearest_in_node;
  }
  if (iter->second.size() >= kMinTaskSize) {
    auto tmp_iter = iter->second.rbegin();
    ++tmp_iter;
    nearest_in_node = (*tmp_iter).second;
  }
  if (stream_index != -1) {
    iter->second.insert(std::make_pair(stream_index, node));
  }
  return nearest_in_node;
}

void GenerateCMOTypePrefetch::LabeledPrefetch(const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                                              std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map) const {
  std::vector<CmoAttr> vec_attr;
  ge::NodePtr tmp_node;
  for (const auto &in_data_anchor : dst_node->GetAllInDataAnchors()) {
    if (FeGraphUtils::IsPeerOutWeight(dst_node.get(), in_data_anchor->GetIdx(), tmp_node)) {
      bool is_cached = prefetch_cache_map.find(tmp_node) != prefetch_cache_map.end() &&
                       !CheckNeedPretch(prefetch_cache_map[tmp_node], dst_node);
      if (is_cached) {
        FE_LOGD("Op[name=%s] has just been prefetched by Op[name=%s]; no need to prefetch.", tmp_node->GetName().c_str(),
                prefetch_cache_map[tmp_node]->GetName().c_str());
        continue;
      }
      int64_t tensor_size = 0;
      auto weight = dst_node->GetOpDesc()->MutableInputDesc(in_data_anchor->GetIdx());
      if (!CheckAndGetWeightSize(weight, tensor_size)) {
        FE_LOGD("Op[name=%s,type=%s] input:%d size:%ld not in prefetch range[%ld:%ld)", dst_node->GetName().c_str(),
                dst_node->GetType().c_str(), in_data_anchor->GetIdx(), tensor_size, kPrefetchMin, kPrefetchMax);
        continue;
      }
      CmoAttr attr{dst_node, CmoTypeObject::WEIGHT, in_data_anchor->GetIdx()};
      vec_attr.emplace_back(attr);
      prefetch_cache_map[tmp_node] = dst_node;
    }
  }

  if (vec_attr.empty()) {
    return;
  }
  auto src_op_desc = src_node->GetOpDesc();
  auto dst_op_desc = dst_node->GetOpDesc();
  AddToNodeCmoAttr(src_op_desc, kCmoPrefetch, vec_attr);
  FE_LOGD("Op[name=%s, type=%s] for Op[name=%s, type=%s] add label:Prefetch, size:%zu", src_op_desc->GetName().c_str(),
          src_op_desc->GetType().c_str(), dst_op_desc->GetName().c_str(), dst_op_desc->GetType().c_str(),
          vec_attr.size());
  return;
}

bool GenerateCMOTypePrefetch::CheckNeedPretch(const ge::NodePtr &src_node, const ge::NodePtr &dst_node) const {
  int32_t src_index = -1;
  int32_t dst_index = -1;
  (void)ge::AttrUtils::GetInt(src_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, src_index);
  (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, dst_index);
  int32_t threshold = Configuration::Instance(AI_CORE_NAME).GetMemReuseDistThreshold();
  FE_LOGD("Op[name=%s, type=%s, index=%d] and Op[name=%s, type=%s, index=%d] has same weight, threshold=%d",
          src_node->GetName().c_str(), src_node->GetType().c_str(), src_index,
          dst_node->GetName().c_str(), dst_node->GetType().c_str(), dst_index,
          threshold);
  return dst_index - src_index >= threshold;
}

bool GenerateCMOTypePrefetch::CheckSizeIsAvailable(const ge::NodePtr &src_node, const ge::NodePtr &dst_node) const {
  int64_t cur_node_total_weight_size = GetWeightSize(dst_node);
  if (cur_node_total_weight_size == INTERNAL_ERROR) {
    return false;
  }
  int64_t src_node_input_size = GetInputTensorSize(src_node->GetOpDesc());
  int64_t src_node_output_size = GetOutputTensorSize(src_node->GetOpDesc());
  int64_t src_node_workspace_size = GetWorkSpaceSize(src_node->GetOpDesc());
  int64_t cache_size = GetCacheSize();
  FE_LOGD("DstNode[%s], SrcNode[%s]: in_size=%ld, out_size=%ld, workspace_size=%ld, weight_size=%ld, cache_size=%ld",
          dst_node->GetOpDesc()->GetName().c_str(), src_node->GetOpDesc()->GetName().c_str(),
          src_node_input_size, src_node_output_size, src_node_workspace_size, cur_node_total_weight_size, cache_size);
  return (src_node_input_size + src_node_output_size + src_node_workspace_size +
          cur_node_total_weight_size <= cache_size);
}

void GenerateCMOTypePrefetch::GenerateType(const ge::NodePtr &node,
                                           const StreamCtrlMap &stream_ctrls,
                                           std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                                           std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) {
  (void)stream_ctrls;
  /**
   * only prefetch weight input
   * prefetch lable on parent node
   */
  FE_LOGD("begin to generate prefetch for node:[name=%s, type=%s]",
          node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
  ge::NodePtr peer_out_node = GetLastPreNode(node, stream_node_map);
  if (peer_out_node == nullptr) {
    FE_LOGD("Ending generation of prefetch for node: [name=%s, type=%s], reason: no parent node",
            node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
    return;
  }

  vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(*node);
  if (weights.empty()) {
    FE_LOGD("Ending generation of prefetch for node: [name=%s, type=%s], reason: no weights",
            node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
    return;
  }

  if (!CheckSizeIsAvailable(peer_out_node, node)) {
    FE_LOGD("Ending generation of prefetch for node: [name=%s, type=%s], reason: size information not available",
            node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
    return;
  }

  LabeledPrefetch(peer_out_node, node, prefetch_cache_map);
  FE_LOGD("Ending generation of prefetch for node: [name=%s, type=%s]",
          node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
  return;
}
} // namespace fe
