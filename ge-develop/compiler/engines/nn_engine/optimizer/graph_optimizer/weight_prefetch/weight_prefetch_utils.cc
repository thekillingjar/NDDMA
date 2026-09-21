/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/weight_prefetch/weight_prefetch_utils.h"
#include <unordered_set>
#include "common/fe_log.h"
#include "common/math_util.h"
#include "common/string_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace fe {
namespace {
const uint32_t kWeightAlignSize = 4;
const std::string kAttrIsGraphPrefetch = "_is_graph_prefetch";
const std::unordered_set<std::string> kWeightOpTypeSet = {DATA, CONSTANT, CONSTANTOP, OP_TYPE_PLACE_HOLDER};
}

Status WeightPrefetchUtils::HandleWeightPrefetch(const ge::ComputeGraph &graph) {
  if (IsGraphWeightPrefetch(graph)) {
    return HandleGraphPrefetch(graph);
  }

  return HandleNodesPrefetch(graph);
}

bool WeightPrefetchUtils::IsGraphWeightPrefetch(const ge::ComputeGraph &graph) {
  bool is_graph_prefetch = false;
  return ge::AttrUtils::GetBool(graph, kAttrIsGraphPrefetch, is_graph_prefetch) && is_graph_prefetch;
}

bool WeightPrefetchUtils::HasWeightPrefetchAttr(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return false;
  }
  std::vector<std::string> weight_prefetch_type;
  bool has_attr = ge::AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, weight_prefetch_type);
  if (has_attr) {
    std::vector<int64_t> weight_prefetch_dst_offset;
    (void) ge::AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, weight_prefetch_dst_offset);
    if (weight_prefetch_type.size() != weight_prefetch_dst_offset.size()) {
      FE_LOGW("Size of weight prefetch type [%zu] and weight prefetch dst offset [%zu] for node [%s, %s] are not the same.",
              weight_prefetch_type.size(), weight_prefetch_dst_offset.size(),
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    }
  }
  return has_attr;
}

Status WeightPrefetchUtils::HandleGraphPrefetch(const ge::ComputeGraph &graph) {
  FE_LOGD("Begin to handle whole graph weight prefetch for graph[%s].", graph.GetName().c_str());
  ge::OpDescPtr prefetch_op_desc = nullptr;
  std::vector<ge::ConstGeTensorPtr> weights_vec;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (HasWeightPrefetchAttr(node->GetOpDesc())) {
      prefetch_op_desc = node->GetOpDesc();
    }
    if (kWeightOpTypeSet.count(node->GetType()) != 0) {
      std::vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(node);
      for (const ge::ConstGeTensorPtr &weight : weights) {
        if (std::find(weights_vec.begin(), weights_vec.end(), weight) == weights_vec.end()) {
          weights_vec.push_back(weight);
        }
      }
    }
  }
  if (prefetch_op_desc == nullptr) {
    FE_LOGE("There is no node with the weight prefetch attribute in graph [%s].", graph.GetName().c_str());
    return FAILED;
  }

  if (SetWeightPrefetchAttr(prefetch_op_desc, weights_vec) != SUCCESS) {
    FE_LOGE("Failed to set weight prefetch attr for op[%s, %s].",
            prefetch_op_desc->GetName().c_str(), prefetch_op_desc->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status WeightPrefetchUtils::HandleNodesPrefetch(const ge::ComputeGraph &graph) {
  FE_LOGD("Begin to handle node weight prefetch for graph[%s].", graph.GetName().c_str());
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (HasWeightPrefetchAttr(node->GetOpDesc())) {
      if (SetNodePrefetchAttr(node, graph) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status WeightPrefetchUtils::SetNodePrefetchAttr(const ge::NodePtr &prefetch_node, const ge::ComputeGraph &graph) {
  FE_LOGD("Begin to set weight prefetch attr for node[%s, %s].",
          prefetch_node->GetName().c_str(), prefetch_node->GetType().c_str());
  std::vector<std::string> weight_prefetch_node_name;
  (void)ge::AttrUtils::GetListStr(prefetch_node->GetOpDesc(), kAttrWeightPrefetchNodeName, weight_prefetch_node_name);
  if (weight_prefetch_node_name.empty()) {
    FE_LOGW("Weight prefetch node name list of node[%s, %s] is empty.",
            prefetch_node->GetName().c_str(), prefetch_node->GetType().c_str());
    return SUCCESS;
  }

  std::vector<ge::ConstGeTensorPtr> weights_vec;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (std::find(weight_prefetch_node_name.begin(), weight_prefetch_node_name.end(), node->GetName())
        != weight_prefetch_node_name.end()) {
      // GetWeights function will fetch the weight when op is placeholder, data, const and constant
      // but GetWeights function will only fetch the weight from input op who is placeholder or data
      // so it is necessary to get the input data nodes first
      for (const ge::NodePtr &in_node : node->GetInDataNodes()) {
        if (kWeightOpTypeSet.count(in_node->GetType()) == 0) {
          continue;
        }
        std::vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(in_node);
        for (const ge::ConstGeTensorPtr &weight : weights) {
          if (std::find(weights_vec.begin(), weights_vec.end(), weight) == weights_vec.end()) {
            weights_vec.push_back(weight);
          }
        }
      }
    }
  }

  if (SetWeightPrefetchAttr(prefetch_node->GetOpDesc(), weights_vec) != SUCCESS) {
    FE_LOGE("Failed to set weight prefetch attr for op[%s, %s].",
            prefetch_node->GetName().c_str(), prefetch_node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status WeightPrefetchUtils::SetWeightPrefetchAttr(const ge::OpDescPtr &op_desc,
                                                  const std::vector<ge::ConstGeTensorPtr> &weights) {
  std::vector<int64_t> weight_prefetch_src_offset;
  std::vector<int64_t> weight_prefetch_data_size;
  for (const ge::ConstGeTensorPtr &weight_tensor : weights) {
    int64_t weight_offset = 0;
    if (ge::TensorUtils::GetDataOffset(weight_tensor->GetTensorDesc(), weight_offset) != ge::GRAPH_SUCCESS) {
      FE_LOGW("Failed to get weight offset.");
      return FAILED;
    }
    weight_prefetch_src_offset.push_back(weight_offset);
    uint32_t weight_data_size = ge::TensorUtils::GetWeightSize(weight_tensor);
    FE_ADD_OVERFLOW(weight_data_size, kWeightAlignSize - 1, weight_data_size);
    weight_data_size = weight_data_size / kWeightAlignSize;
    FE_MUL_OVERFLOW(weight_data_size, kWeightAlignSize, weight_data_size);
    weight_prefetch_data_size.push_back(static_cast<int64_t>(weight_data_size));
  }
  (void)ge::AttrUtils::SetListInt(op_desc, kAttrWeightPrefetchSrcOffset, weight_prefetch_src_offset);
  FE_LOGD("Set weight prefetch src offset[%s] for op[%s, %s].",
          StringUtils::IntegerVecToString(weight_prefetch_src_offset).c_str(),
          op_desc->GetName().c_str(), op_desc->GetType().c_str());
  (void)ge::AttrUtils::SetListInt(op_desc, kAttrWeightPrefetchDataSize, weight_prefetch_data_size);
  FE_LOGD("Set weight prefetch data size[%s] for op[%s, %s].",
          StringUtils::IntegerVecToString(weight_prefetch_data_size).c_str(),
          op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return SUCCESS;
}
}
