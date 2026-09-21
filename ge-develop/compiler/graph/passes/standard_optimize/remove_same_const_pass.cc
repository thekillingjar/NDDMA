/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_same_const_pass.h"

#include <sstream>
#include <string>
#include <set>

#include "common/base64.h"
#include "common/checker.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
enum class KeyLevel {
  kControlInNodes = 0,
  kWeightSize,
  kWeightDesc,
  kAllAttrsSize,
  kAllAttrsValue,
};
const AttrNameFilter attr_weights_filter = [](const std::string &attr_name) -> bool {
  return (attr_name != ATTR_NAME_WEIGHTS);
};
std::string GetCseKeyByAttrs(const NodePtr node, const KeyLevel key_level,
                             const std::map<std::string, AnyValue> &attrs) {
  std::string ss;
  switch (key_level) {
    case KeyLevel::kControlInNodes: {
      (void)ss.append("type-").append(node->GetType()).append("-");
      std::set<std::string> control_in_node_names;
      for (const auto &src_node : node->GetInControlNodes()) {
        control_in_node_names.insert(src_node->GetName());
      }
      for (const auto &name : control_in_node_names) {
        (void)ss.append(name).append("-");
      }
      break;
    }
    case KeyLevel::kAllAttrsSize:
      (void)ss.append("attr-size-").append(std::to_string(attrs.size()));
      break;
    case KeyLevel::kAllAttrsValue:
      (void)ss.append("attrs-").append(AttrUtils::GetAllAttrsStr(attrs));
      break;
    default:;
  }
  return ss;
}

std::string GetCseKeyByTensor(const KeyLevel key_level, const GeTensorPtr &weight) {
  std::string ss;
  switch (key_level) {
    case KeyLevel::kWeightSize: {
      const size_t output_size = weight->GetData().GetSize();
      (void)ss.append("weight-size-").append(std::to_string(output_size));
      break;
    }
    case KeyLevel::kWeightDesc: {
      std::map<std::string, AnyValue> tensor_desc;
      tensor_desc.emplace(std::make_pair("weight_tensor_desc", AnyValue::CreateFrom(weight->GetTensorDesc())));
      (void)ss.append("tensor_desc-").append(AttrUtils::GetAllAttrsStr(tensor_desc));
      break;
    }
    default:;
  }
  return ss;
}

Status FilterSameKeyNodesByAttr(const std::map<std::string, std::vector<NodePtr>> &level_to_nodes,
                                const KeyLevel &key_level,
                                std::map<NodePtr, std::map<std::string, AnyValue>> &node_attrs,
                                std::map<std::string, std::vector<NodePtr>> &new_level_to_nodes) {
  for (const auto &pair : level_to_nodes) {
    if (pair.second.size() <= 1U) {
      continue;
    }
    for (const auto &node : pair.second) {
      if (key_level == KeyLevel::kAllAttrsSize) {
        const auto &op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        node_attrs[node] = AttrUtils::GetAllAttrsWithFilter(op_desc, attr_weights_filter);
      } else {
        GE_ASSERT_TRUE(node_attrs.find(node) != node_attrs.end());
      }
      const auto &key = GetCseKeyByAttrs(node, key_level, node_attrs[node]);
      new_level_to_nodes[pair.first + key].emplace_back(node);
    }
  }
  return SUCCESS;
}

Status FilterSameKeyNodesByTensor(const std::map<std::string, std::vector<NodePtr>> &level_to_nodes,
                                  const KeyLevel &key_level, std::map<NodePtr, GeTensorPtr> &node_tensor,
                                  std::map<std::string, std::vector<NodePtr>> &new_level_to_nodes) {
  for (const auto &pair : level_to_nodes) {
    if (pair.second.size() <= 1U) {
      continue;
    }
    for (const auto &node : pair.second) {
      if (key_level == KeyLevel::kWeightSize) {
        const auto &op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        GeTensorPtr weight;
        (void)AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight);
        GE_CHECK_NOTNULL(weight);
        node_tensor[node] = weight;
      } else {
        GE_ASSERT_TRUE(node_tensor.find(node) != node_tensor.end());
      }
      const auto &key = GetCseKeyByTensor(key_level, node_tensor[node]);
      new_level_to_nodes[pair.first + key].emplace_back(node);
    }
  }
  return SUCCESS;
}

Status FilterSameKeyNodesByValue(const std::map<std::string, std::vector<NodePtr>> &level_to_nodes,
                                 std::map<NodePtr, GeTensorPtr> &node_tensor,
                                 std::map<std::string, std::map<void *, std::vector<NodePtr>>> &new_level_to_nodes) {
  for (const auto &pair : level_to_nodes) {
    if (pair.second.size() <= 1U) {
      continue;
    }
    for (const auto &node : pair.second) {
      const auto iter_weight = node_tensor.find(node);
      GE_ASSERT_TRUE(iter_weight != node_tensor.end());
      const auto &weight = iter_weight->second;
      GE_CHECK_NOTNULL(weight);
      const auto &values = static_cast<void *>(weight->MutableData().data());
      const auto iter = new_level_to_nodes.find(pair.first);
      if (iter == new_level_to_nodes.end()) {
        std::map<void *, std::vector<NodePtr>> value_nodes;
        value_nodes[values].emplace_back(node);
        new_level_to_nodes[pair.first] = std::move(value_nodes);
        GELOGD("Add const node: %s, key: %s.", node->GetName().c_str(), pair.first.c_str());
        continue;
      }
      auto &value_nodes = iter->second;
      bool has_same_value = false;
      const size_t weight_size = weight->GetData().size();
      for (auto &value_pair : value_nodes) {
        const auto &cur_value = value_pair.first;
        if (memcmp(values, cur_value, weight_size) == 0) {
          GELOGD("Const node: %s, has same key: %s.", node->GetName().c_str(), pair.first.c_str());
          value_pair.second.emplace_back(node);
          has_same_value = true;
          break;
        }
      }
      if (!has_same_value) {
        value_nodes[values].emplace_back(node);
      }
    }
  }
  return SUCCESS;
}

Status FilterSameKeyNodesByGeTensor(const std::map<std::string, std::vector<NodePtr>> &level_to_nodes,
                                    std::map<std::string, std::vector<NodePtr>> &new_level_to_nodes) {
  std::map<std::string, std::vector<NodePtr>> weight_size_to_nodes;
  std::map<NodePtr, GeTensorPtr> node_tensor;
  GE_ASSERT_SUCCESS(
      FilterSameKeyNodesByTensor(level_to_nodes, KeyLevel::kWeightSize, node_tensor, weight_size_to_nodes));

  std::map<std::string, std::vector<NodePtr>> tensor_desc_to_nodes;
  GE_ASSERT_SUCCESS(
      FilterSameKeyNodesByTensor(weight_size_to_nodes, KeyLevel::kWeightDesc, node_tensor, tensor_desc_to_nodes));

  std::map<std::string, std::map<void *, std::vector<NodePtr>>> weight_value_to_nodes;
  GE_ASSERT_SUCCESS(FilterSameKeyNodesByValue(tensor_desc_to_nodes, node_tensor, weight_value_to_nodes));

  for (const auto &pair : weight_value_to_nodes) {
    for (const auto &value_nodes : pair.second) {
      if (value_nodes.second.size() > 1U) {
        const std::string key = pair.first + "weight-ptr-" + std::to_string(PtrToValue(value_nodes.first));
        (void)new_level_to_nodes[key].insert(new_level_to_nodes[key].end(), value_nodes.second.begin(),
                                             value_nodes.second.end());
      }
    }
  }
  return SUCCESS;
}

Status ReplaceConstNode(const ComputeGraphPtr &graph, const NodePtr &node, const NodePtr &to_be_removed_node) {
  if (to_be_removed_node->GetAllOutDataAnchorsSize() != node->GetAllOutDataAnchorsSize()) {
    GELOGW("The const node %s and %s have the same CSE key, but different output anchor count, skip to fusion them",
           node->GetName().c_str(), to_be_removed_node->GetName().c_str());
    return SUCCESS;
  }

  std::vector<int32_t> output_map(to_be_removed_node->GetAllOutDataAnchorsSize());
  for (size_t i = 0U; i < to_be_removed_node->GetAllOutDataAnchorsSize(); ++i) {
    output_map[i] = i;
  }

  if (GraphUtils::ReplaceNodeAnchors(node, to_be_removed_node, {}, output_map) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Replace to_be_removed_node:%s(%s)'s anchor by node:%s(%s) failed",
                      to_be_removed_node->GetName().c_str(), to_be_removed_node->GetType().c_str(),
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Replace][Anchors] of to_be_removed_node:%s(%s) by node:%s(%s) failed",
           to_be_removed_node->GetName().c_str(), to_be_removed_node->GetType().c_str(), node->GetName().c_str(),
           node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  NodeUtils::UnlinkAll(*to_be_removed_node);

  if (GraphUtils::RemoveNodeWithoutRelink(graph, to_be_removed_node) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove to_be_removed_node:%s(%s) without relink in graph:%s failed",
                      to_be_removed_node->GetName().c_str(), to_be_removed_node->GetType().c_str(),
                      graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] %s(%s) without relink in graph:%s failed",
           to_be_removed_node->GetName().c_str(), to_be_removed_node->GetType().c_str(), graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGI("Remove const to_be_removed_node %s by RemoveSameConstPass, replace it with node %s",
         to_be_removed_node->GetName().c_str(), node->GetName().c_str());
  return SUCCESS;
}

Status RemoveDuplicateNodes(const ComputeGraphPtr &graph,
                            const std::map<std::string, std::vector<NodePtr>> &level_last_to_nodes) {
  for (const auto &pair : level_last_to_nodes) {
    if (pair.second.size() <= 1U) {
      continue;
    }
    auto iter = pair.second.begin();
    const auto &saved_node = *iter;
    GELOGD("The const node %s cse key %s", saved_node->GetName().c_str(),
           ge::base64::EncodeToBase64(pair.first).c_str());
    ++iter;
    for (; iter != pair.second.end(); ++iter) {
      GE_CHK_STATUS_RET(ReplaceConstNode(graph, saved_node, *iter), "Replace const %s by node %s failed",
                        (*iter)->GetName().c_str(), saved_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

bool IsConstType(const NodePtr &node) {
  return ((node->GetType() == CONSTANT) || (node->GetType() == CONSTANTOP));
}
}  // namespace

Status RemoveSameConstPass::Run(ComputeGraphPtr graph) {
  GELOGD("Begin to run RemoveSameConstPass on the graph");
  GE_CHECK_NOTNULL(graph);
  std::map<std::string, std::vector<NodePtr>> level0_to_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    if (!IsConstType(node)) {
      continue;
    }
    bool is_unknown = false;
    auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Get node unknown status failed, node name:%s, type:%s.", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    if (is_unknown) {
      GELOGI("Current node %s, type %s is unknown shape which should be skip.", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    const auto &key = GetCseKeyByAttrs(node, KeyLevel::kControlInNodes, {});
    level0_to_nodes[key].emplace_back(node);
  }

  std::map<std::string, std::vector<NodePtr>> level1_to_nodes;
  GE_ASSERT_SUCCESS(FilterSameKeyNodesByGeTensor(level0_to_nodes, level1_to_nodes));

  std::map<NodePtr, std::map<std::string, AnyValue>> node_attrs;
  std::map<std::string, std::vector<NodePtr>> level2_to_nodes;
  GE_ASSERT_SUCCESS(FilterSameKeyNodesByAttr(level1_to_nodes, KeyLevel::kAllAttrsSize, node_attrs, level2_to_nodes));

  std::map<std::string, std::vector<NodePtr>> level3_to_nodes;
  GE_ASSERT_SUCCESS(FilterSameKeyNodesByAttr(level2_to_nodes, KeyLevel::kAllAttrsValue, node_attrs, level3_to_nodes));

  GE_ASSERT_SUCCESS(RemoveDuplicateNodes(graph, level3_to_nodes));
  GELOGD("Success to run RemoveSameConstPass on the graph %s.", graph->GetName().c_str());

  return SUCCESS;
}

REG_PASS_OPTION("RemoveSameConstPass").LEVELS(OoLevel::kO3);
}  // namespace ge
