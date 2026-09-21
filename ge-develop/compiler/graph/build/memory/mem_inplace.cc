/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/mem_inplace.h"
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "utils/extern_math_util.h"

namespace ge {
namespace {
bool IsReadOnlyOpTypes(const NodePtr &node) {
  return OpTypeUtils::IsDataNode(node->GetType()) || OpTypeUtils::IsVariableNode(node->GetType())
         || MemLayoutConflictUtil::IsConst(node);
}

Status SetReuseInput(const std::map<InDataAnchorPtr, OutDataAnchorPtr> &in_anchor_to_out_anchors) {
  for (const auto &in_anchor_to_out_anchor : in_anchor_to_out_anchors) {
    const auto &in_anchor = in_anchor_to_out_anchor.first;
    const auto &out_anchor = in_anchor_to_out_anchor.second;
    const auto &output_tensor = out_anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(out_anchor->GetIdx());
    GE_CHECK_NOTNULL(output_tensor);
    GELOGD("Set reuse input for node %s, input index[%d], output index[%d].",
           out_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetIdx(), out_anchor->GetIdx());
    TensorUtils::SetReuseInput(*output_tensor, true);
    TensorUtils::SetReuseInputIndex(*output_tensor, in_anchor->GetIdx());
  }

  return SUCCESS;
}

void RecoverReuseInput(const std::map<InDataAnchorPtr, OutDataAnchorPtr> &in_anchor_to_out_anchors) {
  for (const auto &in_anchor_to_out_anchor : in_anchor_to_out_anchors) {
    const auto &out_anchor = in_anchor_to_out_anchor.second;
    const auto &output_tensor = out_anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(out_anchor->GetIdx());
    TensorUtils::SetReuseInput(*output_tensor, false);
    GELOGD("Recover reuse input for node %s.", out_anchor->GetOwnerNode()->GetName().c_str());
  }
}

void ConstructSingleNodeSymbolTable(const string &input_symbol,
                                    const string &output_symbol,
                                    const MemAssistInfo &mem_assist_info,
                                    AnchorToSymbol &anchor_to_symbol,
                                    SymbolToAnchors &symbol_to_anchors) {
  // 提取出特定的input_symbol和output_symbol对应的所有anchor的映射关系表
  for (const auto &it : mem_assist_info.anchor_to_symbol) {
    if (it.second == input_symbol || it.second == output_symbol) {
      anchor_to_symbol[it.first] = it.second;
    }
  }
  for (const auto &it : mem_assist_info.symbol_to_anchors) {
    if (it.first == input_symbol || it.first == output_symbol) {
      symbol_to_anchors[it.first].assign(it.second.begin(), it.second.end());
    }
  }
  // 删除output_symbol对应的symbol，合并到input_symbol
  for (auto &it : symbol_to_anchors[output_symbol]) {
    symbol_to_anchors[input_symbol].push_back(it);
    anchor_to_symbol[it.ToString()] = input_symbol;
  }
  symbol_to_anchors.erase(output_symbol);
}

Status GetReadOnlySymbol(const MemAssistInfo &mem_assist_info, std::set<std::string> &read_only_symbols) {
  for (const auto &node : mem_assist_info.compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!IsReadOnlyOpTypes(node)) {
      continue;
    }
    for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
      const NodeIndexIO output_info(node, out_anchor->GetIdx(), kOut);
      const auto output_symbol = mem_assist_info.anchor_to_symbol.find(output_info.ToString())->second;
      read_only_symbols.insert(output_symbol);
    }
  }
  return SUCCESS;
}

Status RemoveSizeNotEqual(const NodePtr &node,
                          std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::map<size_t, std::vector<size_t>> size_equal_indexes;
  for (auto &pair : out_index_to_refable_in_indexes) {
    size_t out_index = pair.first;
    const std::vector<size_t> &in_indexes = pair.second;

    auto output_desc = node->GetOpDesc()->GetOutputDescPtr(out_index);
    GE_CHECK_NOTNULL(output_desc);
    int64_t output_size;
    GE_ASSERT_SUCCESS(TensorUtils::GetTensorSizeInBytes(*output_desc, output_size));
    for (size_t input_index : in_indexes) {
      GE_ASSERT_TRUE(ge::IntegerChecker<uint32_t>::Compat(input_index));
      auto input_desc = node->GetOpDesc()->GetInputDescPtr(input_index);
      GE_CHECK_NOTNULL(input_desc);

      int64_t input_size;
      GE_ASSERT_SUCCESS(TensorUtils::GetTensorSizeInBytes(*input_desc, input_size));
      if (output_size == input_size) {
        size_equal_indexes[out_index].push_back(input_index);
      } else {
        GELOGD("Node %s input size is not equal to output size, cannot inplace, input index[%zu], output index[%zu].",
               node->GetName().c_str(), input_index, out_index);
      }
    }
  }
  out_index_to_refable_in_indexes.swap(size_equal_indexes);

  return SUCCESS;
}

Status RemoveInRwConflicts(const NodePtr &node,
                           const MemAssistInfo &mem_assist_info,
                           const std::set<std::string> &read_only_symbols,
                           std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes) {
  std::map<size_t, std::vector<size_t>> no_rw_conflict_indexes;
  for (auto &pair : out_index_to_refable_in_indexes) {
    size_t out_index = pair.first;
    const std::vector<size_t> &in_indexes = pair.second;
    for (size_t input_index : in_indexes) {
      GE_ASSERT_TRUE(ge::IntegerChecker<uint32_t>::Compat(input_index));
      const NodeIndexIO cur_node_input_info(node, static_cast<uint32_t>(input_index), kIn);
      if (mem_assist_info.anchor_to_symbol.find(cur_node_input_info.ToString()) ==
          mem_assist_info.anchor_to_symbol.end()) {
        GELOGD("Node %s input anchor has no symbol.", node->GetName().c_str());
        continue;
      }
      GELOGD("Check rw conflict, node %s input index[%zu], input list size[%zu].",
             node->GetName().c_str(), input_index, in_indexes.size());
      const auto input_symbol = mem_assist_info.anchor_to_symbol.find(cur_node_input_info.ToString())->second;
      if (read_only_symbols.find(input_symbol) != read_only_symbols.end()) {
        GELOGD("Node %s input symbol is read only, can not inplace, input index[%zu], input list size[%zu].",
               node->GetName().c_str(), input_index, in_indexes.size());
      } else {
        no_rw_conflict_indexes[out_index].push_back(input_index);
      }
    }
  }
  out_index_to_refable_in_indexes.swap(no_rw_conflict_indexes);

  return SUCCESS;
}

Status RemoveOutRwConflicts(const NodePtr &node,
                            std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  bool is_continuous_output = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, is_continuous_output);
  bool is_no_padding_continuous_output = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(),
                               ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_no_padding_continuous_output);
  if (is_continuous_output || is_no_padding_continuous_output) {
    GELOGD("Node %s is continuous output, not support inplace.", node->GetName().c_str());
    out_index_to_refable_in_indexes.clear();
    return SUCCESS;
  }
  // 判断输出是否有var shared memory, 有则不可写
  for (auto out_iter = out_index_to_refable_in_indexes.begin(); out_iter!= out_index_to_refable_in_indexes.end();) {
    auto output_desc = node->GetOpDesc()->GetOutputDescPtr(out_iter->first);
    GE_CHECK_NOTNULL(output_desc);
    std::string var_shared_memory;
    (void)ge::AttrUtils::GetStr(output_desc, REF_VAR_SRC_VAR_NAME, var_shared_memory);
    if (!var_shared_memory.empty()) {
      GELOGD("Node %s output index[%zu] has var shared memory, can not inplace.",
             node->GetName().c_str(), out_iter->first);
      out_iter = out_index_to_refable_in_indexes.erase(out_iter);
    } else {
      ++out_iter;
    }
  }

  return SUCCESS;
}

Status RemoveSymbolConflicts(const MemAssistInfo &mem_assist_info, const NodePtr &node,
                             std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes) {
  std::map<size_t, std::vector<size_t>> no_symblo_conflict_indexes;
  for (auto &pair : out_index_to_refable_in_indexes) {
    size_t output_index = pair.first;
    for (size_t input_index : pair.second) {
      GE_ASSERT_TRUE(ge::IntegerChecker<uint32_t>::Compat(input_index));
      const NodeIndexIO cur_node_input_info(node, static_cast<uint32_t>(input_index), kIn);
      const NodeIndexIO cur_node_output_info(node, static_cast<uint32_t>(output_index), kOut);
      const auto &input_symbol = mem_assist_info.anchor_to_symbol.find(cur_node_input_info.ToString())->second;
      const auto &output_symbol = mem_assist_info.anchor_to_symbol.find(cur_node_output_info.ToString())->second;
      if (input_symbol == output_symbol) { // 输入符号和输出符号相同，不需要合并和设置复用关系
        GELOGD("Node %s input symbol[%s] is equal to output symbol[%s], skip inplace.",
               node->GetName().c_str(), input_symbol.c_str(), output_symbol.c_str());
        continue;
      }

      // 创建单节点符号表，判断是否引入冲突
      AnchorToSymbol anchor_to_symbol;
      SymbolToAnchors symbol_to_anchors;
      ConstructSingleNodeSymbolTable(input_symbol, output_symbol,
                                     mem_assist_info, anchor_to_symbol, symbol_to_anchors);
      bool is_conflict = false;
      GE_ASSERT_SUCCESS(MemLayoutConflictUtil::IsGraphExistMemConflictSymbol(mem_assist_info.compute_graph,
                        anchor_to_symbol, symbol_to_anchors, is_conflict));
      if (is_conflict) {
        GELOGI("Symbol conflict, node %s can not inplace, input index[%zu], output index[%zu].",
          node->GetName().c_str(), input_index, output_index);
      } else {
        no_symblo_conflict_indexes[output_index].push_back(input_index);
      }
    }
  }
  out_index_to_refable_in_indexes.swap(no_symblo_conflict_indexes);

  return SUCCESS;
}

Status GetReuseAnchors(const NodePtr &node,
                       const std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes,
                       std::map<InDataAnchorPtr, OutDataAnchorPtr> &in_anchor_to_out_anchors) {
  // 需要一个数据结构保存已经复用的节点的输入输出index
  std::set<size_t> already_reuse_input_index;
  for (auto inplace_index : out_index_to_refable_in_indexes) {
    size_t output_index = inplace_index.first;
    for (auto input_index : inplace_index.second) {
      if (!already_reuse_input_index.insert(input_index).second) {
        GELOGD("Node %s input index[%zu] has already inplace.", node->GetName().c_str(), input_index);
        continue;
      }
      const auto in_anchor = node->GetInDataAnchor(input_index);
      const auto out_anchor = node->GetOutDataAnchor(output_index);
      GE_CHECK_NOTNULL(in_anchor);
      GE_CHECK_NOTNULL(out_anchor);
      in_anchor_to_out_anchors[in_anchor] = out_anchor;
      break;
    }
  }

  return SUCCESS;
}

Status MergeSymbolTable(const std::map<InDataAnchorPtr, OutDataAnchorPtr> &in_anchor_to_out_anchors,
                      MemAssistInfo &mem_assist_info) {
  GELOGD("After checking the conflicts of single nodes, there are indexes that can be inplace."
         "Prepare to check the conflicts of control subgraphs");
  GE_ASSERT_SUCCESS(SetReuseInput(in_anchor_to_out_anchors));
  // 判断控制子图是否冲突，根据图结构判断
  // 后续看看是否可以找到并删除冲突的anchors，并恢复复用关系
  // 当前如果子图出现新的冲突则不进行inplace操作，主要是为了编译性能考虑，当前没有为单个算子也传入子图进行判断，而是整体判断
  if (MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(mem_assist_info.compute_graph)) {
    RecoverReuseInput(in_anchor_to_out_anchors);
    GELOGI("Ctrl node subgraph conflict, can not inplace.");
  } else {
    // 合并符号表
    for (const auto &input_anchor_to_output_anchor : in_anchor_to_out_anchors) {
      const auto &in_anchor = input_anchor_to_output_anchor.first;
      const auto &out_anchor = input_anchor_to_output_anchor.second;
      const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      const NodeIndexIO input_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut); // inplace节点的输入节点
      const NodeIndexIO output_info(out_anchor->GetOwnerNode(), out_anchor->GetIdx(), kOut);
      const auto input_symbol = mem_assist_info.anchor_to_symbol[input_info.ToString()];
      const auto output_symbol = mem_assist_info.anchor_to_symbol[output_info.ToString()];
      GELOGD("Merge symbol table for node: %s, input anchor: %s, output anchor: %s"
             ", input symbol: %s, output symbol: %s.",
             out_anchor->GetOwnerNode()->GetName().c_str(), input_info.ToString().c_str(),
             output_info.ToString().c_str(), input_symbol.c_str(), output_symbol.c_str());
      auto &symbol_to_anchors = mem_assist_info.symbol_to_anchors[input_symbol];
      for (auto &it : mem_assist_info.symbol_to_anchors[output_symbol]) {
        symbol_to_anchors.emplace_back(it);
        mem_assist_info.anchor_to_symbol[it.ToString()] = input_symbol;
      }
      mem_assist_info.symbol_to_anchors.erase(output_symbol);
      GELOGI("Node %s can inplace, output[%d] can reuse input[%d]", out_anchor->GetOwnerNode()->GetName().c_str(),
              out_anchor->GetIdx(), in_anchor->GetIdx());
    }
  }

  return SUCCESS;
}
} // namespace

// 1. 遍历所有节点，找到满足基本inplace条件的节点,保存inplace的节点的输入输出index
// 2. 判断单个节点是否有符号冲突，保存没有符号冲突的所有anchors
// 3. 判断控制子图是否冲突，返回冲突的anchors
// 4. 删除冲突的anchors，并恢复复用关系
// 5. 合并符号表
Status ProcessInplace(MemAssistInfo &mem_assist_info) {
  ComputeGraphPtr compute_graph = mem_assist_info.compute_graph;
  GE_CHECK_NOTNULL(compute_graph);
  // 获取readonly的symbol表
  std::set<std::string> read_only_symbols;
  GE_ASSERT_SUCCESS(GetReadOnlySymbol(mem_assist_info, read_only_symbols));
  
  std::map<InDataAnchorPtr, OutDataAnchorPtr> in_anchor_to_out_anchors;
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    std::map<size_t, std::vector<size_t>> out_index_to_refable_in_indexes;
    GE_ASSERT_SUCCESS(GraphUtils::GetSupportInplaceOutput(node, out_index_to_refable_in_indexes));
    if (!out_index_to_refable_in_indexes.empty()) {
      GELOGI("Node %s has basic inplace capabilities, and it's necessary to check if there are any conflicts symbols.",
             node->GetName().c_str());
      GE_ASSERT_SUCCESS(RemoveSizeNotEqual(node, out_index_to_refable_in_indexes));
      GE_ASSERT_SUCCESS(RemoveInRwConflicts(node, mem_assist_info, read_only_symbols,
                                            out_index_to_refable_in_indexes));
      GE_ASSERT_SUCCESS(RemoveOutRwConflicts(node, out_index_to_refable_in_indexes));
      GE_ASSERT_SUCCESS(RemoveSymbolConflicts(mem_assist_info, node, out_index_to_refable_in_indexes));
      GE_ASSERT_SUCCESS(GetReuseAnchors(node, out_index_to_refable_in_indexes, in_anchor_to_out_anchors));
    }
  }

  // 合并符号表
  if (!in_anchor_to_out_anchors.empty()) {
    GE_ASSERT_SUCCESS(MergeSymbolTable(in_anchor_to_out_anchors, mem_assist_info));
  } else {
    GELOGD("After checking the conflicts of single nodes for graph[%s], there are no indexes that can be inplace.",
           compute_graph->GetName().c_str());
  }

  return SUCCESS;
}
} // namespace ge
