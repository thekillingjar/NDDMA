/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atomic_clean_checker.h"
#include "node_checker_utils.h"
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/unfold/graph_unfolder.h"
#include "node_checker_register.h"
#include "graph/ge_context.h"
#include "graph/build/memory/graph_mem_assigner.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr const char *kCleanSeparately = "1";
constexpr int64_t kAllInputAddrIsAtomic = -1;
constexpr size_t kMaxLogCharNum = 1200U;

bool NeedAtomicCleanConcentratively(const Node *const node) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  // 在集中清零的策略下，有些场景的算子也会使用单独清零
  bool is_clean_separately = false;
  (void) ge::AttrUtils::GetBool(op_desc, "need_gentask_atomic", is_clean_separately);
  if (is_clean_separately) {
    return false;
  }

  const auto has_atomic_input = op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  const auto has_atomic_output = op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
  const auto atomic_workspace_index_size = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO,
                                                                  std::map<std::string, std::map<int64_t, int64_t>>{});
  bool is_atomic_node = false;
  // If GetBool fail, is_atomic_node is false.
  (void) ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node);
  if (is_atomic_node || (has_atomic_input || has_atomic_output || (!atomic_workspace_index_size.empty()))) {
    return true;
  }

  return false;
}

std::string GetAtomicCleanAttr(const Node *const node) {
  std::stringstream ss;
  std::vector<int64_t> atomic_input_index;
  (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
  ss << "atomic_input_index[";
  for (size_t i = 0U; i < atomic_input_index.size(); ++i) {
    ss << atomic_input_index.at(i);
    if (i + 1U != atomic_input_index.size()) {
      ss << ", ";
    }
  }
  ss << "], atomic_output_index[";
  std::vector<int64_t> atomic_output_index;
  (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  for (size_t i = 0U; i < atomic_output_index.size(); ++i) {
    ss << atomic_output_index.at(i);
    if (i + 1U != atomic_output_index.size()) {
      ss << ", ";
    }
  }
  ss <<  "], sub_node_workspace_info[";
  const auto atomic_workspace_info = node->GetOpDescBarePtr()->TryGetExtAttr(
      EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
  for (const auto &info : atomic_workspace_info) {
    for (const auto &index_size_pair : info.second) {
      ss << "[index: " <<index_size_pair.first << ", size: " << index_size_pair.second << "]";
    }
  }
  ss << "]";
  return ss.str();
}

Status GetWorkspaceCleanAddrs(const Node *const node,  AtomicNodeCleanTypeVals &type_vals,
                              std::vector<CleanAddrSizeType> &clean_addrs) {
  const auto op_desc = node->GetOpDescBarePtr();
  // 融合算子的workspace清零，没有将地址写到MemSet中
  bool is_fusion_node = false;
  (void) ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
  const auto workspace_offsets = op_desc->GetWorkspace();
  if (workspace_offsets.empty() || is_fusion_node) {
    return SUCCESS;
  }

  const auto workspace_size = node->GetOpDescBarePtr()->GetWorkspaceBytes();
  GE_ASSERT_TRUE(workspace_offsets.size() == workspace_size.size(),
                 "node %s workspace_offsets.size[%zu] != workspace_size.size[%zu]",
                 workspace_offsets.size(), workspace_size.size());

  std::vector<int64_t> tvm_workspace_types;
  const bool has_tvm_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);

  std::vector<int64_t> workspace_type_list;
  const bool has_workspace_type_list_attr =
      ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);

  const auto atomic_workspace_index_size = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO,
                                                                  std::map<std::string, std::map<int64_t, int64_t>>{});
  for (const auto &workspace_info : atomic_workspace_index_size) {
    for (const auto &index_size_pair : workspace_info.second) {
      const auto index = static_cast<size_t>(index_size_pair.first);
      GE_ASSERT_TRUE(index < workspace_offsets.size());
      CleanMemInfo mem_info;
      GE_ASSERT_SUCCESS(type_vals.GetNextAttr(mem_info.type_val), "get next attr failed.");
      if (has_tvm_workspace_mem_type_attr && (index < tvm_workspace_types.size())) {
        // 这两种类型不分配内存
        if ((tvm_workspace_types.at(index) == RT_MEM_TYPE_L1) || (tvm_workspace_types.at(index) == kRtMemoryUB)) {
          continue;
        }
      }
      // ascend c算子的workspace 默认-1，要跳过
      if (workspace_size.at(index) < 0) {
        continue;
      }
      mem_info.offset = workspace_offsets.at(index);
      mem_info.size = workspace_size.at(index);
      if (has_workspace_type_list_attr && (index < workspace_type_list.size())) {
        mem_info.memory_type = workspace_type_list.at(index) == RT_MEMORY_P2P_DDR ? RT_MEMORY_P2P_DDR : RT_MEMORY_HBM;
      }
      clean_addrs.emplace_back(CleanAddrSizeType{kWorkspace, static_cast<int64_t>(index), mem_info});
    }
  }
  return SUCCESS;
}

std::vector<int32_t> GetMemsetDataTypeList(const Node *atomic_node) {
  std::vector<int32_t> data_type_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list);
  return data_type_list;
}
std::vector<int64_t> GetMemsetWorkspaceTypeList(const Node *atomic_node) {
  std::vector<int64_t> type_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_WORKSPACE_TYPE_LIST, type_list);
  return type_list;
}

std::vector<float32_t> GetMemsetFloatValList(const Node *atomic_node) {
  std::vector<float32_t> float_list;
  (void) AttrUtils::GetListFloat(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT, float_list);
  return float_list;
}

std::vector<int64_t> GetMemsetIntValList(const Node *atomic_node) {
  std::vector<int64_t> int_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list);
  return int_list;
}

Status GetMemsetNodeAddrAndAttr(const Node *node, MemsetNodeAddrAndAttr &addr_attr) {
  const auto op_desc = node->GetOpDescBarePtr();
  addr_attr.offsets = op_desc->GetWorkspace();
  addr_attr.sizes = op_desc->GetWorkspaceBytes();
  GE_ASSERT_TRUE(addr_attr.offsets.size() == addr_attr.sizes.size(),
                 "memset %s offsets.size[%zu] is not equal to workspace size[%zu]", node->GetNamePtr(),
                 addr_attr.offsets.size(), addr_attr.sizes.size());

  addr_attr.memory_types = GetMemsetWorkspaceTypeList(node);
  if (!addr_attr.memory_types.empty()) {
    GE_ASSERT_TRUE(addr_attr.memory_types.size() == addr_attr.offsets.size(),
                   "memset %s memory_types.size[%zu] is not equal to workspace size[%zu]", node->GetNamePtr(),
                   addr_attr.memory_types.size(), addr_attr.offsets.size());
  }
  auto float_list = GetMemsetFloatValList(node);
  auto int_list = GetMemsetIntValList(node);
  addr_attr.data_type_list = GetMemsetDataTypeList(node);
  GE_ASSERT_TRUE(addr_attr.data_type_list.size() == addr_attr.offsets.size(),
                 "memset %s data_type_list.size[%zu] is not equal to workspace.size[%zu], data_type_list:%s,"
                 " workspace:%s", node->GetNamePtr(), addr_attr.data_type_list.size(), addr_attr.offsets.size(),
                 ToString(addr_attr.data_type_list).c_str(), ToString(addr_attr.offsets).c_str());
  GE_ASSERT_TRUE((float_list.size() + int_list.size()) == addr_attr.data_type_list.size(),
                 "memset %s float value list size[%zu] + int value list size[%zu] is not equal to data_type_list"
                 " size[%zu]", node->GetNamePtr(), float_list.size(), int_list.size(),
                 addr_attr.data_type_list.size());
  size_t float_index = 0U;
  size_t int_index = 0U;
  // 这里把float_list/int_list的长度和data_type_list长度填充成一样，方便使用index索引
  addr_attr.float_list.resize(addr_attr.data_type_list.size());
  addr_attr.int_list.resize(addr_attr.data_type_list.size());
  for (size_t i = 0U; i < addr_attr.data_type_list.size(); ++i) {
    const auto data_type = addr_attr.data_type_list.at(i);
    if (IsFloatType(static_cast<ge::DataType>(data_type))) {
      GE_ASSERT_TRUE(float_index < float_list.size(), "float_index[%zu] >= float_list size[%zu], i:%zu",
                     float_index, float_list.size(), i);
      addr_attr.float_list[i] = float_list.at(float_index++);
    } else {
      GE_ASSERT_TRUE(int_index < int_list.size(), "int_index[%zu] >= int_list size[%zu], i:%zu",
                     int_index, int_list.size(), i);
      addr_attr.int_list[i] = int_list.at(int_index++);
    }
  }
  return SUCCESS;
}
void FindMemSets(const Node *atomic_node, std::vector<Node *> &memset_nodes) {
  for (const auto peer_out_ctr_anchor : atomic_node->GetInControlAnchor()->GetPeerOutControlAnchorsPtr()) {
    auto ctr_in_node = peer_out_ctr_anchor->GetOwnerNode();
    if (NodeUtils::IsLikeAtomicClean(ctr_in_node)) {
      memset_nodes.emplace_back(ctr_in_node.get());
    }
  }
}
}

Status AtomicCleanChecker::Check(const ge::ComputeGraphPtr &graph) {
  // 只检查集中清零
  std::string atomic_clean_policy;
  const bool need_clean_separately = (GetContext().GetOption(ge::ATOMIC_CLEAN_POLICY, atomic_clean_policy) == SUCCESS)
      && (atomic_clean_policy == kCleanSeparately);
  if (need_clean_separately) {
    GELOGI("atomic clean separately, no need to check, graph: %s", graph->GetName().c_str());
    return SUCCESS;
  }
  for (const auto node : graph->GetAllNodesPtr()) {
    if (NodeUtils::IsLikeAtomicClean(node->shared_from_this())) {
      GE_ASSERT_SUCCESS(GetMemSetAddrs(node), "node: %s get memset addrs failed", node->GetNamePtr());
      continue;
    }
    if (!NeedAtomicCleanConcentratively(node)) {
      continue;
    }
    // 查出要清理的地址段
    std::vector<CleanAddrSizeType> clean_addrs;
    GE_ASSERT_SUCCESS(GetCleanAddrs(node, clean_addrs),
                      "node %s need atomic clean, but get clean address info failed, atomic attr: %s",
                      node->GetNamePtr(), GetAtomicCleanAttr(node).c_str());
    std::vector<Node *> memset_nodes;
    FindMemSets(node, memset_nodes);
    // 检查要清理的输入输出workspace地址是不是都在MemSet的workspace地址范围内
    for (const auto &clean_addr : clean_addrs) {
      if (!FindInMemSetOffsets(memset_nodes, clean_addr)) {
        PrintMemSet(memset_nodes);
        (void)FindInMemSetOffsets(memset_nodes, clean_addr, true);
        REPORT_INNER_ERR_MSG("E19999", "graph %s node %s check atomic clean address failed, need to clean %s",
                           graph->GetName().c_str(), node->GetNamePtr(), clean_addr.ToString().c_str());
        GELOGE(FAILED, "graph %s node %s check atomic clean address failed, need to clean %s",
               graph->GetName().c_str(), node->GetNamePtr(), clean_addr.ToString().c_str());
        const auto attrs = GetAtomicCleanAttr(node);
        GELOGE(FAILED, "node: %s attrs: %s", node->GetNamePtr(), attrs.c_str());
        return FAILED;
      }
    }
  }
  GELOGI("atomic clean check success, graph: %s", graph->GetName().c_str());
  return SUCCESS;
}

Status AtomicCleanChecker::GetMemSetAddrs(const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
  MemsetNodeAddrAndAttr addr_attr(node->GetOpDescBarePtr()->GetWorkspace().size());
  GE_ASSERT_SUCCESS(GetMemsetNodeAddrAndAttr(node, addr_attr),
                    "get memset node address and attrs failed, node: %s", node->GetNamePtr());
  auto &clean_mem_infos = memset_to_clean_infos_[node];
  for (size_t i = 0U; i < addr_attr.offsets.size(); ++i) {
    CleanMemInfo mem_info;
    mem_info.offset = addr_attr.offsets.at(i);
    mem_info.size = addr_attr.sizes.at(i);
    mem_info.memory_type = addr_attr.memory_types.empty() ? RT_MEMORY_HBM : addr_attr.memory_types.at(i);
    mem_info.type_val.data_type = addr_attr.data_type_list.at(i);
    mem_info.type_val.float_val = addr_attr.float_list.at(i);
    mem_info.type_val.int_val = addr_attr.int_list.at(i);
    clean_mem_infos.insert(mem_info);
  }
  return SUCCESS;
}

Status AtomicCleanChecker::GetCleanAddrs(const Node *const node, std::vector<CleanAddrSizeType> &clean_addrs) const {
  GE_ASSERT_SUCCESS(GetInputCleanAddrs(node, clean_addrs), "node: %s get input clean addrs failed", node->GetNamePtr());
  AtomicNodeCleanTypeVals type_vals;
  GE_ASSERT_SUCCESS(type_vals.Init(node), "data type and value attrs init failed, node: %s(%s)",
                    node->GetNamePtr(), node->GetTypePtr());
  GE_ASSERT_SUCCESS(GetOutputCleanAddrs(node, type_vals, clean_addrs),
                    "node: %s get out clean addrs failed", node->GetNamePtr());
  GE_ASSERT_SUCCESS(GetWorkspaceCleanAddrs(node, type_vals, clean_addrs), "node: %s get workspace clean addrs failed.",
                    node->GetNamePtr());
  return SUCCESS;
}

Status AtomicCleanChecker::GetInputCleanAddrs(const Node *const node,
                                              std::vector<CleanAddrSizeType> &clean_addrs) const {
  const auto op_desc = node->GetOpDescBarePtr();
  const auto input_offsets = op_desc->GetInputOffset();

  std::vector<int64_t> atomic_input_index;
  (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
  GE_ASSERT_TRUE(input_offsets.size() >= atomic_input_index.size(),
                 "node %s input_offsets.size[%zu] < atomic_input_index.size[%zu]", node->GetNamePtr(),
                 input_offsets.size(), atomic_input_index.size());
  if ((atomic_input_index.size() == 1U) && (atomic_input_index.at(0) == kAllInputAddrIsAtomic)) {
    atomic_input_index.clear();
    for (int64_t i = 0; static_cast<size_t>(i) < input_offsets.size(); ++i) {
      atomic_input_index.emplace_back(i);
    }
  }
  for (const auto index : atomic_input_index) {
    GE_ASSERT_TRUE(static_cast<size_t>(index) < input_offsets.size(),
                   "node %s atomic_input_index[%lld] >= input_offsets.size[%zu]",
                   node->GetNamePtr(), index, input_offsets.size());
    CleanMemInfo clean_mem;
    clean_mem.offset = input_offsets.at(index);
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetInputSize(node, index, clean_mem.size));
    GE_ASSERT_SUCCESS(GetMemType(node, kIn, index, clean_mem.memory_type),
                      "node %s get output memory type failed, index: %lld", node->GetNamePtr(), index);
    clean_addrs.emplace_back(CleanAddrSizeType{kInput, index, clean_mem});
  }
  return SUCCESS;
}

Status AtomicCleanChecker::GetOutputCleanAddrs(const Node *const node, AtomicNodeCleanTypeVals &type_vals,
                                               std::vector<CleanAddrSizeType> &clean_addrs) const {
  const auto op_desc = node->GetOpDescBarePtr();
  const auto output_offsets = op_desc->GetOutputOffset();

  std::vector<int64_t> atomic_output_index;
  (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  GE_ASSERT_TRUE(output_offsets.size() >= atomic_output_index.size(),
                 "node %s output_offsets.size[%zu] < atomic_output_index.size[%zu]", node->GetNamePtr(),
                 output_offsets.size(), atomic_output_index.size());
  for (const auto index : atomic_output_index) {
    GE_ASSERT_TRUE(static_cast<size_t>(index) < output_offsets.size(),
                   "node %s atomic_output_index[%lld] >= output_offsets.size[%zu]",
                   node->GetNamePtr(), index, output_offsets.size());
    CleanMemInfo mem_info;
    mem_info.offset = output_offsets.at(index);
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputSize(node, index, mem_info.size));
    GE_ASSERT_SUCCESS(GetMemType(node, kOut, index, mem_info.memory_type),
                      "node %s get input memory type failed, index: %lld", node->GetNamePtr(), index);
    GE_ASSERT_SUCCESS(type_vals.GetNextAttr(mem_info.type_val), "get next attr failed");
    clean_addrs.emplace_back(CleanAddrSizeType{kOutput, index, mem_info});
  }
  return SUCCESS;
}

Status AtomicCleanChecker::GetMemType(const Node *const node, const IOType &io_type, const uint32_t index,
                                      uint32_t &mem_type) const {
  return graph_memory_assigner_->GetMemType(node, io_type, index, mem_type);
}

bool AtomicCleanChecker::FindInMemSetOffsets(const std::vector<Node *> &memset_nodes,
                                             const CleanAddrSizeType &addr_size_type,
                                             const bool error_mode) const {
  for (const auto memset_node : memset_nodes) {
    const auto &memset_iter = memset_to_clean_infos_.find(memset_node);
    GE_ASSERT_TRUE(memset_iter != memset_to_clean_infos_.end(),
                   "can not find %s in memset_to_clean_infos_", memset_node->GetNamePtr());
    const auto &memset_clean_infos = memset_iter->second;
    // 找到第一个大于等于被清理的offset的
    auto mem_info_iter = memset_clean_infos.lower_bound(addr_size_type.mem_info);
    if ((mem_info_iter != memset_clean_infos.end()) && (mem_info_iter->Contain(addr_size_type.mem_info))) {
      return true;
    } else if (error_mode) {
      GELOGE(FAILED, "%s", mem_info_iter != memset_clean_infos.end() ? mem_info_iter->ToStr().c_str() :
                                                                     "mem_info_iter == memset_clean_infos.end()");
    }
    while (mem_info_iter != memset_clean_infos.begin()) {
      --mem_info_iter;
      if (mem_info_iter->Contain(addr_size_type.mem_info)) {
        return true;
      } else if (error_mode) {
        GELOGE(FAILED, "Contain return false, %s", mem_info_iter->ToStr().c_str());
      }
    }
  }
  return false;
}

void AtomicCleanChecker::PrintMemSet(const std::vector<Node *> &memset_nodes) const {
  std::stringstream ss0;
  ss0 << "connect to atomic node memset_nodes: ";
  for (const auto &memset_node : memset_nodes) {
    ss0 << "[" << memset_node->GetType() << ", " << memset_node->GetNamePtr() << "]";
  }
  GELOGE(FAILED, "%s%s", ss0.str().c_str(), memset_nodes.empty() ? "[]" : ".");
  for (const auto &iter : memset_to_clean_infos_) {
    std::stringstream ss;
    ss << "[" << iter.first->GetType() << "]" << iter.first->GetNamePtr();
    for (const auto &clean_mem_info : iter.second) {
      ss << "[" << clean_mem_info.ToStr() << "]";
      if (ss.str().length() > kMaxLogCharNum) {
        GELOGE(FAILED, "%s", ss.str().c_str());
        ss.str("");
        ss.clear();
      }
    }
    GELOGE(FAILED, "%s%s", ss.str().c_str(), iter.second.empty() ? " has no clean address." : ".");
  }
}
}  // namespace ge