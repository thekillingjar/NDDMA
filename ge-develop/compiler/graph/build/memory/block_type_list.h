/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_TYPE_LIST_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_TYPE_LIST_H_

#include <cstdint>
#include <list>
#include <set>
#include "graph/small_vector.h"
#include "graph/node.h"
#include "ge_common/ge_api_types.h"

namespace ge {
enum class NodeMemAttr {
  kData, // 根图或子图的输入节点
  kConcentrateAtomic, // 需要集中清零的内存
  kMaxLen
};

using NodeMemAttrVector = SmallVector<NodeMemAttr, static_cast<size_t>(NodeMemAttr::kMaxLen)>;

struct NodeTypeIndex;
class MemoryBlock;

class BlockTypeList {
 public:
  BlockTypeList() = default;
  ~BlockTypeList() = default;
  // Copy constructor
  BlockTypeList(const BlockTypeList &other) : all_nodes_mem_attrs_(other.all_nodes_mem_attrs_) {}

  // Move constructor
  BlockTypeList(BlockTypeList &&other) noexcept : all_nodes_mem_attrs_(other.all_nodes_mem_attrs_) {}

  // Copy assignment operator
  BlockTypeList& operator=(const BlockTypeList &other) {
    if (this != &other) {
      all_nodes_mem_attrs_ = other.all_nodes_mem_attrs_;
    }
    return *this;
  }

  // Move assignment operator
  BlockTypeList& operator=(BlockTypeList &&other) noexcept {
    if (this != &other) {
      all_nodes_mem_attrs_ = other.all_nodes_mem_attrs_;
    }
    return *this;
  }

  [[nodiscard]] bool IsConflictWithBlock(const BlockTypeList &other) const;
  [[nodiscard]] bool IsConflictWithOneNode(const NodeTypeIndex &node_type_index) const;
  void WithAdded(const NodeTypeIndex &node_type_index);
  void WithDeleted(const MemoryBlock &memory_block, const NodeTypeIndex &node_type_index);
  [[nodiscard]] std::string ToString() const;
  void swap(BlockTypeList& other) noexcept;

 private:
  std::set<NodeMemAttr> all_nodes_mem_attrs_;
};

class NodeMemAttrUtils {
 public:
  static NodeMemAttrVector GetNodeMemAttrs(const NodeTypeIndex &node_type_index);
  static bool IsConcentrateAtomic(const Node *const node, const int32_t out_index);
  static std::string GetAttrStr(const NodeTypeIndex &node_type_index);
};

class NodeMemAttrConflictRegister {
 public:
  // 仅静态初始化时注册会写，此时单线程无须加锁
 NodeMemAttrConflictRegister(const NodeMemAttr &attr1, const NodeMemAttr &attr2) {
   if ((attr1 < NodeMemAttr::kMaxLen) && (attr2 < NodeMemAttr::kMaxLen)) {
     conflict_matrix_[static_cast<size_t>(attr1)][static_cast<size_t>(attr2)] = true;
     conflict_matrix_[static_cast<size_t>(attr2)][static_cast<size_t>(attr1)] = true;
   }
 }
 // read without lock
 static bool HasConflict(const NodeMemAttr &attr1, const NodeMemAttr &attr2) {
   if ((attr1 < NodeMemAttr::kMaxLen) && (attr2 < NodeMemAttr::kMaxLen)) {
     return conflict_matrix_[static_cast<size_t>(attr1)][static_cast<size_t>(attr2)];
   }
   return false;
 }
 private:
  static std::vector<std::vector<bool>> conflict_matrix_;
};

#define REGISTER_NODE_MEM_ATTR_CONFLICT_COUNTER2(attr1, attr2, counter) \
  static auto g_register_conflict_##counter = NodeMemAttrConflictRegister(attr1, attr2)
#define REGISTER_NODE_MEM_ATTR_CONFLICT_COUNTER(attr1, attr2, counter) \
  REGISTER_NODE_MEM_ATTR_CONFLICT_COUNTER2(attr1, attr2, counter)
#define REGISTER_NODE_MEM_ATTR_CONFLICT(attr1, attr2) REGISTER_NODE_MEM_ATTR_CONFLICT_COUNTER(attr1, attr2, __COUNTER__)
} // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_TYPE_LIST_H_
