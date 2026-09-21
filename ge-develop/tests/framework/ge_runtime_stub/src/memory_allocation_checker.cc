/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check/memory_allocation_checker.h"
#include <sstream>
#include "core/builder/node_types.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "lowering/pass/utils/resource_guarder.h"

namespace gert {
namespace {
bool IsAllocOutputMemory(const char *node_type) {
  return strcmp(node_type, "AllocMemory") == 0 || strcmp(node_type, "AllocMemHbm") == 0 ||
         strcmp(node_type, "AllocMemHost") == 0;
}
bool IsAllocWorkspaceMemory(const char *node_type) {
  return strcmp(node_type, "AllocBatchHbm") == 0;
}
bool IsFreeOutputMemory(const char *node_type) {
  return strcmp(node_type, "FreeMemory") == 0 || strcmp(node_type, "FreeMemHbm") == 0;
}
bool IsFreeWorkspaceMemory(const char *node_type) {
  return strcmp(node_type, "FreeBatchHbm") == 0;
}
}  // namespace
void MemoryAllocationChecker::OnExecuteEvent(int type, void *void_arg, ExecutorEvent event, const void *node,
                                             KernelStatus result) {
  reinterpret_cast<MemoryAllocationChecker *>(void_arg)->OnEvent(event, static_cast<const Node *>(node), result);
}
ge::graphStatus MemoryAllocationChecker::OnEvent(ExecutorEvent event, const ::Node *node, ::KernelStatus result) {
  if (node == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto context = reinterpret_cast<const KernelContext *>(&node->context);
  auto extended_context = reinterpret_cast<const ExtendedKernelContext *>(&node->context);
  auto kernel_type = extended_context->GetKernelType();

  if (event == kExecuteEnd) {
    if (IsAllocOutputMemory(kernel_type)) {
      auto td = context->GetOutputPointer<TensorData>(0);
      AddOperation(MemoryOperationType::kAlloc, td->GetAddr(), extended_context->GetKernelName());
      return ge::GRAPH_SUCCESS;
    }
    if (IsAllocWorkspaceMemory(kernel_type)) {
      auto addrs = context->GetOutputPointer<TypedContinuousVector<ge::MemBlock *>>(0);
      for (size_t i = 0U; i < addrs->GetSize(); ++i) {
        AddOperation(MemoryOperationType::kAlloc, addrs->GetData()[i]->GetAddr(), extended_context->GetKernelName());
      }
      return ge::GRAPH_SUCCESS;
    }
  }

  if (event == kExecuteStart) {
    if (IsFreeOutputMemory(kernel_type)) {
      auto td = context->GetInputPointer<TensorData>(0);
      AddOperation(MemoryOperationType::kFree, td->GetAddr(), extended_context->GetKernelName());
      return ge::GRAPH_SUCCESS;
    }
    if (IsFreeWorkspaceMemory(kernel_type)) {
      auto addrs = context->MutableInputPointer<TypedContinuousVector<ge::MemBlock *>>(0);
      for (size_t i = 0U; i < addrs->GetSize(); ++i) {
        AddOperation(MemoryOperationType::kFree, addrs->GetData()[i]->GetAddr(), extended_context->GetKernelName());
      }
      return ge::GRAPH_SUCCESS;
    }
  }

  if (IsLaunchNode(kernel_type)) {
    AddOperation(MemoryOperationType::kLaunch, nullptr, extended_context->GetKernelName());
    return ge::GRAPH_SUCCESS;
  }

  AddOperation(MemoryOperationType::kNormal, nullptr, extended_context->GetKernelName());

  return ge::GRAPH_SUCCESS;
}
MemoryAllocationChecker::MemoryAllocationChecker(ge::ExecuteGraphPtr exe_graph) : exe_graph_(std::move(exe_graph)) {
  for (const auto &alloc_node : exe_graph_->GetAllNodes()) {
    auto &free_nodes = alloc_nodes_to_free_nodes_[alloc_node->GetName()];
    auto &normal_nodes = alloc_nodes_to_normal_nodes_[alloc_node->GetName()];
    if (IsAllocNode(alloc_node->GetTypePtr())) {
      for (const auto &out_node : alloc_node->GetOutDataNodes()) {
        if (bg::IsGuarderOf(alloc_node, out_node)) {
          free_nodes.insert(out_node->GetName());
        } else {
          normal_nodes.insert(out_node->GetName());
        }

        if(IsLaunchNode(out_node->GetTypePtr())) {
          launch_nodes_to_relevant_alloc_nodes_[out_node->GetName()].insert(alloc_node->GetName());
        }
      }
    }
  }
}
void MemoryAllocationChecker::AddOperation(MemoryOperationType op_type, const void *addr, const char *kernel_name) {
  memory_operations_.push_back({op_type, addr, kernel_name});
  if (addr != nullptr) {
    addresses_to_operations_[addr].push_back({op_type, addr, kernel_name});
  }
}
void MemoryAllocationChecker::Clear() {
  memory_operations_.clear();
  addresses_to_operations_.clear();
}
bool MemoryAllocationChecker::CheckFreeEarlyEnough() const {
  std::stringstream ss;
  std::unordered_map<std::string, const void *> alloc_names_to_addr;
  std::map<const void *, std::string> addrs_to_alloc_name;
  std::set<const void *> ready_to_free;
  std::map<const void *, std::set<std::string>> addrs_to_executed_normal_names;
  std::map<const void *, std::set<std::string>> addrs_to_executed_free_names;
  std::map<const void *, const std::set<std::string> *> addrs_to_all_normal_names;
  std::map<const void *, const std::set<std::string> *> addrs_to_all_free_names;

  std::unordered_map<std::string, const void *> normal_node_names_to_addr;

  for (const auto &op : memory_operations_) {
    if (op.op_type == MemoryOperationType::kAlloc) {
      if (!ready_to_free.empty()) {
        for (const auto &addr : ready_to_free) {
          const auto &executed_free_names = addrs_to_executed_free_names[addr];
          ss << "The address " << addr << " which allocated by " << addrs_to_alloc_name[addr] << " Should be freed by ";
          for (const auto &free_name : alloc_nodes_to_free_nodes_.find(addrs_to_alloc_name[addr])->second) {
            if (executed_free_names.count(free_name) == 0) {
              ss << free_name << ", ";
            }
          }
          ss << "have not been executed, current running alloc node " << op.kernel_name << std::endl;
          GELOGE(ge::GRAPH_FAILED, "%s", ss.str().c_str());
        }
        return false;
      }

      alloc_names_to_addr[op.kernel_name] = op.address;
      addrs_to_alloc_name[op.address] = op.kernel_name;
      addrs_to_all_free_names[op.address] = &(alloc_nodes_to_free_nodes_.find(op.kernel_name)->second);
      addrs_to_all_normal_names[op.address] = &(alloc_nodes_to_normal_nodes_.find(op.kernel_name)->second);
      for (const auto &normal_node_name : *addrs_to_all_normal_names[op.address]) {
        normal_node_names_to_addr[normal_node_name] = op.address;
      }
      continue;
    }

    if (op.op_type == MemoryOperationType::kFree) {
      auto &executed_free_names = addrs_to_executed_free_names[op.address];
      executed_free_names.insert(op.kernel_name);
      if (addrs_to_all_free_names.count(op.address) == 0) {
        // 一个FreeMemory的输入可能不是由alloc node产生的，例如SlitTensor等
        continue;
      }
      if (executed_free_names == *addrs_to_all_free_names[op.address]) {
        ready_to_free.erase(op.address);
      }
      continue;
    }

    if (op.op_type == MemoryOperationType::kNormal || op.op_type == MemoryOperationType::kLaunch) {
      auto iter = normal_node_names_to_addr.find(op.kernel_name);
      if (iter == normal_node_names_to_addr.end()) {
        continue;
      }
      auto &executed_normal_names = addrs_to_executed_normal_names[iter->second];
      executed_normal_names.insert(op.kernel_name);

      if (executed_normal_names == *addrs_to_all_normal_names[iter->second]) {
        ready_to_free.insert(iter->second);
      }
    }
  }
  return true;
}
bool MemoryAllocationChecker::CheckAllocLatelyEnough() const {
  std::unordered_set<std::string> executed_unused_alloc_nodes;
  std::unordered_set<std::string> executed_launch_nodes;
  for (const auto &op : memory_operations_) {
    if (op.op_type == MemoryOperationType::kAlloc) {
      executed_unused_alloc_nodes.insert(op.kernel_name);
      continue;
    }
    if (op.op_type == MemoryOperationType::kLaunch) {
      executed_launch_nodes.insert(op.kernel_name);
      const auto iter = launch_nodes_to_relevant_alloc_nodes_.find(op.kernel_name);
      if (iter == launch_nodes_to_relevant_alloc_nodes_.end()) {
        continue;
      }
      for (const auto &relevant_alloc_node : iter->second) {
        executed_unused_alloc_nodes.erase(relevant_alloc_node);
      }
      if (!executed_unused_alloc_nodes.empty()) {
        std::stringstream ss;
        ss << "The alloc nodes ";
        for (const auto &alloc_node : executed_unused_alloc_nodes) {
          ss << alloc_node.c_str() << ", ";
        }
        ss << " are executed too early, no executed launch nodes need them, executed launch nodes: ";
        for (const auto &launch_node : executed_launch_nodes) {
          ss << launch_node.c_str() << ", ";
        }
        GELOGE(ge::GRAPH_FAILED, "%s", ss.str().c_str());
        return false;
      }
    }
  }
  return true;
}
}  // namespace gert
