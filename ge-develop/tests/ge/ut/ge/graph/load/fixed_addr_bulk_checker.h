/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_FIXED_ADDR_BULK_CHECKER_H_
#define AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_FIXED_ADDR_BULK_CHECKER_H_
#include <set>
#include "stub/gert_runtime_stub.h"
#include "graph/load/model_manager/task_args_refresh_type_classifier.h"
#include "task_args_refresh_type_classifier_common_header.h"
namespace ge {
class StubMemoryBlockManager : public MemoryBlockManager {
 public:
  explicit StubMemoryBlockManager(const uint32_t mem_type = RT_MEMORY_HBM)
      : MemoryBlockManager(mem_type), memory_type_(mem_type) {
  }
  ~StubMemoryBlockManager() = default;
  void *Malloc(const std::string &purpose, const size_t size) override {
    auto ptr = MemoryBlockManager::Malloc(purpose, size);
    addrs_to_mem_info_[(void *)ptr] = gert::RuntimeStubImpl::MemoryInfo{(void *)ptr, static_cast<size_t>(size),
                                                                        memory_type_, GE_MODULE_NAME_U16};
    GELOGI("stub malloc success, ptr: %p, size: %zu, mem_type: %d", ptr, size, memory_type_);
    return ptr;
  }
  const std::unordered_map<void *, gert::RuntimeStubImpl::MemoryInfo> &GetAllocatedRtsMemory() const {
    return addrs_to_mem_info_;
  }
 private:
  const rtMemType_t memory_type_;
  std::unordered_map<void *, gert::RuntimeStubImpl::MemoryInfo> addrs_to_mem_info_;
};

class FixedAddrBulkChecker {
 public:
  FixedAddrBulkChecker(const gert::GertRuntimeStub &runtime_stub, StubMemoryBlockManager *allocator,
                       const ModelArgsManager::FixedAddrBulk &fab,
                       const std::unordered_map<std::string, std::vector<size_t>> &node_names_to_task_indexes)
      : runtime_stub_(runtime_stub), allocator_(allocator), fab_(fab),
        node_names_to_task_indexes_(node_names_to_task_indexes) {}

  bool Check(std::string &ret) const {
    std::stringstream ss;
    auto mem_info = CheckExists(ss);
    if (mem_info == nullptr) {
      ret = ss.str();
      return false;
    }
    auto result = CheckAllPiecesAddrInBaseRange(mem_info, ss);
    result &= CheckAllPiecesConnectionCorrect(ss);
    ret = ss.str();
    return result;
  }

  struct FixedMemoryInput {
    const char *node_name;
    TaskArgsRefreshTypeClassifier::IndexType iow_type;
    size_t index;
  };

  FixedAddrBulkChecker &FixedMemory(std::vector<FixedMemoryInput> same_addrs) {
    std::set<TaskArgsRefreshTypeClassifier::TaskFixedAddr> fixed_memories;
    for (const auto &fixed_mem : same_addrs) {
      fixed_memories.insert(TaskArgsRefreshTypeClassifier::TaskFixedAddr{
          node_names_to_task_indexes_.at(fixed_mem.node_name).at(0), fixed_mem.index, fixed_mem.iow_type});
    }
    same_addr_fixed_mems_.emplace_back(std::move(fixed_memories));
    return *this;
  }

 private:
  const gert::RuntimeStubImpl::MemoryInfo *CheckExists(std::stringstream &ss) const {
    if (fab_.device_addr == nullptr) {
      ss << "Fixed addr bulk does not exists";
      return nullptr;
    }
    const gert::RuntimeStubImpl::MemoryInfo *mem_info = nullptr;
    if (allocator_ != nullptr) {
      auto &addrs_to_mem_info = allocator_->GetAllocatedRtsMemory();
      auto base_addr = fab_.device_addr;
      auto iter = addrs_to_mem_info.find(base_addr);
      if (iter == addrs_to_mem_info.end()) {
        ss << "Fixed memory addr " << std::hex << base_addr << "does not exists from allocator";
        return nullptr;
      }
      mem_info = &iter->second;
    } else {
      auto &addrs_to_mem_info = runtime_stub_.GetRtsRuntimeStub().GetAllocatedRtsMemory();
      auto base_addr = fab_.device_addr;
      auto iter = addrs_to_mem_info.find(base_addr);
      if (iter == addrs_to_mem_info.end()) {
        ss << "Fixed memory addr " << std::hex << base_addr << "does not exists from rts";
        return nullptr;
      }
      mem_info = &iter->second;
    }
    if (mem_info->rts_mem_type != RT_MEMORY_TS) {
      ss << "Invalid rts memory type " << std::hex << mem_info->rts_mem_type << ", expect RT_MEMORY_TS";
      return nullptr;
    }
    return mem_info;
  }
  bool CheckAllPiecesAddrInBaseRange(const gert::RuntimeStubImpl::MemoryInfo *mem_info, std::stringstream &ss) const {
    auto base_addr = PtrToValue(fab_.device_addr);
    auto len = mem_info->size;

    for (const auto &piece : fab_.pieces) {
      if (piece.device_addr < base_addr) {
        ss << "Invalid device addr " << std::hex << piece.device_addr << ", task index " << piece.desc.task_index
           << TaskArgsRefreshTypeClassifier::GetIndexTypeStr(piece.desc.iow_index_type) << " " << piece.desc.iow_index
           << ", less than base addr " << std::hex << base_addr;
        return false;
      }
      if (piece.device_addr > base_addr + len) {
        ss << "Invalid device addr " << std::hex << piece.device_addr << ", task index " << piece.desc.task_index
           << TaskArgsRefreshTypeClassifier::GetIndexTypeStr(piece.desc.iow_index_type) << " " << piece.desc.iow_index
           << ", out of range " << std::hex << base_addr << "~" << std::hex << base_addr + len;
        return false;
      }
    }
    return true;
  }
  bool CheckAllPiecesConnectionCorrect(std::stringstream &ss) const {
    std::unordered_map<uint64_t, std::set<TaskArgsRefreshTypeClassifier::TaskFixedAddr>> addrs_to_descs;
    for (const auto &piece : fab_.pieces) {
      addrs_to_descs[piece.device_addr].insert(piece.desc);
    }
    if (same_addr_fixed_mems_.size() != addrs_to_descs.size()) {
      ss << "fixed memory addr count is wrong " << addrs_to_descs.size() << ", expect " << same_addr_fixed_mems_.size();
      return false;
    }
    std::vector<bool> correct(same_addr_fixed_mems_.size(), false);
    for (const auto &addr_and_descs : addrs_to_descs) {
      for (size_t i = 0UL; i < same_addr_fixed_mems_.size(); ++i) {
        if (correct[i]) {
          continue;
        }
        if (same_addr_fixed_mems_[i].count(*(addr_and_descs.second.begin())) > 0) {
          if (same_addr_fixed_mems_[i] != addr_and_descs.second) {
            ss << "Unexpected fixed addrs " << std::endl;
            return false;
          }
          correct[i] = true;
        }
      }
    }

    for (size_t i = 0UL; i < correct.size(); ++i) {
      if (!correct[i]) {
        ss << "fixed addr index " << i << " does not correct " << std::endl;
        return false;
      }
    }
    return true;
  }

 private:
  const gert::GertRuntimeStub &runtime_stub_;
  const StubMemoryBlockManager *allocator_;
  const ModelArgsManager::FixedAddrBulk &fab_;
  const std::unordered_map<std::string, std::vector<size_t>> &node_names_to_task_indexes_;
  // 每个vector元素代表相同fixed地址的所有piece
  std::vector<std::set<TaskArgsRefreshTypeClassifier::TaskFixedAddr>> same_addr_fixed_mems_;
};
}
#endif  // AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_FIXED_ADDR_BULK_CHECKER_H_
