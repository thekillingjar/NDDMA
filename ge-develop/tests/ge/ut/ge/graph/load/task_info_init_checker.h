/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_INIT_CHECKER_H_
#define AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_INIT_CHECKER_H_
#include "task_info_stubs.h"
namespace ge {
class TaskInfoInitChecker {
 public:
  TaskInfoInitChecker(size_t task_index, StubTaskInfo *task_info) : task_index_(task_index), task_info_(task_info) {}
  TaskInfoInitChecker &ArgsMemory(const gert::RuntimeStubImpl::MemoryInfo *args_mem) {
    args_memory_infos_.emplace_back(args_mem);
    return *this;
  }
  TaskInfoInitChecker &FixedAddrBulk(const ModelArgsManager::FixedAddrBulk &fixed_addr_bulk) {
    for (const auto &piece : fixed_addr_bulk.pieces) {
      if (piece.desc.task_index != task_index_) {
        continue;
      }
      iow_types_to_indexes_to_fixed_addr_[piece.desc.iow_index_type][piece.desc.iow_index] = piece.device_addr;
    }
    return *this;
  }
  bool GeneralCheck(std::string &ret) const {
    std::stringstream ss;
    ss << "Task index " << task_index_ << ":" << std::endl;
    auto result = CheckCallTimes(ss);
    result &= CheckArgsLengthOk(ss);
    result &= CheckPersistentWorkspaceOk(ss);
    result &= CheckArgsPlacementCorrect(ss);
    ret = ss.str();
    return result;
  }

  bool CheckInputAddrs(std::string &ret, const std::vector<std::pair<MemoryAppType, bool>> &input_addr_descs) {
    std::stringstream ss;

    ss << "Task index " << task_index_ << " input:" << std::endl;
    auto result =
        CheckAddrs(ss, task_info_->GetInitCalls().at(0).iow_addrs.input_logic_addrs, task_info_->GetGenInputAddrs(),
                   input_addr_descs, iow_types_to_indexes_to_fixed_addr_[TaskArgsRefreshTypeClassifier::kInput]);
    ret = ss.str();
    return result;
  }

  bool CheckOutputAddrs(std::string &ret, const std::vector<std::pair<MemoryAppType, bool>> &addr_descs) {
    std::stringstream ss;
    ss << "Task index " << task_index_ << " output:" << std::endl;
    auto result =
        CheckAddrs(ss, task_info_->GetInitCalls().at(0).iow_addrs.output_logic_addrs, task_info_->GetGenOutputAddrs(),
                   addr_descs, iow_types_to_indexes_to_fixed_addr_[TaskArgsRefreshTypeClassifier::kOutput]);
    ret = ss.str();
    return result;
  }

  bool CheckWsAddrs(std::string &ret, const std::vector<std::pair<MemoryAppType, bool>> &addr_descs) {
    std::stringstream ss;
    ss << "Task index " << task_index_ << " workspace:" << std::endl;
    auto result = CheckAddrs(ss, task_info_->GetInitCalls().at(0).iow_addrs.workspace_logic_addrs,
                             task_info_->GetGenWsAddrs(), addr_descs,
                             iow_types_to_indexes_to_fixed_addr_[TaskArgsRefreshTypeClassifier::kWorkspace]);
    ret = ss.str();
    return result;
  }

 private:
  bool CheckCallTimes(std::stringstream &ss) const {
    if (task_info_->GetInitCalls().size() != init_call_times_) {
      ss << "Unexpect TaskInfo::Init call times " << task_info_->GetInitCalls().size() << ", expect " << init_call_times_
         << std::endl;
      return false;
    }
    return true;
  }

  bool CheckArgsLengthOk(std::stringstream &ss) const {
    if (task_info_->GetInitCalls().size() != 1UL) {
      return false;
    }
    auto expect_placements = task_info_->GetArgsPlacements();
    std::set<ArgsPlacement> expect_placements_set(expect_placements.begin(), expect_placements.end());
    for (int32_t i = 0; i < static_cast<int32_t>(ArgsPlacement::kEnd); ++i) {
      auto &args = task_info_->GetInitCalls().at(0).args.at(i);
      auto plm = static_cast<ArgsPlacement>(i);
      if (expect_placements_set.count(plm) > 0) {
        if (args.dev_addr == 0UL) {
          ss << "No " << GetArgsPlacementStr(plm) << " dev args in task" << std::endl;
          return false;
        }
        if (args.host_addr == nullptr) {
          ss << "No " << GetArgsPlacementStr(plm) << " host args in task" << std::endl;
          return false;
        }
        if (args.len < task_info_->GetGenTaskArgsLen()) {
          ss << "Invalid " << GetArgsPlacementStr(plm) << " args len " << args.len << ", at least "
             << task_info_->GetGenTaskArgsLen() << std::endl;
          return false;
        }
      } else {
        if (args.dev_addr != 0UL) {
          ss << "Unexpect existing dev args " << GetArgsPlacementStr(plm) << ", len " << args.len << std::endl;
          return false;
        }
        if (args.host_addr != nullptr) {
          ss << "Unexpect existing host args " << GetArgsPlacementStr(plm) << ", len " << args.len << std::endl;
          return false;
        }
      }
    }
    return true;
  }
  bool CheckArgsPlacementCorrect(std::stringstream &ss) const {
    if (task_info_->GetInitCalls().size() != 1UL) {
      return false;
    }
    for (const auto &arg : task_info_->GetInitCalls().at(0).args) {
      if (arg.dev_addr == 0 && arg.host_addr == nullptr && arg.len == 0) {
        continue;
      }
      bool found_and_ok = false;
      for (const auto &model_arg : args_memory_infos_) {
        auto dev_model_arg_addr = (uint64_t)(model_arg->addr);
        if (dev_model_arg_addr > arg.dev_addr) {
          continue;
        }
        if (dev_model_arg_addr + model_arg->size < arg.dev_addr + arg.len) {
          ss << "The task arg out of range (" << std::hex << arg.dev_addr << "+" << arg.len << "=" << std::hex
             << arg.dev_addr + arg.len << ") > (" << std::hex << dev_model_arg_addr << "+" << model_arg->size << "="
             << std::hex << dev_model_arg_addr + model_arg->size << std::endl;
          return false;
        }
        found_and_ok = true;
      }
      if (!found_and_ok) {
        ss << "The args addr 0x" << std::hex << arg.dev_addr << " can not be found in model args table" << std::endl;
        return false;
      }
    }
    return true;
  }

  bool CheckPersistentWorkspaceOk(std::stringstream &ss) const {
    if (task_info_->GetInitCalls().size() != 1UL) {
      return false;
    }
    for (int32_t i = static_cast<int32_t>(ArgsPlacement::kArgsPlacementHbm);
         i < static_cast<int32_t>(ArgsPlacement::kEnd); ++i) {
      auto &pw = task_info_->GetInitCalls().at(0).persistent_workspace[i];
      if (pw.dev_addr != 0UL) {
        ss << "Unexpect existing persistent workspace " << GetArgsPlacementStr(static_cast<ArgsPlacement>(i))
           << ", len " << pw.len << std::endl;
        return false;
      }
    }
    return true;
  }

  bool CheckAddrs(std::stringstream &ss, const std::vector<AddrDesc> &addrs, const std::vector<uint64_t> &gen_addrs,
                  const std::vector<std::pair<MemoryAppType, bool>> &input_addr_descs,
                  const std::map<int32_t, uint64_t> &indexes_to_fixed_addr) const {
    if (addrs.size() != input_addr_descs.size()) {
      ss << "Wrong iow.size " << addrs.size() << ", expect " << input_addr_descs.size() << std::endl;
      return false;
    }
    for (size_t i = 0UL; i < addrs.size(); ++i) {
      auto expect_addr = gen_addrs[i];
      auto iter = indexes_to_fixed_addr.find(static_cast<int32_t>(i));
      if (iter != indexes_to_fixed_addr.end()) {
        expect_addr = iter->second;
      }
      if (addrs[i].logic_addr != expect_addr) {
        ss << "Wrong logical addr " << std::hex << addrs[i].logic_addr << ", expect " << std::hex << expect_addr
           << ", index " << i << std::endl;
        return false;
      }
      if (addrs[i].memory_type != static_cast<uint64_t>(input_addr_descs.at(i).first)) {
        ss << "Wrong memory app type " << addrs[i].memory_type << ", expect "
           << static_cast<uint64_t>(input_addr_descs.at(i).first) << ", index " << i << std::endl;
        return false;
      }
      if (addrs[i].support_refresh != input_addr_descs.at(i).second) {
        ss << "Wrong refresh type " << addrs[i].support_refresh << ", expect " << input_addr_descs.at(i).second
           << std::endl;
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<const gert::RuntimeStubImpl::MemoryInfo *> args_memory_infos_;
  size_t task_index_;
  StubTaskInfo *task_info_;
  size_t init_call_times_{1};
  size_t init_host_args_call_times{1};
  std::map<TaskArgsRefreshTypeClassifier::IndexType, std::map<int32_t, uint64_t>> iow_types_to_indexes_to_fixed_addr_;
};
#define TASK_INFO_CHECKER(task_list, index) \
  TaskInfoInitChecker(index, reinterpret_cast<StubTaskInfo *>(task_list.at(index).get()))

}
#endif  // AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_INIT_CHECKER_H_
