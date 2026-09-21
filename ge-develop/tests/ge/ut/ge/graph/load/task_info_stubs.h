/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_STUBS_H_
#define AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_STUBS_H_
#include "graph/load/model_manager/task_info/task_info.h"
namespace ge {
namespace {
uint32_t GenTaskId() {
  static std::atomic<uint32_t> next_task_id{500};
  return next_task_id++;
}
}
struct InitCall {
  PisToArgs args;
  PisToPersistentWorkspace persistent_workspace;
  IowAddrs iow_addrs;
};
struct UpdateHostArgsCall {
  std::vector<uint64_t> active_mem_base_addr;
  std::vector<HostArg> host_args;
};
struct UpdateDumpInfosCall {
  std::vector<HostArg> host_args;
};
class StubTaskInfo : public TaskInfo {
 public:
  explicit StubTaskInfo(TaskInfoRegistryStub *registry) : registry_(registry) {}
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override {
    return task_def.kernel().context().op_index();
  }
  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override {
    auto &ios_to_symbol = registry_->GetIOsToSymbol();
    auto &symbols_to_addr = registry_->GetSymbolsToAddr();
    auto &op_indexes_to_ws_addrs = registry_->GetWsAddrs();
    auto op_index = ParseOpIndex(task_def);
    int32_t io_num = 0;
    for (int32_t i = 0;; ++i, ++io_num) {
      OpIndexAndIo opio{kIn, i, op_index};
      auto iter = ios_to_symbol.find(opio);
      if (iter == ios_to_symbol.end()) {
        break;
      }
      auto addr = symbols_to_addr.at(iter->second);
      gen_input_addrs_.emplace_back(addr);
      task_run_param.parsed_input_addrs.emplace_back(AddrDesc{addr, RT_MEMORY_HBM, SupportRefresh(), {}});
    }
    for (int32_t i = 0;; ++i, ++io_num) {
      OpIndexAndIo opio{kOut, i, op_index};
      auto iter = ios_to_symbol.find(opio);
      if (iter == ios_to_symbol.end()) {
        break;
      }
      auto addr = symbols_to_addr.at(iter->second);
      gen_output_addrs_.emplace_back(addr);
      task_run_param.parsed_output_addrs.emplace_back(AddrDesc{addr, RT_MEMORY_HBM, SupportRefresh(), {}});
    }

    auto iter = op_indexes_to_ws_addrs.find(op_index);
    if (iter != op_indexes_to_ws_addrs.end()) {
      for (auto addr : iter->second) {
        ++io_num;
        gen_ws_addrs_.emplace_back(addr);
        task_run_param.parsed_workspace_addrs.emplace_back(AddrDesc{addr, RT_MEMORY_HBM, SupportRefresh(), {}});
      }
    }

    gen_task_args_len_ = 8 * io_num;

    for (auto args_placement : GetArgsPlacements()) {
      task_run_param.args_descs.emplace_back(TaskArgsDesc{gen_task_args_len_, args_placement});
    }
    return SUCCESS;
  }
  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
              const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) override {
    std::vector<MemAllocation> logical_mem_allocations = davinci_model->GetLogicalMemAllocation();
    const auto match_id = [&logical_mem_allocations] (const uint64_t addr) -> size_t {
      for (auto &item : logical_mem_allocations) {
        if ((addr >= item.logical_addr) && (addr < (item.logical_addr + item.data_size))) {
          return static_cast<size_t>(item.id);
        }
      }
      return logical_mem_allocations.size();
    };

    std::vector<AddrDesc> addr_desc;
    addr_desc.insert(addr_desc.end(), iow_addrs.input_logic_addrs.begin(), iow_addrs.input_logic_addrs.end());
    addr_desc.insert(addr_desc.end(), iow_addrs.output_logic_addrs.begin(), iow_addrs.output_logic_addrs.end());
    addr_desc.insert(addr_desc.end(), iow_addrs.workspace_logic_addrs.begin(), iow_addrs.workspace_logic_addrs.end());
    for (const auto &addr : addr_desc) {
      const uint64_t id = match_id(addr.logic_addr);
      MemAllocationAndOffset id_and_offset =
      {id, (addr.logic_addr - logical_mem_allocations[id].logical_addr), logical_mem_allocations[id].type};
      v_mem_allocation_id_and_offset_.emplace_back(std::move(id_and_offset));
    }

    init_calls_.emplace_back(InitCall{args, persistent_workspace, iow_addrs});
    return SUCCESS;
  }
  Status UpdateHostArgs(const vector<uint64_t> &active_mem_base_addr, const vector<HostArg> &host_args) override {
    update_host_args_calls_.emplace_back(UpdateHostArgsCall{active_mem_base_addr, host_args});
    return SUCCESS;
  }
  Status UpdateDumpInfos(const std::vector<HostArg> &host_args) override {
    update_dump_infos_calls_.emplace_back(UpdateDumpInfosCall{host_args});
    return SUCCESS;
  }
  Status Distribute() override {
    return SUCCESS;
  }

 public:
  const std::vector<InitCall> &GetInitCalls() const {
    return init_calls_;
  }
  const std::vector<uint64_t> &GetGenInputAddrs() const {
    return gen_input_addrs_;
  }
  const std::vector<uint64_t> &GetGenOutputAddrs() const {
    return gen_output_addrs_;
  }
  const std::vector<uint64_t> &GetGenWsAddrs() const {
    return gen_ws_addrs_;
  }
  int64_t GetGenTaskArgsLen() const {
    return gen_task_args_len_;
  }
  const std::vector<UpdateHostArgsCall> &GetUpdateHostArgsCalls() const {
    return update_host_args_calls_;
  }
  const std::vector<UpdateDumpInfosCall> &GetUpdateDumpInfosCalls() const {
    return update_dump_infos_calls_;
  }
  void Clear() {
    init_calls_.clear();
    update_host_args_calls_.clear();
    update_dump_infos_calls_.clear();
  }

  uint32_t GetTaskID() const override {
    return gen_task_id_;
  }

  uint32_t GetStreamId() const override {
    return gen_stream_id_;
  }
  virtual std::vector<ArgsPlacement> GetArgsPlacements() = 0;
  virtual bool SupportRefresh() const = 0;

 private:
  TaskInfoRegistryStub *registry_;
  ArgsPlacement args_placement_;

  std::vector<uint64_t> gen_input_addrs_;
  std::vector<uint64_t> gen_output_addrs_;
  std::vector<uint64_t> gen_ws_addrs_;
  int64_t gen_task_args_len_;
  uint32_t gen_task_id_{GenTaskId()};
  uint32_t gen_stream_id_{GenTaskId()};

  std::vector<InitCall> init_calls_;
  std::vector<UpdateHostArgsCall> update_host_args_calls_;
  std::vector<UpdateDumpInfosCall> update_dump_infos_calls_;
  std::vector<MemAllocationAndOffset> v_mem_allocation_id_and_offset_;
};
class AicoreStubTaskInfo : public StubTaskInfo {
 public:
  explicit AicoreStubTaskInfo(TaskInfoRegistryStub *registry) : StubTaskInfo(registry) {}
  std::vector<ArgsPlacement> GetArgsPlacements() override {
    return {ArgsPlacement::kArgsPlacementHbm};
  }
  bool SupportRefresh() const override {
    return true;
  }

  Status GetTaskIowPaRemapInfos(std::vector<IowPaRemapInfo> &infos) override {
    for (size_t i = 0U; i < v_mem_allocation_id_and_offset_.size(); i++ ) {
      IowPaRemapInfo iow_pa_remap_info;
      iow_pa_remap_info.allocation_id = v_mem_allocation_id_and_offset_[i].id;
      iow_pa_remap_info.allocation_offset = v_mem_allocation_id_and_offset_[i].offset;
      iow_pa_remap_info.tensor_size = 100;
      iow_pa_remap_info.policy = PaRemapPolicy::KNoSupport;
      infos.emplace_back(std::move(iow_pa_remap_info));
      iow_pa_remap_info.policy = PaRemapPolicy::KSupport;
    }
    return SUCCESS;
  }
};
class RtsStubTaskInfo : public StubTaskInfo {
 public:
  explicit RtsStubTaskInfo(TaskInfoRegistryStub *registry) : StubTaskInfo(registry) {}
  std::vector<ArgsPlacement> GetArgsPlacements() override {
    return {ArgsPlacement::kArgsPlacementTs};
  }
  bool SupportRefresh() const override {
    return true;
  }
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override {
    return task_def.memcpy_async().op_index();
  }
};
class DsaStubTaskInfo : public StubTaskInfo {
 public:
  explicit DsaStubTaskInfo(TaskInfoRegistryStub *registry) : StubTaskInfo(registry) {}
  vector<ArgsPlacement> GetArgsPlacements() override {
    return {ArgsPlacement::kArgsPlacementHbm, ArgsPlacement::kArgsPlacementSqe};
  }
  bool SupportRefresh() const override {
    return false;
  }
};
class RtsLabelSwitchStubTaskInfo : public StubTaskInfo {
 public:
  explicit RtsLabelSwitchStubTaskInfo(TaskInfoRegistryStub *registry) : StubTaskInfo(registry) {}
  std::vector<ArgsPlacement> GetArgsPlacements() override {
    return {};
  }
  bool SupportRefresh() const override {
    return false;
  }
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override {
    return task_def.label_switch_by_index().op_index();
  }
};
class EventStubTaskInfo : public StubTaskInfo {
 public:
  explicit EventStubTaskInfo(TaskInfoRegistryStub *registry) : StubTaskInfo(registry) {}
  std::vector<ArgsPlacement> GetArgsPlacements() override {
    return {};
  }
  bool SupportRefresh() const override {
    return false;
  }
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override {
    return -1;
  }
};
}
#endif  // AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_STUBS_H_
