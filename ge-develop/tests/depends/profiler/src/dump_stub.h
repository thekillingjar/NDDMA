/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DUMP_STUB_H_
#define DUMP_STUB_H_
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "adump_api.h"
#include "adump_pub.h"
#include "exe_graph/runtime/tensor.h"

namespace ge {

std::string AdxGetAdditionalInfo(const Adx::OperatorInfoV2 &info, const std::string key);
std::string AdxGetTilingKey(const Adx::OperatorInfoV2 &info);
bool AdxGetArgsInfo(const Adx::OperatorInfoV2 &info, void *&addr, uint64_t &length);
bool AdxGetWorkspaceInfo(const Adx::OperatorInfoV2 &info, uint32_t index, void *&addr, uint64_t &length);

class DumpStub {
 public:
  static DumpStub &GetInstance();
  void *Get(uint32_t space, uint32_t &atomic_index) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint64_t> unit(100, 0UL);
    for (uint32_t i = 0; i < space; ++i) {
      units_.emplace_back(unit);
    }
    atomic_index = units_.size();
    return units_[units_.size() - 1].data();
  }

  void *GetByIndex(const uint64_t atomic_index) {
    return units_[atomic_index].data();
  }

  void *GetDynamic(uint32_t space, uint64_t &atomic_index) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint64_t> unit(space, 0UL);
    units_dynamic_.emplace_back(unit);
    atomic_index = units_dynamic_.size();
    return units_dynamic_[units_dynamic_.size() - 1].data();
  }

  void *GetDynamicByIndex(const uint64_t atomic_index) {
    return units_dynamic_[atomic_index].data();
  }

  void *GetStatic(uint32_t space, uint64_t &atomic_index) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint64_t> unit(space, 0UL);
    units_static_.emplace_back(unit);
    atomic_index = units_static_.size();
    return units_static_[units_static_.size() - 1].data();
  }

  void *GetStaicByIndex(const uint64_t atomic_index) {
    return units_static_[atomic_index].data();
  }

  void Clear() {
    units_.clear();
    units_dynamic_.clear();
    units_static_.clear();
  }

  std::vector<std::vector<uint64_t>> &GetUnits() {
    std::lock_guard<std::mutex> lk(mu_);
    return units_;
  }

  std::vector<std::vector<uint64_t>> &GetDynamicUnits() {
    std::lock_guard<std::mutex> lk(mu_);
    return units_dynamic_;
  }

  std::vector<std::vector<uint64_t>> &GetStaticUnits() {
    std::lock_guard<std::mutex> lk(mu_);
    return units_static_;
  }

  void AddOpInfo(const Adx::OperatorInfoV2 &op_info);

  const std::vector<Adx::OperatorInfoV2> &GetOpInfos() {
    std::lock_guard<std::mutex> lk(mu_);
    return dump_op_infos_;
  }

  bool GetOpInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id, Adx::OperatorInfoV2 &info) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto &op_info : dump_op_infos_) {
      if ((op_info.deviceId == device_id) && (op_info.streamId == stream_id) && (op_info.taskId == task_id)) {
        info = op_info;
        return true;
      }
    }
    return false;
  }

  bool DelOpInfo(uint32_t device_id, uint32_t stream_id) {
    bool is_found = false;
    for (auto iter = dump_op_infos_.begin(); iter != dump_op_infos_.end();) {
      if ((iter->deviceId == device_id) && (iter->streamId == stream_id)) {
        iter = dump_op_infos_.erase(iter);
        is_found = true;
      } else {
        ++iter;
      }
    }
    return is_found;
  }

  void ClearOpInfos() {
    dump_op_infos_.clear();
    dump_op_tensors_.clear();
  }

  void SetFuncRet(const std::string &fun_name, int32_t ret) {
    stub_func_ret_[fun_name] = ret;
  }

  void ClearFuncRet() {
    stub_func_ret_.clear();
  }

  int GetFuncRet(const std::string &fun_name, int default_ret = 0) {
    auto iter = stub_func_ret_.find(fun_name);
    if (iter != stub_func_ret_.end()) {
      return iter->second;
    }
    return default_ret;
  }

  int32_t SetDumpConfig(Adx::DumpType dump_type, const Adx::DumpConfig &dump_config);
  bool IsDumpEnable(Adx::DumpType dump_type) {
    std::lock_guard<std::mutex> lk(mu_);
    return (dump_configs_.find(dump_type) != dump_configs_.end());
  }

  bool GetDumpPath(Adx::DumpType dump_type, std::string &dump_path) {
    std::lock_guard<std::mutex> lk(mu_);
    if (dump_configs_.find(dump_type) != dump_configs_.end()) {
      dump_path = dump_configs_[dump_type].dumpPath;
      return true;
    }
    return false;
  }

  void Reset() {
    Clear();
    ClearOpInfos();
    ClearFuncRet();
    dump_configs_.clear();
  }

  void SetEnableFlag(bool is_enable) {
    is_enable_ = is_enable;
  }

  bool GetEnableFlag() {
    return is_enable_;
  }

  uint64_t AdumpGetDumpSwitch(const Adx::DumpType dumpType);

  void RecordCall(const std::string& func_name, uint32_t module_id) {
    std::lock_guard<std::mutex> lk(mu_);
    call_records_[func_name] = module_id;
  }

  uint32_t GetCallRecord(const std::string& func_name) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = call_records_.find(func_name);
    return it != call_records_.end() ? it->second : 0;
  }

 private:
  DumpStub() = default;
  std::mutex mu_;
  std::map<std::string, uint32_t> call_records_;
  std::vector<std::vector<uint64_t>> units_;
  std::vector<std::vector<uint64_t>> units_dynamic_;
  std::vector<std::vector<uint64_t>> units_static_;
  std::vector<Adx::OperatorInfoV2> dump_op_infos_;            // for exception dump
  std::vector<std::vector<gert::Tensor>> dump_op_tensors_;  // for exception dump
  std::map<std::string, int32_t> stub_func_ret_;
  std::map<Adx::DumpType, Adx::DumpConfig> dump_configs_;
  bool is_enable_ {true};
};
}  // namespace ge

#endif  // DUMP_STUB_H_
