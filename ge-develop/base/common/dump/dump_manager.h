/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_DUMP_MANAGER_H_
#define GE_COMMON_DUMP_DUMP_MANAGER_H_

#include <mutex>

#include "common/dump/dump_properties.h"
#include "framework/common/ge_types.h"

namespace ge {
class DumpManager {
 public:
  static DumpManager &GetInstance();
  Status Init(const std::map<std::string, std::string> &run_options);
  Status Finalize();

  Status SetDumpConf(const DumpConfig &dump_config);
  DumpProperties GetDumpProperties(const uint64_t session_id);
  const std::map<uint64_t, DumpProperties> &GetDumpPropertiesMap() const {
    return dump_properties_map_;
  }
  Status AddDumpProperties(uint64_t session_id, const DumpProperties &dump_properties);
  void RemoveDumpProperties(uint64_t session_id);
  static bool GetCfgFromOption(const std::map<std::string, std::string> &options_all, DumpConfig &dump_cfg);
  void SetDumpWorkerId(const uint64_t session_id, const std::string &worker_id);
  void SetDumpOpSwitch(const uint64_t session_id, const std::string &op_switch);
  bool CheckDumpFlag() const {
    return dump_flag_;
  }
  Status RegisterCallBackFunc(const std::string &func, const std::function<Status(const DumpProperties &)> &callback);
  bool CheckIfAclDumpOpen() const {
    return if_acl_dump_set_;
  }
  bool CheckIfAclDumpSet();
  void ClearAclDumpSet();

  bool IsDumpExceptionOpen() const {
    return l1_exception_dump_flag_;
  }

 private:
  bool CheckHasNpuCollectPath() const;
  bool NeedDoDump(const DumpConfig &dump_config, DumpProperties &dump_properties);
  void SetDumpDebugConf(const DumpConfig &dump_config, DumpProperties &dump_properties);
  bool SetExceptionDump(const DumpConfig &dump_config);
  Status SetDumpPath(const DumpConfig &dump_config, DumpProperties &dump_properties) const;
  Status SetNormalDumpConf(const DumpConfig &dump_config, DumpProperties &dump_properties);
  void SetDumpList(const DumpConfig &dump_config, DumpProperties &dump_properties) const;
  void SetModelDumpBlacklist(ModelOpBlacklist& model_blacklist,
    const std::vector<DumpBlacklist>& type_blacklists, const std::vector<DumpBlacklist>& name_blacklists) const;
  uint32_t ExtractNumber(const std::string& s) const;
  bool IsValidFormat(const std::string& s) const;
  void ExtractBlacklist(const std::vector<DumpBlacklist>& blacklists,
    std::map<std::string, OpBlacklist>& op_blacklist) const;
  Status ReloadDumpInfo(const DumpProperties &dump_properties);
  Status UnloadDumpInfo(const DumpProperties &dump_properties);
  std::mutex mutex_;
  bool dump_flag_ = false;
  // enable dump from acl first time
  bool if_acl_dump_set_ = false;
  std::map<uint64_t, DumpProperties> dump_properties_map_;
  // enable dump from acl with session_id 0
  std::map<uint64_t, DumpProperties> infer_dump_properties_map_;
  std::map<std::string, std::function<Status(const DumpProperties &)>> callback_map_;

  std::string npu_collect_path_;
  std::string ascend_work_path_;

  bool l1_exception_dump_flag_ = false;
  bool l0_exception_dump_flag_ = false;
};
}  // namespace ge
#endif  // GE_COMMON_DUMP_DUMP_MANAGER_H_
