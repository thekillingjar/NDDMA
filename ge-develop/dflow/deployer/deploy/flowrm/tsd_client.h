/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_DEPLOYER_TSD_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_DEPLOYER_TSD_CLIENT_H_

#include <mutex>
#include <map>
#include <set>
#include "proto/deployer.pb.h"
#include "ge/ge_api_error_codes.h"
#include "aicpu/tsd/tsd_client.h"
#include "common/subprocess/subprocess_manager.h"

namespace ge {
class TsdClient {
 public:
  using TsdCapabilityGet = uint32_t(*)(const uint32_t device_id, const int32_t type, const uint64_t ptr);

  GE_DELETE_ASSIGN_AND_COPY(TsdClient);
  static TsdClient& GetInstance();
  Status SetDlogReportStart(int32_t device_id);
  void SetDlogReportStop();
  Status Initialize();
  Status GetProcStatus(int32_t device_id, pid_t pid, ProcStatus &proc_status, const std::string &proc_type);
  Status GetProcStatus(int32_t device_id, const std::vector<pid_t> &pids, std::map<pid_t, ProcStatus> &proc_status,
                       const std::string &proc_type);
  Status ForkSubprocess(int32_t device_id,
                        const SubprocessManager::SubprocessConfig &subprocess_config,
                        const std::string &file_path,
                        pid_t &pid);
  Status ForkSubprocess(int32_t device_id,
                        const SubprocessManager::SubprocessConfig &subprocess_config,
                        pid_t &pid);
  Status ShutdownSubprocess(int32_t device_id, pid_t pid, const std::string &proc_type);
  Status LoadFile(int32_t device_id, const std::string &file_path, const std::string &file_name);
  Status UnloadFile(int32_t device_id, const std::string &file_path);
  Status StartFlowGw(int32_t device_id, const std::string &group_name, pid_t &pid);

  /**
   * @brief check tsd whether support unpack inner tar pack.
   * @param device_id tsd device id.
   * @param is_support true: support, false: not support
   * @return SUCCESS: check success, other:check failed.
   */
  Status CheckSupportInnerPackUnpack(int32_t device_id, bool &is_support);

  /**
   * @brief check tsd whether support launch builtin udf.
   * @param device_id tsd device id.
   * @param is_support true: support, false: not support
   * @return SUCCESS: check success, other:check failed.
   */
  Status CheckSupportBuiltinUdfLaunch(int32_t device_id, bool &is_support);

  void Finalize();
  static uint64_t GetLoadFileCount();

 private:
  TsdClient() = default;
  ~TsdClient() = default;

  Status LoadTsdClientLib();
  std::mutex &GetDeviceMutex(int32_t device_id);
  Status SetDevice(int32_t device_id);
  Status LoadPackages(int32_t device_id) const;
  static SubProcType TransferProcType(const std::string &proc_type);

  Status LoadFileByTsd(int32_t device_id, const char_t *const file_path, const size_t path_len,
                       const std::string &file_name) const;
  Status CheckCapabilitySupport(int32_t device_id, int32_t capability, bool &is_support, uint64_t required = 1UL);
  void *handle_ = nullptr;
  TsdCapabilityGet tsd_capability_get_ = nullptr;
  std::mutex map_mutex_;
  std::map<int32_t, std::mutex> device_mutexs_;
  std::set<int32_t> set_log_save_mode_;
  std::set<int32_t> set_device_list_;
  std::mutex init_mutex_;
  bool inited_ = false;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_FLOWRM_DEPLOYER_TSD_CLIENT_H_
