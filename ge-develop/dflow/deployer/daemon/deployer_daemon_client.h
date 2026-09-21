/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DAEMON_DEPLOYER_DAEMON_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DAEMON_DEPLOYER_DAEMON_CLIENT_H_

#include <cstdint>
#include <map>
#include <mutex>
#include <functional>
#include <thread>
#include <atomic>
#include "common/config/device_debug_config.h"
#include "common/message_handle/message_client.h"
#include "ge/ge_api_error_codes.h"
#include "deploy/deployer/deploy_context.h"

namespace ge {
class DeployerDaemonClient {
 public:
  explicit DeployerDaemonClient(int64_t client_id);
  virtual ~DeployerDaemonClient();

  /*
   *  @ingroup ge
   *  @brief   record client
   *  @param   [in]  const std::map<std::string, std::string> &
   *  @return  None
   */
  Status Initialize(const std::map<std::string, std::string> &deployer_envs);

  /*
   *  @ingroup ge
   *  @brief   finalize client
   *  @param   [in]  None
   *  @return  None
   */
  Status Finalize();

  /*
   *  @ingroup ge
   *  @brief   check client is expired
   *  @param   [in]  None
   *  @return  Expired or not
   */
  bool IsExpired();

  /*
   *  @ingroup ge
   *  @brief   set client is executing
   *  @param   [in]  bool
   *  @return  None
   */
  void SetIsExecuting(bool is_executing);

  /*
   *  @ingroup ge
   *  @brief   check client is executing
   *  @param   [in]  None
   *  @return  None
   */
  bool IsExecuting();

  /*
   *  @ingroup ge
   *  @brief   record client last heartbeat time
   *  @param   [in]  None
   *  @return  None
   */
  void OnHeartbeat();

  virtual Status ProcessHeartbeatRequest(const deployer::DeployerRequest &request,
                                         deployer::DeployerResponse &response);

  virtual Status ProcessDeployRequest(const deployer::DeployerRequest &request, deployer::DeployerResponse &response);

 protected:
  virtual std::shared_ptr<DeployerMessageClient> CreateMessageClient() {
    return MakeShared<DeployerMessageClient>(0, true);
  }

 private:
  Status ForkSubDeployerProcess(const std::string &group_name);
  Status ForkAndInit();
  void Shutdown() const;

  pid_t pid_ = -1;
  int64_t client_id_ = -1;
  std::map<std::string, std::string> deployer_envs_;
  uint32_t req_msg_queue_id_ = UINT32_MAX;
  uint32_t rsp_msg_queue_id_ = UINT32_MAX;
  std::shared_ptr<DeployerMessageClient> deployer_msg_client_;
  std::atomic<ProcStatus> sub_deployer_proc_stat_;
  std::mutex mu_;
  std::chrono::steady_clock::time_point last_heartbeat_ts_;
  bool is_executing_ = false;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DAEMON_DEPLOYER_DAEMON_CLIENT_H_
