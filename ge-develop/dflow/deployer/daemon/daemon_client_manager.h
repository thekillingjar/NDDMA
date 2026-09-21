/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DAEMON_DAEMON_CLIENT_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DAEMON_DAEMON_CLIENT_MANAGER_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include "daemon/deployer_daemon_client.h"
#include "common/config/device_debug_config.h"
#include "ge/ge_api_error_codes.h"
#include "proto/deployer.pb.h"

namespace ge {
class DaemonClientManager {
 public:
  DaemonClientManager() = default;
  virtual ~DaemonClientManager();

  /*
   *  @ingroup ge
   *  @brief   init heartbeat thread
   *  @param   [in]  None
   *  @return  None
   */
  Status Initialize();

  /*
   *  @ingroup ge
   *  @brief   finalize client manager
   *  @param   [in]  None
   *  @return  None
   */
  void Finalize();

  /*
   *  @ingroup ge
   *  @brief   create and init client
   *  @param   [in]  const std::string &
   *  @param   [in]  const std::map<std::string, std::string> &
   *  @param   [out]  int64_t &
   *  @return  SUCCESS or FAILED
   */
  Status CreateAndInitClient(const std::string &peer_uri, const std::map<std::string, std::string> &deployer_envs,
                             int64_t &client_id);

  /*
   *  @ingroup ge
   *  @brief   close client
   *  @param   [in]  int64_t
   *  @return  SUCCESS or FAILED
   */
  Status CloseClient(const int64_t client_id);

  /*
 *  @ingroup ge
 *  @brief   create client
 *  @param   [in]  int64_t
 *  @return  client ptr
 */
  virtual std::unique_ptr<DeployerDaemonClient> CreateClient(int64_t client_id) {
    return MakeUnique<DeployerDaemonClient>(client_id);
  }

  /*
   *  @ingroup ge
   *  @brief   get client by client_id
   *  @param   [in]  uint32_t
   *  @return  client ptr
   */
  DeployerDaemonClient *GetClient(const int64_t client_id);

  Status RecordClientInfo(const int64_t client_id, const std::string &peer_uri);

  Status DeleteClientInfo(const int64_t client_id);

  void GenDgwPortOffset(const int32_t dev_count, int32_t &offset);

 private:
  struct ClientAddr {
    std::string ip;
    std::string port;
  };

  static Status GetClientIpAndPort(const std::string &uri, ClientAddr &client);
  Status UpdateJsonFile();
  void DeleteAllClientInfo();
  void EvictExpiredClients();

  std::map<int64_t, std::unique_ptr<DeployerDaemonClient>> clients_;
  std::atomic_bool running_{};
  std::mutex mu_;
  std::mutex mu_cv_;
  std::condition_variable running_cv_;
  std::thread evict_thread_;
  int64_t client_id_gen_ = 0;
  std::map<int64_t, ClientAddr> client_addrs_;
  int32_t dgw_port_offset_gen_ = 0;
  int32_t client_fd_ = -1;
};
} // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DAEMON_DAEMON_CLIENT_MANAGER_H_
