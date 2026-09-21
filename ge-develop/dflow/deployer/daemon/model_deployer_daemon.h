/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DAEMON_MODEL_DEPLOYER_DAEMON_H_
#define AIR_RUNTIME_HETEROGENEOUS_DAEMON_MODEL_DEPLOYER_DAEMON_H_

#include <thread>
#include "daemon/daemon_service.h"
#include "common/message_handle/message_server.h"
#include "framework/executor/ge_executor.h"

namespace ge {
class ModelDeployerDaemon {
 public:
  explicit ModelDeployerDaemon(bool is_sub_deployer = false);
  ~ModelDeployerDaemon();

  Status Start(int32_t argc, char_t **argv);

  static void SignalHandler(int32_t sig_num);

 private:
  template<typename T>
  Status ToNumber(const char_t *num_str, T &value) const {
    GE_CHECK_NOTNULL(num_str);
    std::stringstream ss(num_str);
    ss >> value;
    GE_CHK_BOOL_RET_STATUS(!ss.fail(), PARAM_INVALID, "Failed to convert [%s] to number", num_str);
    GE_CHK_BOOL_RET_STATUS(ss.eof(), PARAM_INVALID, "Failed to convert [%s] to number", num_str);
    return ge::SUCCESS;
  }

  static Status RunGrpcService();

  Status StartDeployerDaemon();

  Status StartSubDeployer(int32_t argc, char_t **argv);

  void GrpcWaitForShutdown() const;

  Status ParseCmdLineArgs(int32_t argc, char_t **argv);

  void InitDeployContext();

  Status NotifyInitialized();

  Status WaitDeployRequest();

  void ProcessDeployRequest(const deployer::DeployerRequest &request);

  Status Finalize();

  std::thread shutdown_thread_;
  static ge::DeployerServer grpc_server_;
  GeExecutor ge_executor_;
  std::shared_ptr<MessageServer> deployer_message_server_;
  uint32_t req_msg_queue_id_ = UINT32_MAX;
  uint32_t rsp_msg_queue_id_ = UINT32_MAX;
  bool is_sub_deployer_ = false;
  std::string parent_group_name_;
  ThreadPool pool_{"ge_dpl_dam", 16U, false};
  DeployContext context_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DAEMON_MODEL_DEPLOYER_DAEMON_H_
