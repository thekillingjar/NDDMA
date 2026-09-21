/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "daemon/deployer_daemon_client.h"
#include <signal.h>
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/utils/rts_api_utils.h"

namespace ge {
namespace {
constexpr int64_t kHeartbeatExpireSec = 30;   // s
constexpr int64_t kHeartbeatTimeoutSec = 15;  // s
constexpr uint32_t kMsgQueueDepth = 3U;
const std::string kSubDeployer = "deployer_daemon";
constexpr uint32_t kMaxDeployerShutdownWaitTimeInSec = 50;  // s
}  // namespace

DeployerDaemonClient::~DeployerDaemonClient() {
  (void)Finalize();
}

DeployerDaemonClient::DeployerDaemonClient(int64_t client_id)
    : client_id_(client_id), sub_deployer_proc_stat_(ProcStatus::NORMAL) {}

Status DeployerDaemonClient::Initialize(const std::map<std::string, std::string> &deployer_envs) {
  deployer_envs_ = deployer_envs;
  GE_CHK_STATUS_RET(ForkAndInit(), "Failed to init sub deployer process, client_id:%ld", client_id_);
  OnHeartbeat();
  GEEVENT("[Initialize] deployer daemon client success, client id:%ld, sub deployer pid:%d", client_id_,
          static_cast<int32_t>(pid_));
  return SUCCESS;
}

Status DeployerDaemonClient::Finalize() {
  if (pid_ == -1) {
    return SUCCESS;
  }
  Shutdown();
  if (deployer_msg_client_ != nullptr) {
    (void)deployer_msg_client_->Finalize();
  }
  GEEVENT("[Finalize] deployer daemon client success, client id:%ld, sub deployer pid:%d", client_id_,
          static_cast<int32_t>(pid_));
  pid_ = -1;
  return SUCCESS;
}

void DeployerDaemonClient::Shutdown() const {
  if (deployer_msg_client_ != nullptr) {
    deployer::DeployerRequest request;
    request.set_type(deployer::kDisconnect);
    (void)deployer_msg_client_->SendRequestWithoutResponse(request);
  }
  (void)SubprocessManager::GetInstance().ShutdownSubprocess(pid_, kMaxDeployerShutdownWaitTimeInSec);
  GELOGI("Success to shutdown sub deployer process, pid:%d", static_cast<int32_t>(pid_));
}

bool DeployerDaemonClient::IsExpired() {
  std::lock_guard<std::mutex> lk(mu_);
  int64_t diff =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - last_heartbeat_ts_).count();
  return diff > kHeartbeatExpireSec;
}

bool DeployerDaemonClient::IsExecuting() {
  std::lock_guard<std::mutex> lk(mu_);
  return is_executing_;
}

void DeployerDaemonClient::SetIsExecuting(bool is_executing) {
  std::lock_guard<std::mutex> lk(mu_);
  is_executing_ = is_executing;
}

void DeployerDaemonClient::OnHeartbeat() {
  std::lock_guard<std::mutex> lk(mu_);
  last_heartbeat_ts_ = std::chrono::steady_clock::now();
}

Status DeployerDaemonClient::ForkSubDeployerProcess(const std::string &group_name) {
  SubprocessManager::SubprocessConfig config{};
  config.process_type = kSubDeployer;
  config.death_signal = SIGKILL;
  const std::string process_name = "sub_deployer_" + std::to_string(client_id_);
  config.args = {process_name, group_name, std::to_string(req_msg_queue_id_), std::to_string(rsp_msg_queue_id_)};
  for (const auto &env : deployer_envs_) {
    config.envs.emplace(env.first, env.second);
  }
  GE_CHK_STATUS_RET(SubprocessManager::GetInstance().ForkSubprocess(config, pid_), "Failed to fork %s",
                    process_name.c_str());
  // watch sub deployer daemon
  std::function<void(const ProcStatus &)> excpt_handle_callback = [this](const ProcStatus &proc_status) {
    GEEVENT("Sub deployer process status is %d.", static_cast<int32_t>(proc_status));
    sub_deployer_proc_stat_ = proc_status;
  };
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid_, excpt_handle_callback);
  GELOGI("Fork sub deployer success, process name:%s, pid:%d", process_name.c_str(), static_cast<int32_t>(pid_));
  return SUCCESS;
}

Status DeployerDaemonClient::ForkAndInit() {
  // 1. create message queue
  deployer_msg_client_ = CreateMessageClient();
  GE_CHECK_NOTNULL(deployer_msg_client_);
  const std::string queue_name_suffix = "sub_deployer_" + std::to_string(client_id_);
  GE_CHK_STATUS_RET(deployer_msg_client_->CreateMessageQueue(queue_name_suffix, req_msg_queue_id_, rsp_msg_queue_id_),
                    "Failed to create message queue for sub deployer");

  // 2. start sub deployer process
  const auto &group_name = MemoryGroupManager::GetInstance().GetQsMemGroupName();
  GE_CHK_STATUS_RET(ForkSubDeployerProcess(group_name), "Failed to fork sub deployer process");
  GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().MemGrpAddProc(group_name, pid_, true, true),
                    "Failed to add group, pid:%d", static_cast<int32_t>(pid_));

  // 3. initialize message client
  const auto get_stat_func = [this]() -> Status {
    return (sub_deployer_proc_stat_.load() == ProcStatus::EXITED) ? FAILED : SUCCESS;
  };
  GE_CHK_STATUS_RET(deployer_msg_client_->Initialize(pid_, get_stat_func), "Failed to initialize message client");
  return SUCCESS;
}

Status DeployerDaemonClient::ProcessHeartbeatRequest(const deployer::DeployerRequest &request,
                                                     deployer::DeployerResponse &response) {
  GE_CHECK_NOTNULL(deployer_msg_client_);
  if (sub_deployer_proc_stat_ == ProcStatus::NORMAL) {
    GELOGI("[Process][Request] client heartbeat dose not expired, client_id = %ld.", client_id_);
    GE_CHK_STATUS_RET(deployer_msg_client_->SendRequest(request, response, kHeartbeatTimeoutSec));
  } else if (sub_deployer_proc_stat_ == ProcStatus::EXITED) {
    response.set_error_code(FAILED);
    response.set_error_message("Sub deployer process does not exist");
    GELOGE(FAILED, "Sub deployer process does not exist, client id:%ld", client_id_);
    return FAILED;
  } else if (sub_deployer_proc_stat_ == ProcStatus::STOPPED) {
    response.set_error_code(static_cast<int32_t>(ProcStatus::STOPPED));
    response.set_error_message("Sub deployer process stopped");
    GELOGE(FAILED, "Sub deployer process stopped, client id:%ld", client_id_);
    return FAILED;
  } else {
    // do nothing
  }
  return SUCCESS;
}

Status DeployerDaemonClient::ProcessDeployRequest(const deployer::DeployerRequest &request,
                                                  deployer::DeployerResponse &response) {
  GELOGD("Begin to process deploy request, type:%s", deployer::DeployerRequestType_Name(request.type()).c_str());
  GE_CHECK_NOTNULL(deployer_msg_client_);
  GE_CHK_STATUS_RET_NOLOG(deployer_msg_client_->SendRequest(request, response, -1));
  GELOGD("Success to process deploy request, type:%s", deployer::DeployerRequestType_Name(request.type()).c_str());
  return SUCCESS;
}
}  // namespace ge
