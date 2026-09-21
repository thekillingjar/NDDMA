/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/rpc/deployer_client.h"
#include "grpc++/grpc++.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "proto/deployer.grpc.pb.h"

namespace ge {
class DeployerClient::GrpcClient {
 public:
  /*
   *  @ingroup ge
   *  @brief   init deployer client
   *  @param   [in]  std::string &
   *  @return  SUCCESS or FAILED
   */
  Status Init(const std::string &address);

  Status SendRequest(const deployer::DeployerRequest &request,
                     deployer::DeployerResponse &response,
                     const int64_t timeout_sec,
                     const int32_t retry_times);

 private:
  std::unique_ptr<deployer::DeployerService::Stub> stub_;
  std::string address_;
  gpr_timespec deadline_{1200, 0, GPR_TIMESPAN};  // 20 min
};

Status DeployerClient::GrpcClient::Init(const std::string &address) {
  GELOGI("Start to create channel, address=%s", address.c_str());
  grpc::ChannelArguments channel_arguments;
  channel_arguments.SetMaxReceiveMessageSize(INT32_MAX);
  channel_arguments.SetMaxSendMessageSize(INT32_MAX);
  auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), channel_arguments);
  if (channel == nullptr) {
    GELOGE(FAILED, "[Create][Channel]Failed to create channel, address = %s", address.c_str());
    return FAILED;
  }

  stub_ = deployer::DeployerService::NewStub(channel);
  address_ = address;
  GE_CHECK_NOTNULL(stub_);
  return SUCCESS;
}

Status DeployerClient::GrpcClient::SendRequest(const deployer::DeployerRequest &request,
                                               deployer::DeployerResponse &response,
                                               const int64_t timeout_sec,
                                               const int32_t retry_times) {
  GELOGI("[Send][Request] start, client_id = %ld, type = %s, peer_address = %s",
         request.client_id(),
         deployer::DeployerRequestType_Name(request.type()).c_str(),
         address_.c_str());
  int32_t try_times = (retry_times <= 0) ? 1 : retry_times + 1;
  constexpr uint32_t kTryWaitInterval = 1000U; // millisconds
  ::grpc::Status status;
  for (int32_t i = 0; i < try_times; ++i) {
    if (i != 0) {
      (void) mmSleep(kTryWaitInterval);  // mmSleep max wait 1s
    }
    ::grpc::ClientContext context;
    if (timeout_sec < 0) {
      context.set_deadline(deadline_);
    } else {
      const gpr_timespec time_out{timeout_sec, 0, GPR_TIMESPAN};
      context.set_deadline(time_out);
    }
    status = stub_->DeployerProcess(&context, request, &response);
    if (status.ok()) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS(
      status.ok(), FAILED, "RPC failed, gRPC error code =%d, type = %s, error message=%s, peer_address = %s",
      status.error_code(), deployer::DeployerRequestType_Name(request.type()).c_str(),
      status.error_message().c_str(), address_.c_str());
  return SUCCESS;
}

DeployerClient::DeployerClient() = default;
DeployerClient::~DeployerClient() = default;

Status DeployerClient::Initialize(const std::string &address) {
  grpc_client_ = MakeUnique<DeployerClient::GrpcClient>();
  GE_CHECK_NOTNULL(grpc_client_);
  GE_CHK_BOOL_RET_STATUS(grpc_client_->Init(address) == SUCCESS,
                         FAILED,
                         "[Init][Client]Init grpc client failed, address:%s",
                         address.c_str());
  return SUCCESS;
}

Status DeployerClient::SendRequest(const deployer::DeployerRequest &request, deployer::DeployerResponse &response,
                                   const int64_t timeout_sec) {
  return grpc_client_->SendRequest(request, response, timeout_sec, 0);
}

Status DeployerClient::SendRequestWithRetry(const deployer::DeployerRequest &request,
                                            deployer::DeployerResponse &response,
                                            const int64_t timeout_sec,
                                            const int32_t retry_times) {
  return grpc_client_->SendRequest(request, response, timeout_sec, retry_times);
}
}  // namespace ge
