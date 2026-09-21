/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "deploy/rpc/deployer_client.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "daemon/daemon_service.h"
#include "dflow/base/exec_runtime/execution_runtime.h"

namespace ge {
class DeployerClient::GrpcClient {
 public:
  DeployContext context_;
  DaemonService daemon_service_;
};

DeployerClient::DeployerClient() = default;
DeployerClient::~DeployerClient() = default;

Status DeployerClient::Initialize(const string &address) {
  grpc_client_ = MakeUnique<DeployerClient::GrpcClient>();
  GE_CHECK_NOTNULL(grpc_client_);
  return SUCCESS;
}

Status DeployerClient::SendRequest(const deployer::DeployerRequest &request, deployer::DeployerResponse &response,
                                   const int64_t timeout_sec) {
  auto original_handle = ExecutionRuntime::handle_;
  ExecutionRuntime::handle_ = nullptr;
  std::string peer_uri = "http://127.0.0.1:1234";
  auto ret = grpc_client_->daemon_service_.Process(peer_uri, request, response);
  ExecutionRuntime::handle_ = original_handle;
  return ret;
}
}  // namespace ge
