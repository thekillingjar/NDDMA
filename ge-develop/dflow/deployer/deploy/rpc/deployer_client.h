/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RPC_DEPLOY_SERVICE_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RPC_DEPLOY_SERVICE_H_

#include <string>
#include <memory>
#include "common/plugin/ge_make_unique_util.h"
#include "proto/deployer.pb.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class DeployerClient {
 public:
  explicit DeployerClient();
  virtual ~DeployerClient();
  GE_DELETE_ASSIGN_AND_COPY(DeployerClient);

  Status Initialize(const std::string &address);

  virtual Status SendRequest(const deployer::DeployerRequest &request,
                             deployer::DeployerResponse &response, const int64_t timeout_sec = -1);

  virtual Status SendRequestWithRetry(const deployer::DeployerRequest &request,
                                      deployer::DeployerResponse &response,
                                      const int64_t timeout_sec,
                                      const int32_t retry_times);

 private:
  class GrpcClient;
  std::unique_ptr<GrpcClient> grpc_client_;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RPC_DEPLOY_SERVICE_H_
