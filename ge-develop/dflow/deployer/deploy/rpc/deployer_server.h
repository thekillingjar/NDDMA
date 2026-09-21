/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RPC_DEPLOYER_SERVER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RPC_DEPLOYER_SERVER_H_

#include <string>
#include <memory>
#include "ge/ge_api_error_codes.h"
#include "proto/deployer.pb.h"
#include "common/plugin/ge_make_unique_util.h"

namespace ge {
class DeployerSpi {
 public:
  DeployerSpi() = default;
  virtual ~DeployerSpi() = default;
  GE_DELETE_ASSIGN_AND_COPY(DeployerSpi);

  virtual Status Initialize() = 0;
  virtual void Finalize() = 0;
  virtual Status Process(const std::string &peer_uri,
                         const deployer::DeployerRequest &request,
                         deployer::DeployerResponse &response) = 0;
};

class DeployerServer {
 public:
  DeployerServer();
  ~DeployerServer();

  /*
   *  @ingroup ge
   *  @brief   run grpc server
   *  @return  SUCCESS or FAILED
   */
  Status Run();

  /*
   *  @ingroup ge
   *  @brief   finalize grpc server
   *  @return  SUCCESS or FAILED
   */
  void Finalize();

  void SetServiceProvider(std::unique_ptr<DeployerSpi> service_provider);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<DeployerSpi> service_provider_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RPC_DEPLOYER_SERVER_H_
