/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/rpc/deployer_server.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/log.h"
#include "common/config/configurations.h"
#include "grpc++/grpc++.h"
#include "proto/deployer.grpc.pb.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
class DeployerGrpcService final : public deployer::DeployerService::Service {
 public:
  explicit DeployerGrpcService(DeployerSpi *service_provider) : service_provider_(service_provider) {}
  ~DeployerGrpcService() final = default;

  ::grpc::Status DeployerProcess(::grpc::ServerContext *context, const deployer::DeployerRequest *request,
                                 deployer::DeployerResponse *response) override {
    if (response == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Response param is null.");
      GELOGE(FAILED, "[Check][Response] Response param is null.");
      return ::grpc::Status::CANCELLED;
    }

    if ((context == nullptr) || (request == nullptr)) {
      REPORT_INNER_ERR_MSG("E19999", "Input params is null.");
      GELOGE(FAILED, "[Check][Params] Input params is null.");
      response->set_error_code(FAILED);
      response->set_error_message("Input params is null");
      return ::grpc::Status::OK;
    }

    (void) service_provider_->Process(context->peer(), *request, *response);
    return ::grpc::Status::OK;
  }

 private:
  DeployerSpi *service_provider_;
};
}  // namespace

class DeployerServer::Impl {
 public:
  explicit Impl(DeployerSpi *service_provider) : service_(service_provider) {}
  ~Impl() = default;

  Status Run() {
    const auto &device_info = Configurations::GetInstance().GetLocalNode();
    std::string port = std::to_string(device_info.port);
    std::string ip = device_info.ipaddr;
    std::string server_addr = ip + ":" + port;
    grpc::ServerBuilder server_builder;
    server_builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    server_builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
    server_builder.RegisterService(&service_);
    server_builder.SetMaxReceiveMessageSize(INT32_MAX);
    server_builder.SetMaxSendMessageSize(INT32_MAX);
    grpc_server_ = server_builder.BuildAndStart();
    if (grpc_server_ == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Failed to build and start gRPC service, address = %s is invalid or reused",
                         server_addr.c_str());
      GELOGE(FAILED, "[Build][Server] Failed to build and start gRPC service, address = %s is invalid or reused",
             server_addr.c_str());
      return FAILED;
    }
    GEEVENT("Server listening on %s.", server_addr.c_str());
    grpc_server_->Wait();
    return SUCCESS;
  }

  void Finalize() {
    if (grpc_server_ != nullptr) {
      grpc_server_->Shutdown();
    }
  }

 private:
  DeployerGrpcService service_;
  std::unique_ptr<grpc::Server> grpc_server_;
};

DeployerServer::DeployerServer() = default;
DeployerServer::~DeployerServer() = default;

Status DeployerServer::Run() {
  GE_CHECK_NOTNULL(service_provider_);
  GE_CHK_STATUS_RET(service_provider_->Initialize(), "Failed to initialize service provider");
  impl_ = MakeUnique<DeployerServer::Impl>(service_provider_.get());
  GE_CHECK_NOTNULL(impl_);
  return impl_->Run();
}

void DeployerServer::Finalize() {
  if (impl_ != nullptr) {
    impl_->Finalize();
    impl_.reset();
  }
  if (service_provider_ != nullptr) {
    service_provider_->Finalize();
  }
}

void DeployerServer::SetServiceProvider(std::unique_ptr<DeployerSpi> service_provider) {
  service_provider_ = std::move(service_provider);
}
}  // namespace ge
