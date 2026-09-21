/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "daemon/grpc_server.h"
#include <condition_variable>
#include <mutex>
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/log.h"
#include "daemon/deployer_service_impl.h"

namespace ge {
class GrpcServer::Impl {
 public:
  bool running_ = true;
  std::mutex mu_;
  std::condition_variable cv_;
};

GrpcServer::GrpcServer() = default;
GrpcServer::~GrpcServer() = default;

Status GrpcServer::Run() {
  impl_ = MakeUnique<GrpcServer::Impl>();
  GE_CHECK_NOTNULL(impl_);
  std::unique_lock<std::mutex> lk(impl_->mu_);
  impl_->cv_.wait(lk, [&] () { return !impl_->running_; });
  return SUCCESS;
}

void GrpcServer::Finalize() {
  std::unique_lock<std::mutex> lk(impl_->mu_);
  impl_->running_ = false;
  impl_->cv_.notify_all();
}
} // namespace ge
