/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_engine.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_log.h"
#include "util/util.h"

namespace dvpp {
DvppEngine& DvppEngine::Instance() {
  static DvppEngine instance;
  return instance;
}

ge::Status DvppEngine::Initialize(
    const std::map<std::string, std::string>& options) {
  DVPP_ENGINE_LOG_DEBUG("dvpp engine Initialize start");

  // 检查当前版本，不支持的版本不进行初始化
  auto iter = options.find(ge::SOC_VERSION);
  DVPP_CHECK_IF_THEN_DO(iter == options.end(),
      DVPP_REPORT_INNER_ERR_MSG("can not find [%s]", ge::SOC_VERSION.c_str());
      return ge::FAILED);

  DVPP_CHECK_IF_THEN_DO(!IsSocVersionSupported(iter->second),
      DVPP_ENGINE_LOG_EVENT("dvpp engine is not supported");
      return ge::SUCCESS);

  try {
    if (dvpp_graph_optimizer_ == nullptr) {
      dvpp_graph_optimizer_ = std::make_shared<DvppGraphOptimizer>();
    }

    if (dvpp_ops_kernel_builder_ == nullptr) {
      dvpp_ops_kernel_builder_ = std::make_shared<DvppOpsKernelBuilder>();
    }

    if (dvpp_ops_kernel_info_store_ == nullptr) {
      dvpp_ops_kernel_info_store_ = std::make_shared<DvppOpsKernelInfoStore>();
    }
  } catch (...) {
    DVPP_REPORT_INNER_ERR_MSG("dvpp engine Initialize failed");
    return ge::FAILED;
  }

  DVPP_ENGINE_LOG_DEBUG("dvpp engine Initialize success");
  return ge::SUCCESS;
}

void DvppEngine::GetGraphOptimizerObjs(
    std::map<std::string, GraphOptimizerPtr>& graph_optimizers) {
  DVPP_ENGINE_LOG_DEBUG("dvpp engine GetGraphOptimizerObjs start");
  DVPP_CHECK_IF_THEN_DO(dvpp_graph_optimizer_ == nullptr,
      DVPP_ENGINE_LOG_EVENT("dvpp graph optimizer is not supported");
      return);

  graph_optimizers[kDvppGraphOptimizer] = dvpp_graph_optimizer_;
  DVPP_ENGINE_LOG_DEBUG("dvpp engine GetGraphOptimizerObjs success");
}

void DvppEngine::GetOpsKernelBuilderObjs(
    std::map<std::string, OpsKernelBuilderPtr>& ops_kernel_builders) {
  // GE实际未使用该接口 而是dvpp使用REGISTER_OPS_KERNEL_BUILDER
  DVPP_ENGINE_LOG_DEBUG("dvpp engine GetOpsKernelBuilderObjs start");
  DVPP_CHECK_IF_THEN_DO(dvpp_ops_kernel_builder_ == nullptr,
      DVPP_ENGINE_LOG_EVENT("dvpp ops kernel builder is not supported");
      return);

  ops_kernel_builders[kDvppOpsKernel] = dvpp_ops_kernel_builder_;
  DVPP_ENGINE_LOG_DEBUG("dvpp engine GetOpsKernelBuilderObjs success");
}

void DvppEngine::GetOpsKernelInfoStores(
    std::map<std::string, OpsKernelInfoStorePtr>& ops_kernel_info_stores) {
  DVPP_ENGINE_LOG_DEBUG("dvpp engine GetOpsKernelInfoStores start");
  DVPP_CHECK_IF_THEN_DO(dvpp_ops_kernel_info_store_ == nullptr,
      DVPP_ENGINE_LOG_EVENT("dvpp ops kernel info store is not supported");
      return);

  ops_kernel_info_stores[kDvppOpsKernel] = dvpp_ops_kernel_info_store_;
  DVPP_ENGINE_LOG_DEBUG("dvpp engine GetOpsKernelInfoStores success");
}

ge::Status DvppEngine::Finalize() const {
  DVPP_ENGINE_LOG_DEBUG("dvpp engine Finalize start");
  DVPP_ENGINE_LOG_DEBUG("dvpp engine Finalize success");
  return ge::SUCCESS;
}
} // namespace dvpp

extern "C" {
ge::Status Initialize(const std::map<std::string, std::string>& options) {
  return dvpp::DvppEngine::Instance().Initialize(options);
}

void GetGraphOptimizerObjs(
    std::map<std::string, GraphOptimizerPtr>& graph_optimizers) {
  dvpp::DvppEngine::Instance().GetGraphOptimizerObjs(graph_optimizers);
}

void GetOpsKernelBuilderObjs(
    std::map<std::string, OpsKernelBuilderPtr>& ops_kernel_builders) {
  dvpp::DvppEngine::Instance().GetOpsKernelBuilderObjs(ops_kernel_builders);
}

void GetOpsKernelInfoStores(
    std::map<std::string, OpsKernelInfoStorePtr>& ops_kernel_info_stores) {
  dvpp::DvppEngine::Instance().GetOpsKernelInfoStores(ops_kernel_info_stores);
}

ge::Status Finalize() {
  return dvpp::DvppEngine::Instance().Finalize();
}
} // extern "C"
