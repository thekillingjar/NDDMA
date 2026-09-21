/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rts_engine.h"
#include <map>
#include <memory>
#include <string>
#include "common/constant/constant.h"

#include "ops_kernel_store/rts_ops_kernel_info.h"
#include "graph_optimizer/rts_graph_optimizer.h"
#include "ops_kernel_store/rts_ffts_plus_ops_kernel_info.h"
#include "graph_optimizer/rts_ffts_plus_graph_optimizer.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
RtsEngine &RtsEngine::Instance() {
  static RtsEngine instance;
  return instance;
}

ge::Status RtsEngine::Initialize(const std::map<string, string> &options) {
  (void)options;
  if (ops_kernel_store_ == nullptr) {
    try {
      ops_kernel_store_ = std::make_shared<RtsOpsKernelInfoStore>();
    } catch (...) {
      RTS_REPORT_INNER_ERROR("make_shared RtsOpsKernelInfoStore exception.");
      return FAILED;
    }
    if (ops_kernel_store_ == nullptr) {
      RTS_REPORT_CALL_ERROR("Make RtsOpsKernelInfoStore failed.");
      return FAILED;
    }
  }

  if (graph_optimizer_ptr_ == nullptr) {
    try {
      graph_optimizer_ptr_ = std::make_shared<RtsGraphOptimizer>();
    } catch (...) {
      RTS_REPORT_INNER_ERROR("make_shared RtsGraphOptimizer exception.");
      return FAILED;
    }
    if (graph_optimizer_ptr_ == nullptr) {
      RTS_REPORT_CALL_ERROR("Make RtsGraphOptimizer failed.");
      return FAILED;
    }
  }

  if (ffts_plus_ops_kernel_store_ == nullptr) {
    try {
      ffts_plus_ops_kernel_store_ = std::make_shared<RtsFftsPlusOpsKernelInfoStore>();
    } catch (...) {
      RTS_REPORT_INNER_ERROR("make_shared RtsFftsPlusOpsKernelInfoStore exception.");
      return FAILED;
    }
    if (ffts_plus_ops_kernel_store_ == nullptr) {
      RTS_REPORT_CALL_ERROR("Make RtsFftsPlusOpsKernelInfoStore failed.");
      return FAILED;
    }
  }

  if (ffts_plus_graph_optimizer_ptr_ == nullptr) {
    try {
      ffts_plus_graph_optimizer_ptr_ = std::make_shared<RtsFftsPlusGraphOptimizer>();
    } catch (...) {
      RTS_REPORT_INNER_ERROR("make_shared RtsFftsPlusGraphOptimizer exception.");
      return FAILED;
    }
    if (ffts_plus_graph_optimizer_ptr_ == nullptr) {
      RTS_REPORT_CALL_ERROR("Make RtsFftsPlusGraphOptimizer failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

void RtsEngine::GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &opsKernelMap) {
  if (ops_kernel_store_ != nullptr) {
    // add buildin opsKernel to opsKernelInfoMap
    opsKernelMap[RTS_OP_KERNEL_LIB_NAME] = ops_kernel_store_;
  }

  if (ffts_plus_ops_kernel_store_ != nullptr) {
    // add buildin ffts_plus_ops_kernel_store_to opsKernelInfoMap
    opsKernelMap[RTS_FFTS_PLUS_OP_KERNEL_LIB_NAME] = ffts_plus_ops_kernel_store_;
  }
}

void RtsEngine::GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graphOptimizers) {
  if (graph_optimizer_ptr_ != nullptr) {
    graphOptimizers[RTS_GRAPH_OPTIMIZER_LIB_NAME] = graph_optimizer_ptr_;
  }

  if (ffts_plus_graph_optimizer_ptr_ != nullptr) {
    graphOptimizers[RTS_FFTS_PLUS_GRAPH_OPTIMIZER_LIB_NAME] = ffts_plus_graph_optimizer_ptr_;
  }
}

ge::Status RtsEngine::Finalize() {
  ops_kernel_store_ = nullptr;
  ffts_plus_ops_kernel_store_ = nullptr;
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce

ge::Status Initialize(const std::map<string, string> &options) {
  return cce::runtime::RtsEngine::Instance().Initialize(options);
}

void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &opsKernelMap) {
  cce::runtime::RtsEngine::Instance().GetOpsKernelInfoStores(opsKernelMap);
}

void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graphOptimizers) {
  cce::runtime::RtsEngine::Instance().GetGraphOptimizerObjs(graphOptimizers);
}

ge::Status Finalize() {
  return cce::runtime::RtsEngine::Instance().Finalize();
}
