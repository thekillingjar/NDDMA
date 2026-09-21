/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/local_engine/engine/ge_local_engine.h"
#include <map>
#include <memory>
#include <string>
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "engines/local_engine/common/constant/constant.h"
#include "common/plugin/ge_make_unique_util.h"
#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_info_store.h"
#include "engines/local_engine/engine/ge_local_graph_optimizer.h"
#include "base/err_msg.h"

namespace ge {
namespace ge_local {
const std::string kGeLocalGraphOptimizer = "ge_local_graph_optimizer";

GeLocalEngine &GeLocalEngine::Instance() {
  static GeLocalEngine instance;
  return instance;
}

Status GeLocalEngine::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  if (ops_kernel_store_ == nullptr) {
    ops_kernel_store_ = MakeShared<GeLocalOpsKernelInfoStore>();
    if (ops_kernel_store_ == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "create GeLocalOpsKernelInfoStore failed.");
      GELOGE(FAILED, "[Call][MakeShared] Make GeLocalOpsKernelInfoStore failed.");
      return FAILED;
    }
  }

  if (ge_local_graph_optimizer_ == nullptr) {
    ge_local_graph_optimizer_ = MakeShared<GeLocalGraphOptimizer>();
    GE_ASSERT_NOTNULL(ge_local_graph_optimizer_, "[Call][MakeShared] Make GeLocalGraphOptimizer failed.");
  }
  return SUCCESS;
}

void GeLocalEngine::GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  if (ops_kernel_store_ != nullptr) {
    // add buildin opsKernel to opsKernelInfoMap
    ops_kernel_map[kGeLocalOpKernelLibName] = ops_kernel_store_;
  }
}

void GeLocalEngine::GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) const {
  if (ge_local_graph_optimizer_ != nullptr) {
    GELOGD("Registed ge_local_graph_optimizer.");
    graph_optimizers[kGeLocalGraphOptimizer] = ge_local_graph_optimizer_;
  }
}

Status GeLocalEngine::Finalize() {
  ops_kernel_store_ = nullptr;
  ge_local_graph_optimizer_ = nullptr;
  return SUCCESS;
}
}  // namespace ge_local
}  // namespace ge

ge::Status Initialize(const std::map<std::string, std::string> &options) {
  return ge::ge_local::GeLocalEngine::Instance().Initialize(options);
}

void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  ge::ge_local::GeLocalEngine::Instance().GetOpsKernelInfoStores(ops_kernel_map);
}

void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  ge::ge_local::GeLocalEngine::Instance().GetGraphOptimizerObjs(graph_optimizers);
}

ge::Status Finalize() { return ge::ge_local::GeLocalEngine::Instance().Finalize(); }
