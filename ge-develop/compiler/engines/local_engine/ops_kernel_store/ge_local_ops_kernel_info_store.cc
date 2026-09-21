/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_info_store.h"
#include <memory>
#include "engines/local_engine/common/constant/constant.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "engines/local_engine/ops_kernel_store/op/op_factory.h"

namespace ge {
namespace ge_local {
Status GeLocalOpsKernelInfoStore::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  GELOGI("GeLocalOpsKernelInfoStore init start.");

  OpInfo default_op_info = {.engine = kGeLocalEngineName,
                            .opKernelLib = kGeLocalOpKernelLibName,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false,
                            .opFileName = "",
                            .opFuncName = ""};
  // Init op_info_map_
  auto all_ops = OpFactory::Instance().GetAllOps();
  for (auto &op : all_ops) {
    op_info_map_[op] = default_op_info;
  }

  GELOGI("GeLocalOpsKernelInfoStore inited success. op num=%zu", op_info_map_.size());

  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

void GeLocalOpsKernelInfoStore::GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const {
  infos = op_info_map_;
}

bool GeLocalOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &reason) const {
  (void)reason;
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}

Status GeLocalOpsKernelInfoStore::CreateSession(const std::map<std::string, std::string> &session_options) {
  // Do nothing
  (void)session_options;
  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::DestroySession(const std::map<std::string, std::string> &session_options) {
  // Do nothing
  (void)session_options;
  return SUCCESS;
}
}  // namespace ge_local
}  // namespace ge
