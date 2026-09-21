/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_info_store.h"
#include "common/dvpp_ops_checker.h"
#include "common/dvpp_ops_lib.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"

namespace dvpp {
void DvppInfoStore::LoadAllOpsKernelInfo() {
  auto all_ops = DvppOpsLib::Instance().AllOps();
  for (const auto& dvpp_op : all_ops) {
    const auto& dvpp_op_info = dvpp_op.second;
    ge::OpInfo op_info;
    op_info.engine = dvpp_op_info.engine;
    op_info.computeCost = dvpp_op_info.computeCost;
    op_info.flagAsync = dvpp_op_info.flagAsync;
    op_info.flagPartial = dvpp_op_info.flagPartial;
    op_info.opKernelLib = dvpp_op_info.opKernelLib;
    op_info.isAtomic = false;
    all_ops_kernel_info_[dvpp_op.first] = op_info;
  }
}

DvppErrorCode DvppInfoStore::InitInfoStore(std::string& file_name) {
  DVPP_ENGINE_LOG_INFO("Begin to initialize dvpp ops info store");

  auto error_code = DvppOpsLib::Instance().Initialize(file_name);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops lib init failed");
      return error_code);

  LoadAllOpsKernelInfo();
  return DvppErrorCode::kSuccess;
}

void DvppInfoStore::GetAllOpsKernelInfo(
    std::map<std::string, ge::OpInfo>& infos) const {
  infos = all_ops_kernel_info_;
}

bool DvppInfoStore::CheckSupported(
    const ge::NodePtr& node, std::string& unsupported_reason) const {
  // 虽然原图优化已经调用过CheckSupported 这里依然调用 避免中途被修改
  // 这里无需判断是否使能dvpp engine 由GE保证
  return DvppOpsChecker::CheckSupported(node, unsupported_reason);
}
} // namespace dvpp
