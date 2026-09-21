/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_ops_kernel_info_store.h"
#include "ascend910b/dvpp_info_store_910b.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"
#include "ge/ge_api_types.h"

namespace dvpp {
ge::Status DvppOpsKernelInfoStore::Initialize(
    const std::map<std::string, std::string>& options) {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store Initialize start");
  DVPP_CHECK_IF_THEN_DO(dvpp_info_store_ != nullptr,
      DVPP_ENGINE_LOG_INFO("dvpp ops kernel info store already inited");
      return ge::SUCCESS);

  auto iter = options.find(ge::SOC_VERSION);
  DVPP_CHECK_IF_THEN_DO(iter == options.end(),
      DVPP_REPORT_INNER_ERR_MSG("can not find [%s]", ge::SOC_VERSION.c_str());
      return ge::FAILED);

  std::string json_file;
  try {
    if (IsSocVersionMLR1(iter->second)) {
      dvpp_info_store_ = std::make_shared<DvppInfoStore910B>();
      json_file = kJsonFileAscend910B;
    } else {
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp ops kernel info store not support version[%s]",
          iter->second.c_str());
      return ge::FAILED;
    }
  } catch (...) {
    DVPP_REPORT_INNER_ERR_MSG("create dvpp ops kernel info store failed");
    return ge::FAILED;
  }

  auto error_code = dvpp_info_store_->InitInfoStore(json_file);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store Initialize failed");
      dvpp_info_store_ = nullptr; // 这里置空 避免再次调用时误以为已完成初始化
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store Initialize success");
  return ge::SUCCESS;
}

ge::Status DvppOpsKernelInfoStore::Finalize() {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store Finalize start");
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store Finalize success");
  return ge::SUCCESS;
}

void DvppOpsKernelInfoStore::GetAllOpsKernelInfo(
    std::map<std::string, ge::OpInfo>& infos) const {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store GetAllOpsKernelInfo start");
  DVPP_CHECK_IF_THEN_DO(dvpp_info_store_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store is not inited, "
                              "please call Initialize first");
      return);

  dvpp_info_store_->GetAllOpsKernelInfo(infos);
  DVPP_ENGINE_LOG_DEBUG(
      "dvpp ops kernel info store GetAllOpsKernelInfo success");
}

bool DvppOpsKernelInfoStore::CheckSupported(
    const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason) const {
  (void)op_desc_ptr;
  (void)unsupported_reason;
  DVPP_ENGINE_LOG_WARNING(
      "should call the CheckSupported function with parameter node");
  return false;
}

bool DvppOpsKernelInfoStore::CheckSupported(
    const ge::NodePtr& node, std::string& unsupported_reason,
    ge::CheckSupportFlag& flag) const {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store CheckSupported start");
  DVPP_CHECK_IF_THEN_DO(dvpp_info_store_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store is not inited, "
                              "please call Initialize first");
      return false);

  (void)flag;
  bool ret = dvpp_info_store_->CheckSupported(node, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_ENGINE_LOG_EVENT("dvpp ops kernel info store CheckSupported failed");
      return false);

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store CheckSupported success");
  return true;
}

ge::Status DvppOpsKernelInfoStore::LoadTask(ge::GETaskInfo& task) {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store LoadTask start");
  DVPP_CHECK_IF_THEN_DO(dvpp_info_store_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store is not inited, "
                              "please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_info_store_->LoadTask(task);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store LoadTask failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store LoadTask success");
  return ge::SUCCESS;
}

ge::Status DvppOpsKernelInfoStore::UnloadTask(ge::GETaskInfo& task) {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store UnloadTask start");
  DVPP_CHECK_IF_THEN_DO(dvpp_info_store_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store is not inited, "
                              "please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_info_store_->UnloadTask(task);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel info store UnloadTask failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel info store UnloadTask success");
  return ge::SUCCESS;
}
} // namespace dvpp
