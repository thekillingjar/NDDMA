/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_ops_kernel_builder.h"
#include "ascend910b/dvpp_builder_910b.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"
#include "register/ops_kernel_builder_registry.h"

namespace dvpp {
REGISTER_OPS_KERNEL_BUILDER(kDvppOpsKernel, DvppOpsKernelBuilder);
ge::Status DvppOpsKernelBuilder::Initialize(
    const std::map<std::string, std::string>& options) {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder Initialize start");
  DVPP_CHECK_IF_THEN_DO(dvpp_builder_ != nullptr,
      DVPP_ENGINE_LOG_INFO("dvpp ops kernel builder already inited");
      return ge::SUCCESS);

  auto iter = options.find(ge::SOC_VERSION);
  DVPP_CHECK_IF_THEN_DO(iter == options.end(),
      DVPP_REPORT_INNER_ERR_MSG("can not find [%s]", ge::SOC_VERSION.c_str());
      return ge::FAILED);

  try {
    if (IsSocVersionMLR1(iter->second)) {
      dvpp_builder_ = std::make_shared<DvppBuilder910B>();
    } else {
      // 因为GE对于builder采用宏注册 绕过了engine的拦截 需要这里拦截
      DVPP_ENGINE_LOG_EVENT("dvpp ops kernel builder is not supported");
      return ge::SUCCESS;
    }
  } catch (...) {
    DVPP_REPORT_INNER_ERR_MSG("create dvpp ops kernel builder failed");
    return ge::FAILED;
  }

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder Initialize success");
  return ge::SUCCESS;
}

ge::Status DvppOpsKernelBuilder::Finalize() {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder Finalize start");
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder Finalize success");
  return ge::SUCCESS;
}

ge::Status DvppOpsKernelBuilder::CalcOpRunningParam(ge::Node& node) {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder CalcOpRunningParam start");
  DVPP_CHECK_IF_THEN_DO(dvpp_builder_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel builder is not inited, "
                              "please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_builder_->CalcOpRunningParam(node);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp ops kernel builder CalcOpRunningParam failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder CalcOpRunningParam success");
  return ge::SUCCESS;
}

ge::Status DvppOpsKernelBuilder::GenerateTask(const ge::Node& node,
    ge::RunContext& context, std::vector<domi::TaskDef>& tasks) {
  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder GenerateTask start");
  DVPP_CHECK_IF_THEN_DO(dvpp_builder_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel builder is not inited, "
                              "please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_builder_->GenerateTask(node, context, tasks);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp ops kernel builder GenerateTask failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp ops kernel builder GenerateTask success");
  return ge::SUCCESS;
}
} // namespace dvpp
