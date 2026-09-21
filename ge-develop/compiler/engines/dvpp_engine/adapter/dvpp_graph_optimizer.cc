/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_graph_optimizer.h"
#include "ascend910b/dvpp_optimizer_910b.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"
#include "ge/ge_api_types.h"

namespace dvpp {
ge::Status DvppGraphOptimizer::Initialize(
    const std::map<std::string, std::string>& options,
    ge::OptimizeUtility* const optimize_utility) {
  (void)optimize_utility;
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer Initialize start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ != nullptr,
      DVPP_ENGINE_LOG_INFO("dvpp graph optimizer already inited");
      return ge::SUCCESS);

  auto iter = options.find(ge::SOC_VERSION);
  DVPP_CHECK_IF_THEN_DO(iter == options.end(),
      DVPP_REPORT_INNER_ERR_MSG("can not find [%s]", ge::SOC_VERSION.c_str());
      return ge::FAILED);

  try {
    if (IsSocVersionMLR1(iter->second)) {
      dvpp_optimizer_ = std::make_shared<DvppOptimizer910B>();
    } else {
      DVPP_REPORT_INNER_ERR_MSG("dvpp graph optimizer not support version[%s]",
                              iter->second.c_str());
      return ge::FAILED;
    }
  } catch (...) {
    DVPP_REPORT_INNER_ERR_MSG("create dvpp graph optimizer failed");
    return ge::FAILED;
  }

  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer Initialize success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::Finalize() {
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer Finalize start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ == nullptr,
      DVPP_ENGINE_LOG_WARNING(
          "dvpp graph optimizer is not inited, please call Initialize first");
      return ge::SUCCESS);

  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer Finalize success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::OptimizeGraphPrepare(ge::ComputeGraph& graph) {
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeGraphPrepare start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer is not inited, please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_optimizer_->OptimizeGraphPrepare(graph);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer OptimizeGraphPrepare failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeGraphPrepare success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph& graph) {
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeOriginalGraph start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer is not inited, please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_optimizer_->OptimizeOriginalGraph(graph);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer OptimizeOriginalGraph failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeOriginalGraph success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::OptimizeOriginalGraphJudgeInsert(
    ge::ComputeGraph& graph) {
  DVPP_ENGINE_LOG_DEBUG(
      "dvpp graph optimizer OptimizeOriginalGraphJudgeInsert start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer is not inited, please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_optimizer_->OptimizeOriginalGraphJudgeInsert(graph);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer OptimizeOriginalGraphJudgeInsert failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG(
      "dvpp graph optimizer OptimizeOriginalGraphJudgeInsert success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph& graph) {
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeFusedGraph start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer is not inited, please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_optimizer_->OptimizeFusedGraph(graph);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp graph optimizer OptimizeFusedGraph failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeFusedGraph success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph& graph) {
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeWholeGraph start");
  DVPP_CHECK_IF_THEN_DO(dvpp_optimizer_ == nullptr,
      DVPP_REPORT_INNER_ERR_MSG(
          "dvpp graph optimizer is not inited, please call Initialize first");
      return ge::FAILED);

  auto error_code = dvpp_optimizer_->OptimizeWholeGraph(graph);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("dvpp graph optimizer OptimizeWholeGraph failed");
      return ge::FAILED);

  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer OptimizeWholeGraph success");
  return ge::SUCCESS;
}

ge::Status DvppGraphOptimizer::GetAttributes(
    ge::GraphOptimizerAttribute& attrs) const {
  // 这里注意 GE的流程是先调用GetAttributes 后调用Initialize
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer GetAttributes start");
  attrs.engineName = kDvppEngineName;
  DVPP_ENGINE_LOG_DEBUG("dvpp graph optimizer GetAttributes success");
  return ge::SUCCESS;
}
} // namespace dvpp
