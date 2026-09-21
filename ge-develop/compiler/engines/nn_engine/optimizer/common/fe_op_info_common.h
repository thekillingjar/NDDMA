/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_FE_OP_INFO_COMMON_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_FE_OP_INFO_COMMON_H_

#include "common/op_info_common.h"
#include "ops_store/ops_kernel_common.h"
#include "ops_store/op_kernel_info.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "tensor_engine/fusion_types.h"

namespace fe {
const std::map<te::CheckSupportedResult, std::string> CHECKSUPPORTED_STR_MAP = {
          {te::NOT_SUPPORTED, STR_NOT_SUPPORTED},
          {te::FULLY_SUPPORTED, STR_FULLY_SUPPORTED},
          {te::PARTIALLY_SUPPORTED, STR_PARTIALLY_SUPPORTED}};

const std::map<OpConstValueDepend, te::OP_VALUE_DEPEND> VALUE_DEPEND_MAP {
          {CONST_IGNORE, te::VALUE_DEPEND_IGNORE},
          {CONST_REQUIRED, te::VALUE_DEPEND_REQUIRED},
          {CONST_OPTIONAL, te::VALUE_DEPEND_OPTIONAL}};

std::string GetCheckSupportedString(te::CheckSupportedResult &check_supported);

std::string GetStrByFinComTaskVec(const std::vector<te::FinComTask> &fin_com_task);

using RunInfoPtr = std::shared_ptr<optiling::utils::OpRunInfo>;

Status TilingForOneNode(const ge::NodePtr &node);

Status UpdateTileFwkKernelInfo(const ge::OpDescPtr &op_desc);

void GetCoreTypeAndRatio(const ge::OpDescPtr &op_desc, const int64_t cube_ratio, const int64_t vector_ratio,
                         std::string &core_type, int32_t &ratio);

Status UpdateMixByTilingKey(const ge::OpDescPtr &op_desc, const RunInfoPtr &tiling_info);

Status TilingResultCheck(const ge::OpDescPtr &op_desc, RunInfoPtr &tiling_info);

void SetTilingResult(const ge::OpDescPtr &op_desc, const RunInfoPtr &tiling_info);

Status GetRunInfoFromGe(const ge::NodePtr &node, const PlatFormInfos &platform_infos, const RunInfoPtr &tiling_info);

Status UpdateTilingResult(const RunInfoPtr &tiling_info, const RunInfoPtr &temp_tiling_info);

Status TilingForSoftSyncOp(const ge::NodePtr &node, PlatFormInfos &platform_infos,
                           const RunInfoPtr &tiling_info);
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_FE_OP_INFO_COMMON_H_
