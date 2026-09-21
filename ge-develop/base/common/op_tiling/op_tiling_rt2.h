/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_TILING_OP_TILING_RT2_H_
#define GE_COMMON_TILING_OP_TILING_RT2_H_

#include "graph/operator.h"
#include "graph/op_desc.h"
#include "register/op_tiling_registry.h"
#include "platform/platform_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "runtime/kernel.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"

namespace optiling {
using OutputsConvertorFun = std::function<ge::graphStatus(gert::KernelContext *kernel_context)>;
using ParseContextHolderPtr = std::shared_ptr<gert::KernelContextHolder>;
bool EnableRt2Tiling(const ge::OpDescPtr &op_desc);
bool EnableAtomicRt2Tiling(const ge::OpDescPtr &op_desc);
ge::graphStatus RtParseAndTiling(const ge::Operator &op, const char_t * const compile_info,
                                 const fe::PlatFormInfos &platform_infos, const OutputsConvertorFun &callback,
                                 const gert::OpImplSpaceRegistryV2Ptr &space_registry);
ge::graphStatus AutofuseNodeWithMatmulTiling(const ge::Operator &op, const fe::PlatFormInfos &ge_platform_infos,
                                             OpRunInfoV2 &run_info, ge::ConstNodePtr node);
ge::graphStatus AicoreRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info);
ge::graphStatus AtomicRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info);
ge::graphStatus SoftSyncOpRtParseAndTiling(const ge::Operator &op, fe::PlatFormInfos &platform_infos,
                                           OpRunInfoV2 &run_info,
                                           const gert::OpImplSpaceRegistryV2Ptr &space_registry);
ge::graphStatus FftsRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                     std::vector<OpRunInfoV2> &op_run_infos);
ge::graphStatus GetDeterministicLevel(int32_t &deterministic_level);
const std::string OP_TILING_PARSE_RESULT = "tiling_parse_result";
}  // namespace optiling

#endif  // GE_COMMON_TILING_OP_TILING_RT2_H_
