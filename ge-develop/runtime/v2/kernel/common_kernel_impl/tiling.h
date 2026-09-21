/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_TILING_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_TILING_H_
#include "graph/ge_error_codes.h"
#include "common/math/math_util.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/dfx_info_filler.h"
#include "graph/fast_graph/fast_node.h"
#include "register/op_impl_kernel_registry.h"
#include "register/op_tiling_registry.h"
#include "kernel/tiling_cache.h"
#include "engine/aicore/launch_kernel/rt_kernel_launch_args_ex.h"

namespace gert {
namespace kernel {
constexpr size_t kAiCoreWorkspaceAlignment = 512U;

ge::graphStatus Tiling(KernelContext *context);
ge::graphStatus FallibleTiling(KernelContext *context);
ge::graphStatus TilingAppendWorkspace(KernelContext *context);
ge::graphStatus TilingAppendMemCheck(KernelContext *context);
ge::graphStatus FillTilingInfo(const KernelContext *context, ExceptionDumpInfoWrapper &wrapper);
std::vector<std::string> PrintTilingData(const KernelContext *context);
ge::graphStatus RefreshOutputAddr(KernelContext *context, RtKernelLaunchArgsEx *launch_arg);
ge::graphStatus AlignWorkspaceSizes(KernelContext *context);
ge::graphStatus UpdateIOShapeToOp(KernelContext *context, ge::Operator &op);
ge::graphStatus UpdateTilingOutputsToContext(const optiling::OpRunInfoV2 &op_run_info, KernelContext *context);

 /*
 * outputs, find tiling func v1的outputs以如下顺序排列：
 * outputs[0]: tiling-version
 * outputs[1]: tiling-parse-func
 * outputs[2]: tiling-func
 */
enum class FindCompatibleTilingFuncOutputIndex {
  kTilingVersion,
  kTilingParseFunc,
  kTilingFunc,
  // add new output definitions here
  kFindFuncOutputNum
};

enum class CompatibleTilingParseInputIndex {
  kOp,
  kCompileInfoJson,
  kCompileInfoKey,
  kTilingVersion,
  kTilingParseFun,
  // add new output definitions here
  kFindFuncInputNum
};

enum class CompatibleTilingInputIndex {
  kOp,
  kCompileInfo,
  kTilingVersion,
  kTilingFwkData,
  // add new output definitions here
  kTilingFuncInputNum
};

/*
 * outputs, find tiling func v1的outputs以如下顺序排列：
 * outputs[0]: tiling-version
 * outputs[1]: tiling-parse-func
 * outputs[2]: tiling-func
 */
enum class TilingVersion {
  kV3,
  kV4,
  // add new output definitions here
  kInvalidVersion
};

// 为了保证TilingContext接口兼容, FwkData放在kDeterministic之前作为原TilingFunc占位
enum class TilingFixedInputIndex {
  kCompileInfo = 0,
  kPlatfrom,
  kFwkData,
  kDeterministic,
  kDeterministicLevel,
  kNum,
};

enum class TilingExOutputIndex {
  kRtArg = TilingContext::kOutputNum,
  kNum,
};

enum class FallibleTilingExOutputIndex {
  kRtArg = TilingContext::kOutputNum,
  kTilingStatus,
  kFallibleOutputNum,
};

enum class TilingFwkDataInput {
  kTilingFunc,
  kRtArg,
  kNum,
};

enum class CacheableTilingFwkDataInput {
  kDataDependency = static_cast<int32_t>(TilingFwkDataInput::kNum),
  kBuildTilingCacheKeyFuncName
};

// TilingFrameworkDataInput是TilingContext的最后一个输入，后续不对外呈现的输入请在此添加
struct TilingFwkData {
  void *tiling_func;
  RtKernelLaunchArgsEx *launch_arg;
};

struct CacheableTilingFwkData {
  TilingFwkData fwk_data;
  TilingCacheManager tiling_cache_mgr;
  size_t data_dependency;
  char *build_tiling_cache_key_func_name;
};

ge::graphStatus ApplyTilingCache(KernelContext *context, const TilingCacheValue &buffer);
ge::graphStatus AddTilingCache(KernelContext *context, const TilingCacheKey &key,
                               CacheableTilingFwkData &cacheable_fwk_data);

inline ge::graphStatus BuildTilingOutputs(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  //todo tilingfunc
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto node_type = extend_context->GetNodeType();
  auto tiling_data_av = context->GetOutput(TilingContext::kOutputTilingData);
  auto workspace_av = context->GetOutput(TilingContext::kOutputWorkspace);
  auto tiling_key = context->GetOutputPointer<uint64_t>(TilingContext::kOutputTilingKey);
  auto atomic_clean_flag = context->GetOutputPointer<bool>(TilingContext::kOutputAtomicCleanFlag);
  auto block_dim = context->GetOutputPointer<uint64_t>(TilingContext::kOutputBlockDim);
  auto tiling_cond = context->GetOutputPointer<int32_t>(TilingContext::kOutputTilingCond);
  if ((node_type == nullptr) || (tiling_data_av == nullptr) || (workspace_av == nullptr) || (tiling_key == nullptr) ||
      (atomic_clean_flag == nullptr) || (block_dim == nullptr) || (tiling_cond == nullptr)) {
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "[BuildTilingOutputs] para check failed, node_type %p, tiling_data_av %p, workspace_av %p", node_type,
           tiling_data_av, workspace_av);
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "[BuildTilingOutputs] para check failed, tiling_key %p, atomic_clean_flag %p, block_dim %p", tiling_key,
           atomic_clean_flag, block_dim);
    return ge::GRAPH_FAILED;
  }

  *tiling_key = 0UL;
  *atomic_clean_flag = true;
  *block_dim = 0UL;
  *tiling_cond = -1;

  constexpr int64_t kMaxWorkspaceCount = 16;
  auto workspace_size = ContinuousVector::Create<size_t>(kMaxWorkspaceCount);
  if (workspace_size == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspace_av->SetWithDefaultDeleter<uint8_t[]>(workspace_size.release());
  return ge::GRAPH_SUCCESS;
}
} // namespace kernel
} // namespace gert
#endif // AIR_CXX_RUNTIME_V2_KERNEL_TILING_H_
