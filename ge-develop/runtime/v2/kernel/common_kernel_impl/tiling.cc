/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling.h"
#include "register/kernel_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/math_util.h"
#include "graph/args_format_desc.h"
#include "common/checker.h"
#include "engine/node_converter_utils.h"
#include "exe_graph/lowering/shape_utils.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/op_tiling/tiling_dfx.h"
#include "adump_pub.h"
#include "mmpa/mmpa_api.h"
#include "kernel/tiling_cache.h"
#include "kernel/kernel_log.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "aicore/converter/autofuse_node_converter.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kLaunchArgOffset = 0UL;
constexpr size_t kWorkSpaceOffset = 1UL;
constexpr size_t kOriOpParamSizeOffset = 2UL;
constexpr size_t kIsMemCheckEnableOffset = 3UL;
constexpr size_t kIsArgsExceptionOffset = 4UL;
constexpr size_t kArgsSizeListOffset = 5UL;
constexpr size_t kArgsIndexToIoIndexOffset = 6UL;
constexpr size_t kFwkDataOffset =
    static_cast<size_t>(TilingFixedInputIndex::kNum) - static_cast<size_t>(TilingFixedInputIndex::kFwkData);
// 每个算子的最大缓存数和老化阈值
constexpr size_t kTilingCacheSizePerOp = 120UL;
constexpr size_t kTilingCacheEvictNum = 8UL;

std::vector<std::string> TilingAppendWorkSpaceTracer(const KernelContext *context) {
  auto tiling_ws = context->GetInputPointer<gert::ContinuousVector>(0U);
  GE_ASSERT_NOTNULL(tiling_ws);
  auto known_ws = context->GetInputPointer<gert::ContinuousVector>(1U);
  GE_ASSERT_NOTNULL(known_ws);

  std::stringstream ss;
  ss << "Tiling workspace num: " << tiling_ws->GetSize() << ", known workspace num: " << known_ws->GetSize()
     << ", Tiling append workspace num: " << known_ws->GetSize() << ", append workspace size:[";
  auto ws_concat =
      const_cast<TypedContinuousVector<int64_t> *>(context->GetOutputPointer<TypedContinuousVector<int64_t>>(0U));
  GE_ASSERT_NOTNULL(ws_concat);
  auto ws_data = ws_concat->MutableData();
  for (size_t i = 0UL; i < ws_concat->GetSize(); ++i) {
    ss << ws_data[i];
    if (i != ws_concat->GetSize() - 1UL) {
      ss << " ";
    }
  }
  ss << "]";
  return {ss.str()};
}

size_t DfxCalcShapeSizeNum(const KernelContext *context,
                           const TypedContinuousVector<optiling::ArgsIndexToIoIndex> &args_idx_to_io_idx) {
  // 计算shape_size_num
  size_t shape_size_num = 0UL;
  const auto args_idx_to_io_idx_vec = args_idx_to_io_idx.GetData();
  for (size_t i = 0UL; i < args_idx_to_io_idx.GetSize(); i++) {
    const auto io_index = args_idx_to_io_idx_vec[i].io_index;
    if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kInput) {
      const auto input_shape = context->GetInputPointer<StorageShape>(io_index);
      GE_ASSERT_NOTNULL(input_shape);
      auto in_shape = input_shape->GetStorageShape();
      shape_size_num += (in_shape.GetDimNum() + 1U);  // dim + dimnum
    } else if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kOutput) {
      shape_size_num++;  // output shape 填0占位
    }
  }
  return shape_size_num;
}

ge::graphStatus DfxUpdateTensorSizeAndShape(
    KernelContext *context, uint64_t *dfx_dump_addr, const size_t args_size_num, const size_t workspace_num,
    const TypedContinuousVector<optiling::ArgsIndexToIoIndex> &args_idx_to_io_idx) {
  const auto extend_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  const size_t io_shape_num =
      extend_kernel_context->GetComputeNodeInputNum() + extend_kernel_context->GetComputeNodeOutputNum();
  const auto args_idx_to_io_idx_vec = args_idx_to_io_idx.GetData();
  uint64_t *shape_addr = dfx_dump_addr + args_size_num + workspace_num;
  for (size_t i = 0UL; i < args_idx_to_io_idx.GetSize(); i++) {
    const auto args_idx = args_idx_to_io_idx_vec[i].args_index;
    const auto io_idx = args_idx_to_io_idx_vec[i].io_index;

    GE_ASSERT_TRUE((args_idx < args_size_num), "args_idx[%zu] is not less than args_size_num[%zu]", args_idx,
                   args_size_num);

    if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kInput) {
      const auto input_shape = context->GetInputPointer<StorageShape>(io_idx);
      GE_ASSERT_NOTNULL(input_shape);
      const CompileTimeTensorDesc *io_desc = nullptr;
      io_desc = extend_kernel_context->GetInputDesc(io_idx);
      GE_ASSERT_NOTNULL(io_desc);
      uint64_t tensor_size = 0UL;
      auto in_shape = input_shape->GetStorageShape();
      GE_ASSERT_SUCCESS(CalcAlignedSizeByShape(in_shape, io_desc->GetDataType(), tensor_size));
      GELOGD("[TilingAppendDfxInfo]update input tensor size, index:%zu, args index:%zu, io index:%zu, tensor_size:%llu",
             i, args_idx, io_idx, tensor_size);
      // size
      *(dfx_dump_addr + args_idx) = tensor_size;
      // shape
      const auto dim_num = in_shape.GetDimNum();
      *shape_addr++ = static_cast<uint64_t>(dim_num);
      for (size_t j = 0U; j < dim_num; j++) {
        *shape_addr++ = static_cast<uint64_t>(in_shape[j]);
      }
    } else if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kOutput) {
      const auto tensor_size = context->GetInputPointer<uint64_t>(io_shape_num + io_idx);
      GE_ASSERT_NOTNULL(tensor_size);
      GELOGD(
          "[TilingAppendDfxInfo]update output tensor size, index:%u, args index: %zu, io index: %zu, tensor_size: %llu",
          i, args_idx, io_idx, *tensor_size);
      *(dfx_dump_addr + args_idx) = *tensor_size;
      *shape_addr++ = 0U;  // output shape为0
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DfxGetDumpDataFromCache(const TilingCache *const tiling_cache, uint64_t &atomic_index) {
  // 添加缓存时保证了dfx_dump_data_num大于零且和dfx_dump_data_holder中元素(uint64_t)个数一致
  const auto &cache_value = tiling_cache->GetTilingCacheValue();
  uint64_t *dump_addr =
      ge::PtrToPtr<void, uint64_t>(Adx::AdumpGetDFXInfoAddrForDynamic(cache_value.dfx_dump_data_num, atomic_index));
  GE_ASSERT_NOTNULL(dump_addr, "dump_addr total num[%zu]", cache_value.dfx_dump_data_num);
  errno_t ret = memcpy_s(dump_addr, cache_value.dfx_dump_data_num * sizeof(uint64_t),
                         cache_value.dfx_dump_data_holder.get(), cache_value.dfx_dump_data_num * sizeof(uint64_t));
  GE_ASSERT_EOK(ret, "[TilingAppendDfxInfo] copy args size from cache failed ret[%d]", ret);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DfxAddDumpDataToCache(TilingCache *const tiling_cache, const uint64_t *const dump_addr,
                                      const size_t dump_data_num) {
  auto &cache_value = tiling_cache->GetTilingCacheValue();
  cache_value.dfx_dump_data_num = dump_data_num;
  cache_value.dfx_dump_data_holder.reset(new (std::nothrow) uint64_t[dump_data_num]);
  GE_ASSERT_NOTNULL(cache_value.dfx_dump_data_holder);
  errno_t ret = memcpy_s(cache_value.dfx_dump_data_holder.get(), cache_value.dfx_dump_data_num * sizeof(uint64_t),
                         dump_addr, dump_data_num * sizeof(uint64_t));
  GE_ASSERT_EOK(ret, "[TilingAppendDfxInfo] copy args size to cache failed ret[%d]", ret);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DfxUpdateMemcheckTensorSize(
    KernelContext *context, const TypedContinuousVector<optiling::ArgsIndexToIoIndex> &args_idx_to_io_idx,
    const size_t args_size_num, int64_t *const append_size_ptr) {
  const auto extend_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  const auto io_shape_num =
      extend_kernel_context->GetComputeNodeInputNum() + extend_kernel_context->GetComputeNodeOutputNum();
  const auto args_idx_to_io_idx_vec = args_idx_to_io_idx.GetData();
  for (size_t i = 0UL; i < args_idx_to_io_idx.GetSize(); i++) {
    auto args_idx = args_idx_to_io_idx_vec[i].args_index;
    auto io_idx = args_idx_to_io_idx_vec[i].io_index;

    GE_ASSERT_TRUE((args_idx < args_size_num), "args idx[%zu] is no less than args size num[%zu]", args_idx,
                   args_size_num);

    if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kInput) {
      const auto input_shape = context->GetInputPointer<StorageShape>(io_idx);
      GE_ASSERT_NOTNULL(input_shape);
      const CompileTimeTensorDesc *io_desc = nullptr;
      io_desc = extend_kernel_context->GetInputDesc(io_idx);
      GE_ASSERT_NOTNULL(io_desc);
      uint64_t tensor_size = 0UL;
      auto in_shape = input_shape->GetStorageShape();
      GE_ASSERT_SUCCESS(CalcAlignedSizeByShape(in_shape, io_desc->GetDataType(), tensor_size));
      GELOGD("[TilingAppendDfxInfo]update input tensor size, index:%zu, args index:%zu, io index:%zu, tensor_size:%llu",
             i, args_idx, io_idx, tensor_size);

      append_size_ptr[args_idx] = static_cast<int64_t>(tensor_size);
    } else if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kOutput) {
      const auto tensor_size = context->GetInputPointer<uint64_t>(io_shape_num + io_idx);
      GE_ASSERT_NOTNULL(tensor_size);
      GELOGD(
          "[TilingAppendDfxInfo]update output tensor size, index:%u, args index: %zu, io index: %zu, tensor_size: %llu",
          i, args_idx, io_idx, *tensor_size);
      append_size_ptr[args_idx] = static_cast<int64_t>(*tensor_size);
    }
  }

  for (size_t i = 0U; i < args_size_num; i++) {
    GELOGI("[TilingAppendDfxInfo] io size idx[%zu], val[%lld]", i, append_size_ptr[i]);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DfxUpdateMemcheckWorkspaceSize(const TypedContinuousVector<int64_t> &workspace_sizes,
                                               TilingData &tiling_data) {
  const auto workspace_sizes_vec = workspace_sizes.GetData();
  int64_t *workspace_append_addr =
      ge::PtrToPtr<void, int64_t>(tiling_data.Expand(workspace_sizes.GetSize() * sizeof(uint64_t)));
  GE_ASSERT_NOTNULL(workspace_append_addr);
  for (size_t index = 0UL; index < workspace_sizes.GetSize(); index++) {
    workspace_append_addr[index] = workspace_sizes_vec[index];
  }

  for (size_t i = 0U; i < workspace_sizes.GetSize(); i++) {
    GELOGI("[TilingAppendDfxInfo] workspace size idx[%zu], val[%lld]", i, workspace_append_addr[i]);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildGeneralTilingCacheKey(const KernelContext *context, HashBuffer &hash_buf) {
  auto input_num = context->GetInputNum();
  const auto cacheable_fwk_data = context->MutableInputPointer<CacheableTilingFwkData>(input_num - kFwkDataOffset);
  GE_ASSERT_NOTNULL(cacheable_fwk_data);
  auto &data_dependency = cacheable_fwk_data->data_dependency;

  const auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  const auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  const auto input_shape_num = compute_node_info->GetInputsNum();

  for (size_t i = 0UL; i < input_shape_num; ++i) {
    if ((data_dependency & (static_cast<size_t>(1) << i)) != 0U) {
      const auto input_tensor = context->GetInputPointer<Tensor>(i);
      GE_ASSERT_NOTNULL(input_tensor);
      hash_buf.AddParamToBuf(*input_tensor);
    } else {
      const auto input_shape = context->GetInputPointer<StorageShape>(i);
      GE_ASSERT_NOTNULL(input_shape);
      hash_buf.AddParamToBuf(input_shape->GetOriginShape());
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildSymbolTilingCacheKey(const KernelContext *context, HashBuffer &hash_buf) {
  auto input_data_num = context->GetInputValue<size_t>(0U);
  auto used_sym_src = context->GetInputPointer<TypedContinuousVector<int64_t>>(
      input_data_num + static_cast<size_t>(CacheableSymbolTilingInput::kUsedSymbolSource));
  if (used_sym_src->GetSize() == 0) {
    hash_buf.AddParamToBuf(-1);
  } else {
    for (size_t i = 0; i < used_sym_src->GetSize(); ++i) {
      auto src_value = used_sym_src->GetData()[i];
      hash_buf.AddParamToBuf(src_value);
    }
  }
  return ge::GRAPH_SUCCESS;
}
using BuildTilingCacheKey = ge::graphStatus (*)(const KernelContext *, HashBuffer &hash_buf);

const static std::unordered_map<std::string, BuildTilingCacheKey> kBuildTilingCacheKyeFuncs = {
    {"BuildGeneralTilingCacheKey", BuildGeneralTilingCacheKey},
    {"BuildSymbolTilingCacheKey", BuildSymbolTilingCacheKey}
} ;

BuildTilingCacheKey FindBuildSymbolTilingCacheKeyFunc(const std::string &func_name) {
  const auto iter = kBuildTilingCacheKyeFuncs.find(func_name);
  if (iter == kBuildTilingCacheKyeFuncs.end()) {
    return nullptr;
  }
  return iter->second;
}

}  // namespace

#define FIND_IMPL_FOR_TILING(node_type, func_name, space_registry)                                      \
  auto funcs = (space_registry)->GetOpImpl(node_type);                                                  \
  if (funcs == nullptr || funcs->func_name == nullptr) {                                                \
    funcs = (space_registry)->GetOpImpl("DefaultImpl");                                                 \
    if (funcs == nullptr || funcs->func_name == nullptr) {                                              \
      GELOGE(ge::PARAM_INVALID, "Failed to find func by type %s, func_name %s", node_type, #func_name); \
      return ge::GRAPH_FAILED;                                                                          \
    }                                                                                                   \
  }

ge::graphStatus TilingParse(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(context->GetInputNum() - 2U);
  auto space_registry = context->GetInputValue<gert::OpImplSpaceRegistryV2 *>(context->GetInputNum() - 1U);
  if (node_type == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to tiling parse, the node type is nullptr");
    return ge::GRAPH_FAILED;
  }
  if (space_registry == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to tiling parse, the registry is nullptr");
    return ge::GRAPH_FAILED;
  }

  FIND_IMPL_FOR_TILING(node_type, tiling_parse, space_registry)
  return funcs->tiling_parse(context);
}

ge::graphStatus BuildTilingParseOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto node_type = context->GetInputValue<char *>(context->GetInputNum() - 2U);
  auto space_registry = context->GetInputValue<gert::OpImplSpaceRegistryV2 *>(context->GetInputNum() - 1U);
  auto av = context->GetOutput(0);
  if (node_type == nullptr || av == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "[BuildTilingParseOutputs] para check failed node_type %s", node_type);
    return ge::GRAPH_FAILED;
  }
  if (space_registry == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to tiling parse, the registry is nullptr");
    return ge::GRAPH_FAILED;
  }

  FIND_IMPL_FOR_TILING(node_type, compile_info_creator, space_registry)
  av->Set(funcs->compile_info_creator(), funcs->compile_info_deleter);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(TilingParse).RunFunc(TilingParse).OutputsCreator(BuildTilingParseOutputs);

ge::graphStatus FindTilingFunc(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(0);
  auto space_registry = context->GetInputValue<gert::OpImplSpaceRegistryV2 *>(1);
  auto tiling_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::TilingKernelFunc>(0);
  if (node_type == nullptr || tiling_fun_ptr == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (space_registry == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to tiling parse, the registry is nullptr");
    return ge::GRAPH_FAILED;
  }

  FIND_IMPL_FOR_TILING(node_type, tiling, space_registry)
  *tiling_fun_ptr = funcs->tiling;
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FindTilingFunc).RunFunc(FindTilingFunc);

ge::graphStatus BuildTilingOutputsAppendWorkspace(const ge::FastNode *node, KernelContext *context) {
  (void)node;

  auto workspace_av = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(workspace_av);

  auto workspace_size = ContinuousVector::Create<size_t>(gert::kDefaultWorkspaceCap);
  GE_ASSERT_NOTNULL(workspace_size);
  workspace_av->SetWithDefaultDeleter<uint8_t[]>(workspace_size.release());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAppendWorkspace(KernelContext *context) {
  auto tiling_ws = context->GetInputPointer<gert::ContinuousVector>(0U);
  GE_ASSERT_NOTNULL(tiling_ws);
  std::vector<int64_t> tiling_workspaces(static_cast<const size_t *>(tiling_ws->GetData()),
                                         static_cast<const size_t *>(tiling_ws->GetData()) + tiling_ws->GetSize());

  auto known_ws = context->GetInputPointer<gert::ContinuousVector>(1U);
  GE_ASSERT_NOTNULL(known_ws);
  std::vector<int64_t> known_workspaces(static_cast<const size_t *>(known_ws->GetData()),
                                        static_cast<const size_t *>(known_ws->GetData()) + known_ws->GetSize());
  GE_ASSERT_TRUE(tiling_workspaces.size() <= known_workspaces.size(),
                 "known_workspaces size[%zu] should be larger than tiling_workspaces size[%zu]",
                 known_workspaces.size(), tiling_workspaces.size());

  auto ws_concat = context->GetOutputPointer<TypedContinuousVector<int64_t>>(0U);
  GE_ASSERT_NOTNULL(ws_concat);
  ws_concat->SetSize(known_ws->GetSize());

  auto ws_data = ws_concat->MutableData();
  for (size_t i = 0U; i < tiling_ws->GetSize(); ++i) {
    ws_data[i] = tiling_workspaces[i];
  }
  for (size_t i = tiling_ws->GetSize(); i < known_ws->GetSize(); ++i) {
    int64_t workspace_size = known_workspaces[i];
    workspace_size = (workspace_size == -1) ? 0 : workspace_size;
    GE_ASSERT_TRUE(workspace_size >= 0, "size[%ld] of workspace[%zu] cannot be negative", workspace_size, i);
    ws_data[i] = workspace_size;
  }

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(TilingAppendWorkspace)
    .RunFunc(TilingAppendWorkspace)
    .OutputsCreator(BuildTilingOutputsAppendWorkspace)
    .TracePrinter(TilingAppendWorkSpaceTracer);

ge::graphStatus RefreshOutputAddr(KernelContext *context, RtKernelLaunchArgsEx *launch_arg) {
  GE_ASSERT_NOTNULL(launch_arg);
  auto tiling_data_av = context->GetOutput(TilingContext::kOutputTilingData);
  GE_ASSERT_NOTNULL(tiling_data_av);
  auto launch_arg_av = context->GetOutput(static_cast<size_t>(TilingExOutputIndex::kRtArg));
  GE_ASSERT_NOTNULL(launch_arg_av);
  tiling_data_av->Set(&launch_arg->GetTilingData(), nullptr);
  launch_arg_av->Set(launch_arg, nullptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlignWorkspaceSizes(KernelContext *context) {
  auto workspace = context->GetOutputPointer<TypedContinuousVector<size_t>>(TilingContext::kOutputWorkspace);
  if (workspace != nullptr) {
    for (size_t i = 0U; i < workspace->GetSize(); ++i) {
      GE_ASSERT_TRUE(
          !ge::RoundUpOverflow(workspace->MutableData()[i], kAiCoreWorkspaceAlignment, workspace->MutableData()[i]),
          "size[%zu] of workspace[%zu] exceeds max size with %zu alignment",
          workspace->MutableData()[i], i, kAiCoreWorkspaceAlignment);
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingProc(KernelContext *context, ge::graphStatus &tiling_func_result) {
  auto input_num = context->GetInputNum();
  if (input_num < kFwkDataOffset) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "[Tiling] para check failed input_num %" PRId64 "", input_num);
    return ge::GRAPH_FAILED;
  }
  auto tiling_data = reinterpret_cast<TilingContext *>(context)->GetRawTilingData();
  if (tiling_data != nullptr) {
    tiling_data->SetDataSize(0);
  }
  const auto tiling_fwk_data = context->GetInputPointer<TilingFwkData>(input_num - kFwkDataOffset);
  GE_ASSERT_NOTNULL(tiling_fwk_data);
  // refresh tiling_data output addr before calling tiling_func
  GE_ASSERT_SUCCESS(RefreshOutputAddr(context, tiling_fwk_data->launch_arg));

  const auto tiling_func = reinterpret_cast<KernelRegistry::KernelFunc>(tiling_fwk_data->tiling_func);
  GE_ASSERT_NOTNULL(tiling_func);
  tiling_func_result = tiling_func(context);
  if (tiling_func_result == ge::GRAPH_SUCCESS) {
    return AlignWorkspaceSizes(context);
  }
  return ge::GRAPH_SUCCESS;
}

std::vector<std::string> PrintTilingData(const KernelContext *context) {
  const auto tiling_data = context->GetOutputPointer<TilingData *>(TilingContext::kOutputTilingData);
  if ((tiling_data == nullptr) || ((*tiling_data) == nullptr)) {
    return {};
  }
  const auto tiling_key = context->GetOutputPointer<uint64_t>(TilingContext::kOutputTilingKey);
  const auto atomic_clean_flag = context->GetOutputPointer<bool>(TilingContext::kOutputAtomicCleanFlag);
  const auto block_dim = context->GetOutputPointer<uint64_t>(TilingContext::kOutputBlockDim);
  const auto tiling_cond = context->GetOutputPointer<int32_t>(TilingContext::kOutputTilingCond);
  const auto launch_args =
      context->GetOutputPointer<RtKernelLaunchArgsEx *>(static_cast<size_t>(TilingExOutputIndex::kRtArg));
  const auto addr = (*tiling_data)->GetData();
  const auto size = (*tiling_data)->GetDataSize();

  std::stringstream ss;
  if (tiling_key != nullptr) {
    ss << "Tiling key is " << *tiling_key << ", ";
  }
  if (atomic_clean_flag != nullptr) {
    ss << "atomic clean flag is " << *atomic_clean_flag << ", ";
  }
  if (block_dim != nullptr) {
    ss << "block_dim is " << *block_dim << ", ";
  }
  if (tiling_cond != nullptr) {
    ss << "tiling_cond is " << *tiling_cond << ", ";
  }
  if (launch_args != nullptr) {
    const auto &args_cache_info = (*launch_args)->GetArgsCacheInfo();
    if (args_cache_info.cache_status == RtKernelLaunchArgsEx::TilingCacheStatus::kDisabled) {
      ss << "Tiling cache status: disabled, ";
    } else if (args_cache_info.cache_status == RtKernelLaunchArgsEx::TilingCacheStatus::kMissed ||
               args_cache_info.cache_status == RtKernelLaunchArgsEx::TilingCacheStatus::kHit) {
      ss << "Tiling cache status: "
         << (args_cache_info.cache_status == RtKernelLaunchArgsEx::TilingCacheStatus::kMissed ? "missed" : "hit")
         << ", ";
    }
  }
  ss << "tilingData size: " << size << " bytes, TilingData: ";
  PrintHex(static_cast<const uint8_t *>(addr), size, ss);
  return {ss.str()};
}

ge::graphStatus FillTilingInfo(const KernelContext *context, ExceptionDumpInfoWrapper &wrapper) {
  const auto tiling_data = context->GetOutputPointer<TilingData *>(TilingContext::kOutputTilingData);
  GE_ASSERT_NOTNULL(tiling_data);
  GE_ASSERT_NOTNULL(*tiling_data);
  wrapper.SetTilingKey(reinterpret_cast<const TilingContext *>(context)->GetTilingKey());
  wrapper.SetTilingData(reinterpret_cast<uintptr_t>((*tiling_data)->GetData()), (*tiling_data)->GetDataSize());
  return ge::SUCCESS;
}

ge::graphStatus Tiling(KernelContext *context) {
  ge::graphStatus tiling_func_result;
  GE_ASSERT_SUCCESS(TilingProc(context, tiling_func_result));
  return tiling_func_result;
}
REGISTER_KERNEL(Tiling)
    .RunFunc(Tiling)
    .OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo)
    .TracePrinter(PrintTilingData);

ge::graphStatus ApplyTilingCache(KernelContext *context, const TilingCacheValue &buffer) {
  const auto tiling_context = reinterpret_cast<gert::TilingContext *>(context);
  GE_ASSERT_SUCCESS(tiling_context->SetBlockDim(buffer.block_dim));
  GE_ASSERT_SUCCESS(tiling_context->SetLocalMemorySize(buffer.local_mem_size));
  GE_ASSERT_SUCCESS(tiling_context->SetTilingKey(buffer.tiling_key));
  GE_ASSERT_SUCCESS(tiling_context->SetNeedAtomic(buffer.atomic_clean_flag));
  GE_ASSERT_SUCCESS(tiling_context->SetTilingCond(buffer.tiling_cond));

  // 应用缓存时, tiling_context中的workspace_sizes_num为0, 此处使用GetWorkspaceSizes接口更好
  const auto cached_workspace_sizes =
      reinterpret_cast<TypedContinuousVector<size_t> *>(buffer.workspace_sizes_holder.get());
  const size_t workspace_size_num = cached_workspace_sizes->GetSize();
  auto workspace_sizes = tiling_context->GetWorkspaceSizes(workspace_size_num);
  GE_ASSERT_NOTNULL(workspace_sizes);
  for (size_t i = 0U; i < workspace_size_num; ++i) {
    workspace_sizes[i] = cached_workspace_sizes->GetData()[i];
  }
  const auto cached_launch_arg = reinterpret_cast<RtKernelLaunchArgsEx *>(buffer.launch_arg_holder.get());
  GE_ASSERT_NOTNULL(cached_launch_arg);
  cached_launch_arg->GetTilingData().SetDataSize(buffer.ori_tiling_data_size);
  cached_launch_arg->SetTilingCacheStatus(RtKernelLaunchArgsEx::TilingCacheStatus::kHit);
  GE_ASSERT_SUCCESS(RefreshOutputAddr(context, cached_launch_arg));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddTilingCache(KernelContext *context, const TilingCacheKey &key,
                               CacheableTilingFwkData &cacheable_fwk_data) {
  if (!key.IsValid()) {
    return ge::GRAPH_SUCCESS;
  }
  auto launch_arg = cacheable_fwk_data.fwk_data.launch_arg;
  auto &tiling_cache_mgr = cacheable_fwk_data.tiling_cache_mgr;
  const auto tiling_context = reinterpret_cast<TilingContext *>(context);
  TilingCacheValue new_cache{};
  new_cache.atomic_clean_flag = tiling_context->NeedAtomic();
  new_cache.tiling_cond = tiling_context->GetTilingCond();
  new_cache.local_mem_size = tiling_context->GetLocalMemorySize();
  new_cache.block_dim = tiling_context->GetBlockDim();
  new_cache.tiling_key = tiling_context->GetTilingKey();
  new_cache.ori_tiling_data_size = launch_arg->GetTilingData().GetDataSize();

  // ContinuousVector对象和数据内存连续，按照workspaces_sizes实际数量创建缓存，workspace_size_num最大是16
  const auto src_workspace_sizes =
      context->GetOutputPointer<gert::TypedContinuousVector<size_t>>(gert::TilingContext::kOutputWorkspace);
  GE_ASSERT_NOTNULL(src_workspace_sizes);
  const size_t workspace_size_num = src_workspace_sizes->GetSize();
  auto dst_workspace_sizes_holder = ContinuousVector::Create<size_t>(workspace_size_num);
  GE_ASSERT_NOTNULL(dst_workspace_sizes_holder);
  const auto dst_workspace_sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(dst_workspace_sizes_holder.get());
  GE_ASSERT_SUCCESS(dst_workspace_sizes->SetSize(workspace_size_num));
  for (size_t i = 0; i < workspace_size_num; ++i) {
    dst_workspace_sizes->MutableData()[i] = src_workspace_sizes->GetData()[i];
  }
  new_cache.workspace_sizes_holder = std::move(dst_workspace_sizes_holder);
  new_cache.launch_arg_holder = launch_arg->MakeCopy();
  GE_ASSERT_NOTNULL(new_cache.launch_arg_holder);
  TilingCache *tiling_cache_ptr = tiling_cache_mgr.AddNewCache(key, std::move(new_cache));
  GE_ASSERT_NOTNULL(tiling_cache_ptr);
  const auto cached_launch_args = static_cast<RtKernelLaunchArgsEx *>(tiling_cache_ptr->GetLaunchArgPtr());
  GE_ASSERT(cached_launch_args);
  cached_launch_args->SetTilingCache(tiling_cache_ptr);
  cached_launch_args->SetTilingCacheStatus(RtKernelLaunchArgsEx::TilingCacheStatus::kMissed);
  GE_ASSERT_SUCCESS(RefreshOutputAddr(context, cached_launch_args));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CacheableTilingProc(KernelContext *context, ge::graphStatus &tiling_func_result) {
  auto input_num = context->GetInputNum();
  if (input_num < kFwkDataOffset) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "[TilingCache] para check failed input_num %" PRId64 "", input_num);
    return ge::GRAPH_FAILED;
  }
  const auto cacheable_fwk_data = context->MutableInputPointer<CacheableTilingFwkData>(input_num - kFwkDataOffset);
  GE_ASSERT_NOTNULL(cacheable_fwk_data);
  GE_ASSERT_NOTNULL(cacheable_fwk_data->fwk_data.launch_arg);
  auto &tiling_cache_mgr = cacheable_fwk_data->tiling_cache_mgr;
  auto func_name = cacheable_fwk_data->build_tiling_cache_key_func_name;
  GE_ASSERT_NOTNULL(func_name);

  auto build_tiling_cache_key_func = FindBuildSymbolTilingCacheKeyFunc(std::string(func_name));
  GE_ASSERT_NOTNULL(build_tiling_cache_key_func);
  HashBuffer hash_buf;
  GE_ASSERT_SUCCESS(build_tiling_cache_key_func(context, hash_buf));
  const auto cache_key = hash_buf.GetTilingCacheKey();
  const TilingCache *tiling_cache = tiling_cache_mgr.TryFetchCache(cache_key);

  if (tiling_cache != nullptr) {
    GE_ASSERT_SUCCESS(ApplyTilingCache(context, tiling_cache->GetTilingCacheValue()));
    tiling_func_result = ge::GRAPH_SUCCESS;
  } else {
    const auto tiling_func = reinterpret_cast<KernelRegistry::KernelFunc>(cacheable_fwk_data->fwk_data.tiling_func);
    GE_ASSERT_NOTNULL(tiling_func);
    // refresh tiling_data output addr before calling tiling_func
    GE_ASSERT_SUCCESS(RefreshOutputAddr(context, cacheable_fwk_data->fwk_data.launch_arg));
    const auto tiling_data = reinterpret_cast<TilingContext *>(context)->GetRawTilingData();
    if (tiling_data != nullptr) {
      tiling_data->SetDataSize(0);
    }
    tiling_func_result = tiling_func(context);
    if (tiling_func_result == ge::GRAPH_SUCCESS) {
      GE_ASSERT_SUCCESS(AlignWorkspaceSizes(context));
      GE_ASSERT_SUCCESS(AddTilingCache(context, hash_buf.GetTilingCacheKey(), *cacheable_fwk_data));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CacheableTiling(KernelContext *context) {
  ge::graphStatus tiling_func_result;
  GE_ASSERT_SUCCESS(CacheableTilingProc(context, tiling_func_result));
  return tiling_func_result;
}
REGISTER_KERNEL(CacheableTiling)
    .RunFunc(CacheableTiling)
    .OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo)
    .TracePrinter(PrintTilingData);

ge::graphStatus FallibleTiling(KernelContext *context) {
  auto result = context->GetOutputPointer<uint32_t>(static_cast<size_t>(FallibleTilingExOutputIndex::kTilingStatus));
  if (result == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::graphStatus op_tiling_result;
  GE_ASSERT_SUCCESS(TilingProc(context, op_tiling_result));
  *result = op_tiling_result == ge::GRAPH_SUCCESS ? 0U : 1U;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FallibleTiling)
    .RunFunc(FallibleTiling)
    .OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo)
    .TracePrinter(PrintTilingData);

ge::graphStatus CacheableFallibleTiling(KernelContext *context) {
  auto result = context->GetOutputPointer<uint32_t>(static_cast<size_t>(FallibleTilingExOutputIndex::kTilingStatus));
  if (result == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::graphStatus op_tiling_result;
  GE_ASSERT_SUCCESS(CacheableTilingProc(context, op_tiling_result));
  *result = op_tiling_result == ge::GRAPH_SUCCESS ? 0U : 1U;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CacheableFallibleTiling)
    .RunFunc(CacheableFallibleTiling)
    .OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo)
    .TracePrinter(PrintTilingData);

ge::graphStatus BuildTilingOutputsAppendDfxInfo(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto extend_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  const size_t input_num = extend_kernel_context->GetComputeNodeInputNum();
  const size_t output_num = extend_kernel_context->GetComputeNodeOutputNum();

  auto input_chain = context->GetInput(input_num + output_num * 2 + 1UL);
  GE_ASSERT_NOTNULL(input_chain);
  auto output_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(output_chain);
  output_chain->Set(input_chain->GetValue<void *>(), nullptr);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FormatArgsException(KernelContext *context, const size_t fixed_input_start_idx,
                                    const RtKernelLaunchArgsEx *launch_arg, uint64_t &atomic_index) {
  const auto workspace =
      context->GetInputPointer<TypedContinuousVector<int64_t>>(fixed_input_start_idx + kWorkSpaceOffset);
  GE_ASSERT_NOTNULL(workspace);
  const auto args_sizes = context->GetInputPointer<ContinuousVector>(fixed_input_start_idx + kArgsSizeListOffset);
  GE_ASSERT_NOTNULL(args_sizes);
  const auto args_idx_to_io_idx = context->GetInputPointer<TypedContinuousVector<optiling::ArgsIndexToIoIndex>>(
      fixed_input_start_idx + kArgsIndexToIoIndexOffset);
  GE_ASSERT_NOTNULL(args_idx_to_io_idx);
  const size_t args_size_num = args_sizes->GetSize();
  const size_t workspace_num = workspace->GetSize();
  const size_t shape_size_num = DfxCalcShapeSizeNum(context, *args_idx_to_io_idx);
  const size_t total_num = args_size_num + workspace_num + shape_size_num;
  // 注意此处的total_num表示size个数(类型为uint64_t)
  if (total_num == 0U) {
    return ge::GRAPH_SUCCESS;
  }

  const auto &arg_cache_info = launch_arg->GetArgsCacheInfo();
  if (arg_cache_info.cache_status == RtKernelLaunchArgsEx::TilingCacheStatus::kHit) {
    GE_ASSERT_NOTNULL(arg_cache_info.tiling_cache);
    return DfxGetDumpDataFromCache(static_cast<const TilingCache *>(arg_cache_info.tiling_cache), atomic_index);
  }

  uint64_t *dump_addr = ge::PtrToPtr<void, uint64_t>(Adx::AdumpGetDFXInfoAddrForDynamic(total_num, atomic_index));
  uint64_t *curr_addr = dump_addr;
  GE_ASSERT_NOTNULL(curr_addr, "total num[%zu]", total_num);

  errno_t ret = EOK;
  if (args_size_num > 0U) {
    ret = memcpy_s(curr_addr, args_size_num * sizeof(int64_t), args_sizes->GetData(), args_size_num * sizeof(int64_t));
    GE_ASSERT_EOK(ret, "[TilingAppendDfxInfo] copy args size failed ret[%d]", ret);
    curr_addr += args_size_num;
  }

  if (workspace_num > 0U) {
    ret = memcpy_s(curr_addr, workspace_num * sizeof(int64_t), workspace->GetData(), workspace_num * sizeof(int64_t));
    GE_ASSERT_EOK(ret, "[TilingAppendDfxInfo] copy workspace size failed ret[%d]", ret);
    curr_addr += workspace_num;
  }

  GE_ASSERT_SUCCESS(DfxUpdateTensorSizeAndShape(context, dump_addr, args_size_num, workspace_num, *args_idx_to_io_idx));

  if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    const auto &tiling_data = launch_arg->GetTilingData();
    GELOGI("[TilingAppendDfxInfo] capacity: %zu, data size: %zu, args size: %zu, workspace size: %zu, shape size: %zu",
           tiling_data.GetCapacity(), tiling_data.GetDataSize(), args_size_num, workspace_num, shape_size_num);
    for (size_t i = 0U; i < (args_size_num + workspace_num); i++) {
      GELOGI("[TilingAppendDfxInfo] size idx[%zu], val[%llu], atomic index[%llu]", i, *(dump_addr + i), atomic_index);
    }
    for (size_t i = (args_size_num + workspace_num); i < total_num; i++) {
      GELOGI("[TilingAppendDfxInfo] shape idx[%zu], val[%llu], atomic index[%llu]", i, *(dump_addr + i), atomic_index);
    }
  }

  if (arg_cache_info.cache_status == RtKernelLaunchArgsEx::TilingCacheStatus::kMissed) {
    GE_ASSERT_NOTNULL(arg_cache_info.tiling_cache);
    // 为优化性能, 算子第二次执行(即命中缓存), 不再上报Shape信息, 因此dump_data_num只包含arg和workspace
    GE_ASSERT_SUCCESS(DfxAddDumpDataToCache(static_cast<TilingCache *>(arg_cache_info.tiling_cache), dump_addr,
                                            args_size_num + workspace_num));
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAppendMemcheckInfo(KernelContext *context, const size_t fixed_input_start_idx,
                                         TilingData &tiling_data) {
  const auto workspace =
      context->GetInputPointer<TypedContinuousVector<int64_t>>(fixed_input_start_idx + kWorkSpaceOffset);
  GE_ASSERT_NOTNULL(workspace);
  const auto args_sizes = context->GetInputPointer<ContinuousVector>(fixed_input_start_idx + kArgsSizeListOffset);
  GE_ASSERT_NOTNULL(args_sizes);
  const auto args_idx_to_io_idx = context->GetInputPointer<TypedContinuousVector<optiling::ArgsIndexToIoIndex>>(
      fixed_input_start_idx + kArgsIndexToIoIndexOffset);
  GE_ASSERT_NOTNULL(args_idx_to_io_idx);
  const auto ori_param_size = context->GetInputValue<size_t>(fixed_input_start_idx + kOriOpParamSizeOffset);

  if (ori_param_size > 0UL) {
    // tik场景下TilingAppendMem添加的数据需要从偏移为ori_param_size的地址开始添加，此处需要将DataSize设置成ori_param_size
    GELOGI("[TilingAppendMemCheck]Tiling data need append memcheck data in offset:%zu", ori_param_size);
    GE_ASSERT_TRUE(tiling_data.GetDataSize() <= ori_param_size);
    tiling_data.SetDataSize(ori_param_size);
  } else {
    const auto aligned_size = ((tiling_data.GetDataSize() + sizeof(int64_t) - 1LL) / sizeof(int64_t)) * sizeof(int64_t);
    GELOGI("[TilingAppendMemCheck]Tiling data need append memcheck data in aligned offset:%zu", aligned_size);
    tiling_data.SetDataSize(aligned_size);
  }

  size_t args_size_num = args_sizes->GetSize();
  int64_t *append_size_ptr = ge::PtrToPtr<void, int64_t>(tiling_data.Expand(args_size_num * sizeof(int64_t)));
  GE_ASSERT_NOTNULL(append_size_ptr);

  if (args_size_num > 0U) {
    errno_t ret = memcpy_s(append_size_ptr, args_size_num * sizeof(int64_t), args_sizes->GetData(),
                           args_size_num * sizeof(int64_t));
    GE_ASSERT_EOK(ret, "[TilingAppendDfxInfo] copy args size failed ret[%d]", ret);
  }

  GE_ASSERT_SUCCESS(DfxUpdateMemcheckTensorSize(context, *args_idx_to_io_idx, args_size_num, append_size_ptr));
  GE_ASSERT_SUCCESS(DfxUpdateMemcheckWorkspaceSize(*workspace, tiling_data));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAppendAtomicIndex(TilingData &tiling_data, const int64_t &atomic_index) {
  int64_t *append_atomic_index_ptr = ge::PtrToPtr<void, int64_t>(tiling_data.Expand(sizeof(int64_t)));
  GE_ASSERT_NOTNULL(append_atomic_index_ptr, "Append atomic_index failed, tilingData size=%zu, capacity=%zu",
                    tiling_data.GetDataSize(), tiling_data.GetCapacity());
  append_atomic_index_ptr[0] = atomic_index;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAppendDfxInfo(KernelContext *context) {
  auto extend_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  // input_shapes|output_shapes|output_sizes|...
  const size_t fixed_input_start_idx =
      extend_kernel_context->GetComputeNodeInputNum() + extend_kernel_context->GetComputeNodeOutputNum() * 2UL;

  const auto launch_arg = context->MutableInputPointer<RtKernelLaunchArgsEx>(fixed_input_start_idx + kLaunchArgOffset);
  GE_ASSERT_NOTNULL(launch_arg);
  auto &tiling_data = launch_arg->GetTilingData();
  const auto is_mem_check_enable = context->GetInputValue<size_t>(fixed_input_start_idx + kIsMemCheckEnableOffset);
  const auto is_args_exception_enable = context->GetInputValue<size_t>(fixed_input_start_idx + kIsArgsExceptionOffset);

  uint64_t atomic_index = 0UL;
  if (is_args_exception_enable) {
    GE_ASSERT_SUCCESS(FormatArgsException(context, fixed_input_start_idx, launch_arg, atomic_index));
  }

  if (is_mem_check_enable) {
    GE_ASSERT_SUCCESS(TilingAppendMemcheckInfo(context, fixed_input_start_idx, tiling_data));
  }

  if (is_args_exception_enable) {
    GE_ASSERT_SUCCESS(TilingAppendAtomicIndex(tiling_data, atomic_index));
  }

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(TilingAppendDfxInfo).RunFunc(TilingAppendDfxInfo).OutputsCreator(BuildTilingOutputsAppendDfxInfo);

ge::graphStatus InitTilingFwkData(KernelContext *context, TilingFwkData &fwk_data) {
  const auto tiling_func = context->GetInputValue<void *>(static_cast<size_t>(TilingFwkDataInput::kTilingFunc));
  GE_ASSERT_NOTNULL(tiling_func);
  const auto raw_args =
      context->MutableInputPointer<RtKernelLaunchArgsEx>(static_cast<size_t>(TilingFwkDataInput::kRtArg));
  GE_ASSERT_NOTNULL(raw_args);
  fwk_data.tiling_func = tiling_func;
  fwk_data.launch_arg = raw_args;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareTilingFwkData(KernelContext *context) {
  const auto tiling_fwk_data = context->GetOutputPointer<TilingFwkData>(0UL);
  GE_ASSERT_NOTNULL(tiling_fwk_data);
  GE_ASSERT_SUCCESS(InitTilingFwkData(context, *tiling_fwk_data));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildTilingFwkDataOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto fwk_data_av = context->GetOutput(0UL);
  GE_ASSERT_NOTNULL(fwk_data_av);
  auto fwk_data_ptr = new (std::nothrow) TilingFwkData{nullptr, nullptr};
  GE_ASSERT_NOTNULL(fwk_data_ptr);
  fwk_data_av->SetWithDefaultDeleter(fwk_data_ptr);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(PrepareTilingFwkData).RunFunc(PrepareTilingFwkData).OutputsCreator(BuildTilingFwkDataOutput);

ge::graphStatus PrepareCacheableTilingFwkData(KernelContext *context) {
  const auto tiling_fwk_data = context->GetOutputPointer<CacheableTilingFwkData>(0UL);
  GE_ASSERT_NOTNULL(tiling_fwk_data);
  GE_ASSERT_SUCCESS(InitTilingFwkData(context, tiling_fwk_data->fwk_data));
  const auto data_dependency =
      context->GetInputValue<size_t>(static_cast<size_t>(CacheableTilingFwkDataInput::kDataDependency));
  tiling_fwk_data->data_dependency = data_dependency;
  const auto build_tiling_cache_key_func_name = context->GetInputValue<char *>(
      static_cast<size_t>(CacheableTilingFwkDataInput::kBuildTilingCacheKeyFuncName));
  tiling_fwk_data->build_tiling_cache_key_func_name = build_tiling_cache_key_func_name;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildCacheableTilingFwkDataOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto fwk_data_av = context->GetOutput(0UL);
  GE_ASSERT_NOTNULL(fwk_data_av);
  std::unique_ptr<TilingCacheStrategy> tiling_cache_strategy(
      new (std::nothrow) TilingCacheLruStrategy(kTilingCacheSizePerOp, kTilingCacheEvictNum));
  GE_ASSERT_NOTNULL(tiling_cache_strategy);
  auto fwk_data_ptr = new (std::nothrow)CacheableTilingFwkData{
      {nullptr, nullptr},
      TilingCacheManager(std::move(tiling_cache_strategy)),
      0UL,
      nullptr};
  GE_ASSERT_NOTNULL(fwk_data_ptr);
  fwk_data_av->SetWithDefaultDeleter(fwk_data_ptr);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(PrepareCacheableTilingFwkData)
    .RunFunc(PrepareCacheableTilingFwkData)
    .OutputsCreator(BuildCacheableTilingFwkDataOutput);
}  // namespace kernel

}  // namespace gert
