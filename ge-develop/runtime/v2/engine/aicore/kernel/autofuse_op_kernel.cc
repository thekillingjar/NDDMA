/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_op_kernel.h"

#include "aclnn_op_execute_kernel.h"
#include "common/checker.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "mmpa/mmpa_api.h"
#include "kernel/kernel_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "engine/aicore/converter/autofuse_node_converter.h"

namespace gert {
namespace kernel {
namespace {

class TilingSymbolEvalContext;
class SymbolTilingParseContext;

using GetTilingCacheKey = ge::graphStatus(*)(TilingSymbolEvalContext *);
using TilingParse = ge::graphStatus(*)(SymbolTilingParseContext *);
}

ge::graphStatus DlcloseSoHandles(KernelContext *context) {
  auto so_handle = context->GetInputValue<void *>(0);
  if (so_handle != nullptr) {
    (void)mmDlclose(so_handle);
  }
  return ge::GRAPH_SUCCESS;
}

// input: platform, TilingParseFunc
// output: TilingParseData
ge::graphStatus SymbolTilingParseKernel(KernelContext *context) {
  auto tiling_parser_func = context->GetInputValue<TilingParse>(1U);
  GE_ASSERT_NOTNULL(tiling_parser_func);
  GE_ASSERT_SUCCESS(tiling_parser_func(reinterpret_cast<SymbolTilingParseContext *>(context)));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAutofuseFuncsKernel(KernelContext *context) {
  auto so_path = context->GetInputValue<char *>(0);
  GE_ASSERT_NOTNULL(so_path);
  auto so_handle_av = context->GetOutput(static_cast<size_t>(GetAutofuseFuncsOutput::kAutofuseSo));
  GE_ASSERT_NOTNULL(so_handle_av);
  auto tiling_func_handle =
      context->GetOutputPointer<void *>(static_cast<size_t>(GetAutofuseFuncsOutput::kTilingFunc));
  GE_ASSERT_NOTNULL(tiling_func_handle);
  auto infer_sahpe_handle =
      context->GetOutputPointer<void *>(static_cast<size_t>(GetAutofuseFuncsOutput::kInferShape));
  GE_ASSERT_NOTNULL(infer_sahpe_handle);
  auto get_tiling_cache_key_handle =
      context->GetOutputPointer<void *>(static_cast<size_t>(GetAutofuseFuncsOutput::kGetTilingCacheKey));
  GE_ASSERT_NOTNULL(get_tiling_cache_key_handle);
  auto tiling_parse_handle =
      context->GetOutputPointer<void *>(static_cast<size_t>(GetAutofuseFuncsOutput::kTilingParse));
  GE_ASSERT_NOTNULL(tiling_parse_handle);
  auto dfx_input_symbol_info_handle =
      context->GetOutputPointer<void *>(static_cast<size_t>(GetAutofuseFuncsOutput::kDfxInputSymbolInfo));
  GE_ASSERT_NOTNULL(dfx_input_symbol_info_handle);

  char real_path[MMPA_MAX_PATH] = {};
  GE_ASSERT_TRUE(mmRealPath(so_path, &real_path[0], MMPA_MAX_PATH) == EN_OK);
  KLOGI("Get so path: %s", real_path);
  auto handle = mmDlopen(real_path, static_cast<int32_t>(MMPA_RTLD_NOW));
  GE_ASSERT_NOTNULL(handle);
  so_handle_av->Set(handle, [](void* h) {
    (void)mmDlclose(h);
  });

  auto tiling_func = mmDlsym(handle, "TilingFunc");
  GE_ASSERT_NOTNULL(tiling_func);
  *tiling_func_handle = tiling_func;

  auto infer_shape = mmDlsym(handle, "InferShape");
  GE_ASSERT_NOTNULL(infer_shape);
  *infer_sahpe_handle = infer_shape;

  auto get_tiling_cache_key = mmDlsym(handle, "GetSymbolTilingCacheKey");
  GE_ASSERT_NOTNULL(get_tiling_cache_key);
  *get_tiling_cache_key_handle = get_tiling_cache_key;

  auto tiling_parse = mmDlsym(handle, "TilingParse");
  GE_ASSERT_NOTNULL(tiling_parse);
  *tiling_parse_handle = tiling_parse;

  auto dfx_input_symbol_info = mmDlsym(handle, "DfxInputSymbolInfo");
  GE_ASSERT_NOTNULL(dfx_input_symbol_info);
  *dfx_input_symbol_info_handle = dfx_input_symbol_info;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetSymbolTilingCacheKeyKernel(KernelContext *context) {
  auto input_data_num = context->GetInputValue<size_t>(0U);
  auto get_tiling_cache_key_func = context->GetInputValue<GetTilingCacheKey>(
      input_data_num + static_cast<size_t>(GetSymbolTilingCacheKeyInput::kGetSymbolTilingCacheKeyFunc));
  GE_ASSERT_NOTNULL(get_tiling_cache_key_func);
  GE_ASSERT_SUCCESS(get_tiling_cache_key_func(reinterpret_cast<TilingSymbolEvalContext *>(context)));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildSymbolTilingCacheKeyOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto input_data_num = context->GetInputValue<size_t>(0U);
  auto all_symbol_num = context->GetInputValue<size_t>(
      input_data_num + static_cast<size_t>(GetSymbolTilingCacheKeyInput::kAllSymbolNum));
  auto used_sym_src_av = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(used_sym_src_av);
  auto used_sym_src_holder = ContinuousVector::Create<int64_t>(all_symbol_num);
  GE_ASSERT_NOTNULL(used_sym_src_holder);
  used_sym_src_av->SetWithDefaultDeleter<uint8_t[]>(used_sym_src_holder.release());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(DlcloseSoHandles).RunFunc(DlcloseSoHandles);
REGISTER_KERNEL(SymbolTilingParse).RunFunc(SymbolTilingParseKernel);
REGISTER_KERNEL(GetAutofuseFuncs).RunFunc(GetAutofuseFuncsKernel);
REGISTER_KERNEL(GetSymbolTilingCacheKey)
    .RunFunc(GetSymbolTilingCacheKeyKernel)
    .OutputsCreator(BuildSymbolTilingCacheKeyOutputs);
}
}